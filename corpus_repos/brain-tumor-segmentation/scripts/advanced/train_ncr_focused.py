#!/usr/bin/env python3
"""
NCR-Focused Training - Simple weighted loss approach
Addresses the NCR prediction failures identified in analysis
"""

import sys
import os
sys.path.insert(0, os.path.abspath('../'))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import json
from pathlib import Path
import time
import numpy as np

print("="*80)
print("NCR-FOCUSED ATTENTION U-NET TRAINING")
print("="*80)

# Import modules
from create_dataset import BraTSSegmentationDataset
from augmentation_safe import SafeMedicalAugmentation
from attention_unet_v2_improved import ImprovedAttentionUNet3D
from metrics import SegmentationMetrics


class NCRWeightedComboLoss(nn.Module):
    """
    Weighted loss that emphasizes NCR prediction
    NCR gets 5x weight, ET gets 2x weight, ED gets 1x weight
    """
    def __init__(self):
        super().__init__()
        # Class weights: [background, NCR, ED, ET]
        # We care most about NCR (5x), then ET (2x), then ED (1x)
        self.class_weights = torch.tensor([0.5, 5.0, 1.0, 2.0])
        
    def dice_loss_weighted(self, pred, target, num_classes=4):
        """Compute weighted dice loss"""
        smooth = 1e-5
        total_loss = 0.0
        
        # Convert predictions to probabilities
        pred = torch.softmax(pred, dim=1)
        
        # One-hot encode targets
        target_one_hot = torch.zeros_like(pred)
        
        # Map BraTS labels (0,1,2,4) to class indices (0,1,2,3)
        target_mapped = target.clone()
        target_mapped[target == 4] = 3  # ET: label 4 -> class 3
        
        target_one_hot.scatter_(1, target_mapped.unsqueeze(1).long(), 1)
        
        # Compute dice for each class with weights
        for c in range(num_classes):
            pred_c = pred[:, c]
            target_c = target_one_hot[:, c]
            
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()
            
            dice = (2. * intersection + smooth) / (union + smooth)
            loss = 1 - dice
            
            # Apply class weight
            weight = self.class_weights[c].to(pred.device)
            total_loss += weight * loss
        
        return total_loss / self.class_weights.sum()
    
    def focal_loss(self, pred, target, alpha=0.25, gamma=2.0):
        """Focal loss to handle class imbalance"""
        # Map targets
        target_mapped = target.clone()
        target_mapped[target == 4] = 3
        
        ce_loss = nn.functional.cross_entropy(pred, target_mapped.long(), reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        
        return focal_loss.mean()
    
    def forward(self, pred, target):
        """Combined loss"""
        dice = self.dice_loss_weighted(pred, target)
        focal = self.focal_loss(pred, target)
        
        return 0.7 * dice + 0.3 * focal


# [Rest is same as train_attention_ultimate.py, but using NCRWeightedComboLoss]

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_epoch = 0
    
    def step(self):
        if self.current_epoch < self.warmup_epochs:
            lr = self.base_lr * (self.current_epoch + 1) / self.warmup_epochs
        else:
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.current_epoch += 1
        return lr


def train_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch):
    model.train()
    total_loss = 0.0
    
    for batch_idx, (images, masks) in enumerate(train_loader):
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(images)
            
            # Handle deep supervision
            if isinstance(outputs, tuple):
                main_output = outputs[0]
            else:
                main_output = outputs
            
            loss = criterion(main_output, masks)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        
        if (batch_idx + 1) % 50 == 0:
            print(f"  Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")
    
    return total_loss / len(train_loader)


def validate_epoch(model, val_loader, criterion, device, metrics_calc):
    model.eval()
    total_loss = 0.0
    wt_sum = tc_sum = et_sum = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            with autocast():
                outputs = model(images)
                if isinstance(outputs, tuple):
                    main_output = outputs[0]
                else:
                    main_output = outputs
                loss = criterion(main_output, masks)
            
            total_loss += loss.item()
            preds = torch.argmax(main_output, dim=1)
            
            for i in range(preds.shape[0]):
                pred_np = preds[i].cpu().numpy()
                mask_np = masks[i].cpu().numpy()
                
                wt_sum += metrics_calc.whole_tumor_dice(pred_np, mask_np)
                tc_sum += metrics_calc.tumor_core_dice(pred_np, mask_np)
                et_sum += metrics_calc.enhancing_tumor_dice(pred_np, mask_np)
                num_samples += 1
    
    return (total_loss / len(val_loader), 
            wt_sum / num_samples, 
            tc_sum / num_samples, 
            et_sum / num_samples)


def main():
    config = {
        'batch_size': 2,
        'num_epochs': 100,
        'base_lr': 0.0005,
        'min_lr': 1e-6,
        'warmup_epochs': 5,
        'num_workers': 1,
        'save_every': 5,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print("\n[CONFIG] NCR-Focused Training")
    print(f"  NCR weight: 5x (highest priority)")
    print(f"  ET weight: 2x")
    print(f"  ED weight: 1x")
    print(f"  Loss: 70% Weighted Dice + 30% Focal")
    
    # Directories
    checkpoint_dir = Path('../../models/checkpoints/v2/ncr_focused')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    with open('../../data/splits/data_splits.json') as f:
        splits = json.load(f)
    
    data_root = Path('../../data/raw/brats_gli/training_data1_v2')
    train_dirs = [data_root / pid for pid in splits['train']]
    val_dirs = [data_root / pid for pid in splits['val']]
    
    print(f"\n[DATA] Train: {len(train_dirs)}, Val: {len(val_dirs)}")
    
    # Datasets
    augmentation = SafeMedicalAugmentation()
    train_dataset = BraTSSegmentationDataset(train_dirs, augmentation, (128,128,128))
    val_dataset = BraTSSegmentationDataset(val_dirs, None, (128,128,128))
    
    train_loader = DataLoader(train_dataset, config['batch_size'], shuffle=True, 
                             num_workers=config['num_workers'], pin_memory=True)
    val_loader = DataLoader(val_dataset, config['batch_size'], shuffle=False,
                           num_workers=config['num_workers'], pin_memory=True)
    
    # Model
    model = ImprovedAttentionUNet3D(4, 4, 32, deep_supervision=True)
    model = model.to(config['device'])
    print(f"[MODEL] Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # NCR-weighted loss
    criterion = NCRWeightedComboLoss().to(config['device'])
    
    # Optimizer & scheduler
    optimizer = optim.AdamW(model.parameters(), lr=config['base_lr'], weight_decay=1e-4)
    scheduler = WarmupCosineScheduler(optimizer, config['warmup_epochs'], 
                                     config['num_epochs'], config['base_lr'])
    scaler = GradScaler()
    metrics_calc = SegmentationMetrics(4)
    
    # Training
    best_wt = 0.0
    history = {'train_loss': [], 'val_wt': [], 'val_tc': [], 'val_et': []}
    
    print("\n[TRAINING] Starting...")
    print("="*80)
    
    for epoch in range(1, config['num_epochs'] + 1):
        print(f"\nEpoch {epoch}/{config['num_epochs']}")
        
        lr = scheduler.step()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, config['device'], epoch)
        val_loss, val_wt, val_tc, val_et = validate_epoch(model, val_loader, criterion, config['device'], metrics_calc)
        
        history['train_loss'].append(train_loss)
        history['val_wt'].append(val_wt)
        history['val_tc'].append(val_tc)
        history['val_et'].append(val_et)
        
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val WT: {val_wt:.4f}, TC: {val_tc:.4f}, ET: {val_et:.4f}")
        print(f"  LR: {lr:.6f}")
        
        if val_wt > best_wt:
            best_wt = val_wt
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_wt': best_wt,
                'val_tc': val_tc,
                'config': config
            }, checkpoint_dir / 'best_model.pth')
            print(f"  ✓ NEW BEST: {best_wt:.4f}")
    
    print("\n" + "="*80)
    print(f"TRAINING COMPLETE! Best WT Dice: {best_wt:.4f}")
    print("="*80)


if __name__ == "__main__":
    main()
