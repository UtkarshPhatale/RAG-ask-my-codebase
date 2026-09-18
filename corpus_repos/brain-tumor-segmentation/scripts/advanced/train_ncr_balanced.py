#!/usr/bin/env python3
"""
BALANCED NCR WEIGHT EXPERIMENT
===============================
Goal: Improve TC without destroying WT
Strategy: NCR 2.5× (not 5×), ET 1.5×, ED 1.0×

Previous NCR 5× results:
- WT: 0.7905 (↓3.76% from baseline)
- TC: 0.7754 (↑30.8% from baseline!)
- ET: 0.7126

Hypothesis: 2.5× NCR weight will give:
- WT: ~0.82 (only ↓0.5%)
- TC: ~0.71 (↑20%)
- NET IMPROVEMENT in combined score
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
from datetime import datetime

print("="*80)
print("BALANCED NCR WEIGHT EXPERIMENT")
print("="*80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

# Import modules
from create_dataset import BraTSSegmentationDataset
from augmentation_safe import SafeMedicalAugmentation
from attention_unet_v2_improved import ImprovedAttentionUNet3D
from metrics import SegmentationMetrics


class BalancedNCRLoss(nn.Module):
    """
    Balanced NCR weighting - less aggressive than 5×
    Weights: [BG: 0.5, NCR: 2.5, ED: 1.0, ET: 1.5]
    """
    def __init__(self):
        super().__init__()
        # BALANCED weights (not aggressive 5×)
        self.class_weights = torch.tensor([0.5, 2.5, 1.0, 1.5])
        print("\n[LOSS CONFIG]")
        print(f"  Background: {self.class_weights[0]:.1f}×")
        print(f"  NCR:        {self.class_weights[1]:.1f}× (was 5.0× before)")
        print(f"  Edema:      {self.class_weights[2]:.1f}×")
        print(f"  Enhancing:  {self.class_weights[3]:.1f}×")
        
    def dice_loss_weighted(self, pred, target, num_classes=4):
        """Weighted dice loss"""
        smooth = 1e-5
        total_loss = 0.0
        
        pred = torch.softmax(pred, dim=1)
        target_one_hot = torch.zeros_like(pred)
        
        # Map BraTS labels
        target_mapped = target.clone()
        target_mapped[target == 4] = 3
        target_one_hot.scatter_(1, target_mapped.unsqueeze(1).long(), 1)
        
        for c in range(num_classes):
            pred_c = pred[:, c]
            target_c = target_one_hot[:, c]
            
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()
            dice = (2. * intersection + smooth) / (union + smooth)
            loss = 1 - dice
            
            weight = self.class_weights[c].to(pred.device)
            total_loss += weight * loss
        
        return total_loss / self.class_weights.sum()
    
    def focal_loss(self, pred, target, alpha=0.25, gamma=2.0):
        """Focal loss"""
        target_mapped = target.clone()
        target_mapped[target == 4] = 3
        
        ce_loss = nn.functional.cross_entropy(pred, target_mapped.long(), reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        return focal_loss.mean()
    
    def forward(self, pred, target):
        """70% Weighted Dice + 30% Focal"""
        dice = self.dice_loss_weighted(pred, target)
        focal = self.focal_loss(pred, target)
        return 0.7 * dice + 0.3 * focal


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
        'model_name': 'ncr_balanced',
        'ncr_weight': 2.5,  # Key difference from previous 5.0
        'batch_size': 2,
        'num_epochs': 100,
        'base_lr': 0.0005,
        'min_lr': 1e-6,
        'warmup_epochs': 5,
        'num_workers': 1,
        'save_every': 5,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print("\n[1/7] Configuration:")
    print("-" * 80)
    print(f"  Model:       Attention U-Net (32 channels)")
    print(f"  NCR weight:  {config['ncr_weight']}× (BALANCED, not 5×)")
    print(f"  Batch size:  {config['batch_size']}")
    print(f"  Epochs:      {config['num_epochs']}")
    print(f"  Base LR:     {config['base_lr']}")
    print(f"  Device:      {config['device']}")
    
    # Directories
    print("\n[2/7] Creating directories...")
    checkpoint_dir = Path('../../models/checkpoints/v2/ncr_balanced')
    log_dir = Path('../../models/logs/v2/ncr_balanced')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"  ✓ Checkpoints: {checkpoint_dir}")
    print(f"  ✓ Logs:        {log_dir}")
    
    # Load data
    print("\n[3/7] Loading data...")
    with open('../../data/splits/data_splits.json') as f:
        splits = json.load(f)
    
    data_root = Path('../../data/raw/brats_gli/training_data1_v2')
    train_dirs = [data_root / pid for pid in splits['train']]
    val_dirs = [data_root / pid for pid in splits['val']]
    print(f"  Train: {len(train_dirs)} patients")
    print(f"  Val:   {len(val_dirs)} patients")
    
    # Datasets
    print("\n[4/7] Creating datasets...")
    augmentation = SafeMedicalAugmentation()
    train_dataset = BraTSSegmentationDataset(train_dirs, augmentation, (128,128,128))
    val_dataset = BraTSSegmentationDataset(val_dirs, None, (128,128,128))
    
    train_loader = DataLoader(train_dataset, config['batch_size'], shuffle=True,
                             num_workers=config['num_workers'], pin_memory=True)
    val_loader = DataLoader(val_dataset, config['batch_size'], shuffle=False,
                           num_workers=config['num_workers'], pin_memory=True)
    
    print(f"  ✓ Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  ✓ Val:   {len(val_dataset)} samples, {len(val_loader)} batches")
    
    # Model
    print("\n[5/7] Initializing model...")
    model = ImprovedAttentionUNet3D(4, 4, 32, deep_supervision=True)
    model = model.to(config['device'])
    params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ Parameters: {params:,} ({params/1e6:.2f}M)")
    
    # Loss & optimizer
    print("\n[6/7] Setting up training...")
    criterion = BalancedNCRLoss().to(config['device'])
    optimizer = optim.AdamW(model.parameters(), lr=config['base_lr'], weight_decay=1e-4)
    scheduler = WarmupCosineScheduler(optimizer, config['warmup_epochs'], 
                                     config['num_epochs'], config['base_lr'])
    scaler = GradScaler()
    metrics_calc = SegmentationMetrics(4)
    
    print(f"  ✓ Loss:      Balanced NCR (2.5× weight)")
    print(f"  ✓ Optimizer: AdamW")
    print(f"  ✓ Scheduler: Warmup + Cosine")
    
    # Training
    print("\n[7/7] Starting training...")
    print("="*80)
    
    best_wt = 0.0
    best_epoch = 0
    history = {
        'train_loss': [],
        'val_wt': [],
        'val_tc': [],
        'val_et': [],
        'learning_rate': [],
        'config': config
    }
    
    start_time = time.time()
    
    for epoch in range(1, config['num_epochs'] + 1):
        print(f"\nEpoch {epoch}/{config['num_epochs']}")
        print("-" * 80)
        
        lr = scheduler.step()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, config['device'], epoch)
        val_loss, val_wt, val_tc, val_et = validate_epoch(model, val_loader, criterion, config['device'], metrics_calc)
        
        history['train_loss'].append(float(train_loss))
        history['val_wt'].append(float(val_wt))
        history['val_tc'].append(float(val_tc))
        history['val_et'].append(float(val_et))
        history['learning_rate'].append(float(lr))
        
        print(f"\n  Train Loss:  {train_loss:.4f}")
        print(f"  Val WT Dice: {val_wt:.4f}")
        print(f"  Val TC Dice: {val_tc:.4f}")
        print(f"  Val ET Dice: {val_et:.4f}")
        print(f"  LR:          {lr:.6f}")
        
        # Compare to baseline
        baseline_wt = 0.8242
        baseline_tc = 0.5931
        wt_change = ((val_wt - baseline_wt) / baseline_wt) * 100
        tc_change = ((val_tc - baseline_tc) / baseline_tc) * 100
        print(f"  vs Baseline: WT {wt_change:+.2f}%, TC {tc_change:+.2f}%")
        
        if val_wt > best_wt:
            best_wt = val_wt
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_wt_dice': float(best_wt),
                'val_wt_dice': float(val_wt),
                'val_tc_dice': float(val_tc),
                'val_et_dice': float(val_et),
                'config': config,
                'history': history
            }, checkpoint_dir / 'best_model.pth')
            print(f"  ✓ NEW BEST! Saved (WT: {best_wt:.4f})")
        
        if epoch % config['save_every'] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_wt_dice': float(val_wt),
                'val_tc_dice': float(val_tc),
                'val_et_dice': float(val_et),
                'config': config
            }, checkpoint_dir / f'epoch_{epoch:03d}.pth')
        
        # Save history every 10 epochs
        if epoch % 10 == 0:
            with open(checkpoint_dir / 'training_history.json', 'w') as f:
                json.dump(history, f, indent=2)
    
    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"  Total time:   {hours}h {minutes}m")
    print(f"  Best epoch:   {best_epoch}")
    print(f"  Best WT Dice: {best_wt:.4f}")
    
    # Final comparison
    final_wt_change = ((best_wt - 0.8242) / 0.8242) * 100
    final_tc_change = ((val_tc - 0.5931) / 0.5931) * 100
    print(f"\n  vs Baseline (0.8242 WT, 0.5931 TC):")
    print(f"    WT: {final_wt_change:+.2f}%")
    print(f"    TC: {final_tc_change:+.2f}%")
    
    # Save final results
    with open(checkpoint_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    summary = {
        'model_name': config['model_name'],
        'ncr_weight': config['ncr_weight'],
        'best_epoch': int(best_epoch),
        'best_val_wt_dice': float(best_wt),
        'baseline_wt_dice': 0.8242,
        'baseline_tc_dice': 0.5931,
        'wt_improvement_pct': float(final_wt_change),
        'tc_improvement_pct': float(final_tc_change),
        'training_time_hours': float(hours + minutes/60),
        'config': config
    }
    
    with open(checkpoint_dir / 'training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n  ✓ Results saved to {checkpoint_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
