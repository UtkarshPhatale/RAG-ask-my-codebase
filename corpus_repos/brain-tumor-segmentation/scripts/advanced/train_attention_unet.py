"""
Train Attention U-Net - Week 1 Improvement (ROBUST VERSION)
===========================================================

This version includes better error handling and debugging output.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('../'))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
from pathlib import Path
import time
import traceback

print("="*80)
print("ATTENTION U-NET TRAINING - WEEK 1 IMPROVEMENT")
print("="*80)

try:
    # Import your existing code
    print("\n[DEBUG] Importing create_dataset...")
    from create_dataset import BraTSSegmentationDataset
    print("  ✓ Successfully imported BraTSSegmentationDataset")
    
    print("[DEBUG] Importing augmentation_safe...")
    from augmentation_safe import SafeMedicalAugmentation
    print("  ✓ Successfully imported SafeMedicalAugmentation")
    
    # Import new attention architecture
    print("[DEBUG] Importing attention_unet...")
    from attention_unet import AttentionUNet3D
    print("  ✓ Successfully imported AttentionUNet3D")
    
    # Import enhanced training tools
    print("[DEBUG] Importing metrics...")
    from metrics import SegmentationMetrics
    print("  ✓ Successfully imported SegmentationMetrics")
    
except Exception as e:
    print(f"\n✗ Import failed: {e}")
    traceback.print_exc()
    sys.exit(1)


def dice_loss(pred, target, smooth=1.0):
    """
    Dice Loss for segmentation
    
    Args:
        pred: Predictions (B, C, H, W, D) - logits
        target: Ground truth (B, H, W, D) - class indices
        smooth: Smoothing factor
    
    Returns:
        Dice loss (scalar)
    """
    num_classes = pred.shape[1]
    
    # Convert predictions to probabilities
    pred = torch.softmax(pred, dim=1)
    
    # One-hot encode target
    target_one_hot = torch.nn.functional.one_hot(target, num_classes)
    target_one_hot = target_one_hot.permute(0, 4, 1, 2, 3).float()
    
    # Calculate Dice for each class
    dice = 0.0
    for c in range(num_classes):
        pred_c = pred[:, c]
        target_c = target_one_hot[:, c]
        
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        
        dice += (2.0 * intersection + smooth) / (union + smooth)
    
    # Average over classes and return loss
    dice = dice / num_classes
    return 1.0 - dice


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (images, masks) in enumerate(train_loader):
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Print progress
        if (batch_idx + 1) % 50 == 0:
            print(f"  Epoch {epoch}, Batch {batch_idx + 1}/{len(train_loader)}, "
                  f"Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / num_batches
    return avg_loss


def validate_epoch(model, val_loader, criterion, device, metrics_calc):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    # Dice scores
    wt_dice_sum = 0.0
    tc_dice_sum = 0.0
    et_dice_sum = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            total_loss += loss.item()
            num_batches += 1
            
            # Calculate Dice scores
            preds = torch.argmax(outputs, dim=1)
            
            for i in range(preds.shape[0]):
                pred_np = preds[i].cpu().numpy()
                mask_np = masks[i].cpu().numpy()
                
                wt_dice = metrics_calc.whole_tumor_dice(pred_np, mask_np)
                tc_dice = metrics_calc.tumor_core_dice(pred_np, mask_np)
                et_dice = metrics_calc.enhancing_tumor_dice(pred_np, mask_np)
                
                wt_dice_sum += wt_dice
                tc_dice_sum += tc_dice
                et_dice_sum += et_dice
                num_samples += 1
    
    avg_loss = total_loss / num_batches
    avg_wt_dice = wt_dice_sum / num_samples
    avg_tc_dice = tc_dice_sum / num_samples
    avg_et_dice = et_dice_sum / num_samples
    
    return avg_loss, avg_wt_dice, avg_tc_dice, avg_et_dice


def main():
    try:
        # Configuration
        config = {
            'batch_size': 2,
            'num_epochs': 75,
            'learning_rate': 0.0001,
            'num_workers': 1,
            'save_every': 5,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
        
        print("\n[1/7] Configuration:")
        print(f"  Batch size: {config['batch_size']}")
        print(f"  Epochs: {config['num_epochs']}")
        print(f"  Learning rate: {config['learning_rate']}")
        print(f"  Device: {config['device']}")
        
        # Create directories
        print("\n[2/7] Creating directories...")
        checkpoint_dir = Path('../../models/checkpoints/v2/attention_unet')
        log_dir = Path('../../models/logs/v2/attention_unet')
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ Checkpoint dir: {checkpoint_dir}")
        print(f"  ✓ Log dir: {log_dir}")
        
        # Load data splits
        print("\n[3/7] Loading data splits...")
        splits_path = '../../data/splits/data_splits.json'
        print(f"  Reading: {splits_path}")
        
        with open(splits_path, 'r') as f:
            splits = json.load(f)
        
        print(f"  ✓ Train: {len(splits['train'])} patients")
        print(f"  ✓ Val: {len(splits['val'])} patients")
        
        # Setup data paths
        print("\n[4/7] Setting up data paths...")
        data_root = Path('../../data/raw/brats_gli/training_data1_v2')
        print(f"  Data root: {data_root}")
        print(f"  Data root exists: {data_root.exists()}")
        
        if not data_root.exists():
            print(f"  ✗ ERROR: Data root does not exist!")
            sys.exit(1)
        
        # Create patient directory lists (as Path objects)
        print("  Creating train patient directories...")
        train_dirs = []
        for pid in splits['train']:
            pdir = data_root / pid
            if pdir.exists():
                train_dirs.append(pdir)
            else:
                print(f"    ! Warning: {pdir} not found")
        
        print("  Creating val patient directories...")
        val_dirs = []
        for pid in splits['val']:
            pdir = data_root / pid
            if pdir.exists():
                val_dirs.append(pdir)
            else:
                print(f"    ! Warning: {pdir} not found")
        
        print(f"  ✓ Train directories: {len(train_dirs)} (out of {len(splits['train'])})")
        print(f"  ✓ Val directories: {len(val_dirs)} (out of {len(splits['val'])})")
        
        if len(train_dirs) == 0 or len(val_dirs) == 0:
            print("  ✗ ERROR: No patient directories found!")
            sys.exit(1)
        
        # Create datasets
        print("\n[5/7] Creating datasets...")
        
        print("  Creating augmentation...")
        augmentation = SafeMedicalAugmentation()
        print("  ✓ Augmentation created")
        
        print("  Creating train dataset...")
        train_dataset = BraTSSegmentationDataset(
            patient_dirs=train_dirs,
            transform=augmentation,
            crop_size=(128, 128, 128)
        )
        print(f"  ✓ Train dataset: {len(train_dataset)} samples")
        
        print("  Creating val dataset...")
        val_dataset = BraTSSegmentationDataset(
            patient_dirs=val_dirs,
            transform=None,  # No augmentation for validation
            crop_size=(128, 128, 128)
        )
        print(f"  ✓ Val dataset: {len(val_dataset)} samples")
        
        # Test loading one sample
        print("\n  [DEBUG] Testing dataset loading...")
        try:
            test_sample = train_dataset[0]
            print(f"    ✓ Loaded sample shape: {test_sample[0].shape}, {test_sample[1].shape}")
        except Exception as e:
            print(f"    ✗ Failed to load sample: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Create data loaders
        print("\n  Creating data loaders...")
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers'],
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            pin_memory=True
        )
        
        print(f"  ✓ Train loader: {len(train_loader)} batches")
        print(f"  ✓ Val loader: {len(val_loader)} batches")
        
        # Initialize model
        print("\n[6/7] Initializing Attention U-Net...")
        model = AttentionUNet3D(in_channels=4, num_classes=4, base_channels=32)
        model = model.to(config['device'])
        
        # Count parameters
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  ✓ Model created")
        print(f"  ✓ Parameters: {params:,} ({params/1e6:.1f}M)")
        print(f"  ✓ Device: {config['device']}")
        
        # Loss and optimizer
        criterion = dice_loss
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )
        
        # Metrics calculator
        metrics_calc = SegmentationMetrics(num_classes=4)
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_wt_dice': [],
            'val_tc_dice': [],
            'val_et_dice': [],
            'learning_rate': []
        }
        
        best_wt_dice = 0.0
        
        # Training loop
        print("\n[7/7] Starting training...")
        print("="*80)
        
        start_time = time.time()
        
        for epoch in range(1, config['num_epochs'] + 1):
            print(f"\nEpoch {epoch}/{config['num_epochs']}")
            print("-" * 80)
            
            # Train
            train_loss = train_epoch(
                model, train_loader, criterion, optimizer, config['device'], epoch
            )
            
            # Validate
            val_loss, val_wt_dice, val_tc_dice, val_et_dice = validate_epoch(
                model, val_loader, criterion, config['device'], metrics_calc
            )
            
            # Update learning rate
            scheduler.step(val_wt_dice)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Record history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_wt_dice'].append(val_wt_dice)
            history['val_tc_dice'].append(val_tc_dice)
            history['val_et_dice'].append(val_et_dice)
            history['learning_rate'].append(current_lr)
            
            # Print epoch summary
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")
            print(f"  Val WT Dice: {val_wt_dice:.4f}")
            print(f"  Val TC Dice: {val_tc_dice:.4f}")
            print(f"  Val ET Dice: {val_et_dice:.4f}")
            print(f"  LR: {current_lr:.6f}")
            
            # Save best model
            if val_wt_dice > best_wt_dice:
                best_wt_dice = val_wt_dice
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_wt_dice': best_wt_dice,
                    'val_wt_dice': val_wt_dice,
                    'val_tc_dice': val_tc_dice,
                    'val_et_dice': val_et_dice
                }
                torch.save(checkpoint, checkpoint_dir / 'attention_unet_best.pth')
                print(f"  ✓ New best model saved! (WT Dice: {best_wt_dice:.4f})")
            
            # Save periodic checkpoints
            if epoch % config['save_every'] == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_wt_dice': val_wt_dice,
                    'val_tc_dice': val_tc_dice,
                    'val_et_dice': val_et_dice
                }
                torch.save(checkpoint, checkpoint_dir / f'attention_unet_epoch_{epoch}.pth')
                print(f"  ✓ Checkpoint saved: epoch_{epoch}.pth")
        
        elapsed_time = time.time() - start_time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        
        print("\n" + "="*80)
        print("TRAINING COMPLETE!")
        print("="*80)
        print(f"  Total time: {hours}h {minutes}m")
        print(f"  Best WT Dice: {best_wt_dice:.4f}")
        print(f"  Improvement over baseline (0.80): {(best_wt_dice - 0.80)*100:+.2f}%")
        
        # Save training history
        print("\n  Saving training history...")
        import json as json_module  # Avoid name conflict
        with open(checkpoint_dir / 'training_history.json', 'w') as f:
            json_module.dump(history, f, indent=2)
        print(f"  ✓ Saved to: {checkpoint_dir / 'training_history.json'}")
        
        # Final summary
        print("\n" + "="*80)
        print("FINAL SUMMARY:")
        print("="*80)
        print(f"  Baseline WT Dice:     0.8006")
        print(f"  Attention U-Net Dice: {best_wt_dice:.4f}")
        print(f"  Improvement:          {(best_wt_dice - 0.8006)*100:+.2f}%")
        print(f"  Target (0.84):        {'✅ REACHED!' if best_wt_dice >= 0.84 else '❌ Not quite'}")
        print("="*80)
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
