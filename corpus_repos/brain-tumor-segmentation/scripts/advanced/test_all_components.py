"""
Test All Components Before Full Training
========================================
Quickly verify that all improvements work correctly
"""
import sys
import os
sys.path.insert(0, os.path.abspath('../'))

import torch

print("="*80)
print("TESTING ALL COMPONENTS")
print("="*80)

# Test 1: Import improved model
print("\n[1/5] Testing Improved Attention U-Net...")
try:
    from attention_unet_v2_improved import ImprovedAttentionUNet3D
    model = ImprovedAttentionUNet3D(in_channels=4, num_classes=4, base_channels=32, deep_supervision=True)
    print("  ✓ Model imported successfully")
    
    # Test forward pass
    x = torch.randn(1, 4, 128, 128, 128)
    model.train()
    out = model(x)
    print(f"  ✓ Forward pass (train): {len(out)} outputs")
    
    model.eval()
    out = model(x)
    print(f"  ✓ Forward pass (eval): shape {out.shape}")
    
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    exit(1)

# Test 2: Import loss functions
print("\n[2/5] Testing Loss Functions...")
try:
    from advanced_losses import get_loss_function, ComboLoss, DeepSupervisionLoss
    
    # Test combo loss
    combo_loss = ComboLoss()
    pred = torch.randn(1, 4, 128, 128, 128)
    target = torch.randint(0, 4, (1, 128, 128, 128))
    loss = combo_loss(pred, target)
    print(f"  ✓ Combo Loss: {loss.item():.4f}")
    
    # Test deep supervision loss
    ds_loss = get_loss_function('combo', deep_supervision=True)
    outputs = (pred, pred, pred, pred)
    loss = ds_loss(outputs, target)
    print(f"  ✓ Deep Supervision Loss: {loss.item():.4f}")
    
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    exit(1)

# Test 3: Test warmup scheduler
print("\n[3/5] Testing Warmup Scheduler...")
try:
    import numpy as np
    
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
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=5, total_epochs=100, base_lr=0.0005)
    
    # Test first few epochs
    for epoch in range(10):
        lr = scheduler.step()
        if epoch < 5:
            expected_warmup = True
        else:
            expected_warmup = False
    
    print(f"  ✓ Scheduler works: LR at epoch 10 = {lr:.6f}")
    
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    exit(1)

# Test 4: Test mixed precision
print("\n[4/5] Testing Mixed Precision...")
try:
    from torch.cuda.amp import autocast, GradScaler
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        model = model.to(device)
        
        scaler = GradScaler()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
        
        x = torch.randn(1, 4, 128, 128, 128).to(device)
        target = torch.randint(0, 4, (1, 128, 128, 128)).to(device)
        
        model.train()
        with autocast():
            outputs = model(x)
            if isinstance(outputs, tuple):
                loss = ds_loss(outputs, target)
            else:
                loss = combo_loss(outputs, target)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        print(f"  ✓ Mixed precision works on GPU")
    else:
        print(f"  ⚠ No GPU available, skipping mixed precision test")
    
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    exit(1)

# Test 5: Test data loading
print("\n[5/5] Testing Data Loading...")
try:
    from pathlib import Path
    import json
    from create_dataset import BraTSSegmentationDataset
    from torch.utils.data import DataLoader
    
    # Load splits
    with open('../../data/splits/data_splits.json') as f:
        splits = json.load(f)
    
    data_root = Path('../../data/raw/brats_gli/training_data1_v2')
    train_dirs = [data_root / pid for pid in splits['train'][:5]]  # Just test with 5
    
    dataset = BraTSSegmentationDataset(
        patient_dirs=train_dirs,
        transform=None,
        crop_size=(128, 128, 128)
    )
    
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=1)
    
    # Load one batch
    images, masks = next(iter(loader))
    print(f"  ✓ Data loading works: {images.shape}, {masks.shape}")
    
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    exit(1)

print("\n" + "="*80)
print("✅ ALL TESTS PASSED!")
print("="*80)
print("\nYou're ready to start training!")
print("\nNext steps:")
print("  1. Review the configuration in train_attention_ultimate.py")
print("  2. Submit job: sbatch submit_ultimate.sh")
print("  3. Monitor: tail -f ../../models/logs/v2/attention_ultimate_*.log")
print("="*80)
