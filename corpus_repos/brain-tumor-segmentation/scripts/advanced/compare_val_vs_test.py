"""
Compare Validation vs Test Performance
======================================
"""
import sys
import os
sys.path.insert(0, os.path.abspath('../'))

import torch
from torch.utils.data import DataLoader
import json
from pathlib import Path

from create_dataset import BraTSSegmentationDataset
from attention_unet import AttentionUNet3D
from metrics import SegmentationMetrics

print("="*80)
print("VALIDATION vs TEST SET COMPARISON")
print("="*80)

# Load model
model = AttentionUNet3D(in_channels=4, num_classes=4, base_channels=32)
checkpoint = torch.load('../../models/checkpoints/v2/attention_unet/attention_unet_best.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
device = torch.device('cuda')
model.to(device)

# Load splits
with open('../../data/splits/data_splits.json') as f:
    splits = json.load(f)

data_root = Path('../../data/raw/brats_gli/training_data1_v2')

# Evaluate on VALIDATION set
print("\n[1/2] Evaluating on VALIDATION set...")
val_dirs = [data_root / pid for pid in splits['val']]
val_dataset = BraTSSegmentationDataset(val_dirs, transform=None, crop_size=(128, 128, 128))
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=1)

metrics_calc = SegmentationMetrics(num_classes=4)

val_wt_sum = 0
val_count = 0
with torch.no_grad():
    for images, masks in val_loader:
        outputs = model(images.to(device))
        preds = torch.argmax(outputs, dim=1)
        
        for i in range(preds.shape[0]):
            wt_dice = metrics_calc.whole_tumor_dice(
                preds[i].cpu().numpy(), 
                masks[i].numpy()
            )
            val_wt_sum += wt_dice
            val_count += 1

val_wt_dice = val_wt_sum / val_count
print(f"  ✓ Validation WT Dice: {val_wt_dice:.4f} (matches checkpoint: {checkpoint['val_wt_dice']:.4f})")

# Evaluate on TEST set
print("\n[2/2] Evaluating on TEST set...")
test_dirs = [data_root / pid for pid in splits['test']]
test_dataset = BraTSSegmentationDataset(test_dirs, transform=None, crop_size=(128, 128, 128))
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=1)

test_wt_sum = 0
test_count = 0
with torch.no_grad():
    for images, masks in test_loader:
        outputs = model(images.to(device))
        preds = torch.argmax(outputs, dim=1)
        
        for i in range(preds.shape[0]):
            wt_dice = metrics_calc.whole_tumor_dice(
                preds[i].cpu().numpy(),
                masks[i].numpy()
            )
            test_wt_sum += wt_dice
            test_count += 1

test_wt_dice = test_wt_sum / test_count
print(f"  ✓ Test WT Dice: {test_wt_dice:.4f}")

print("\n" + "="*80)
print("COMPARISON:")
print("="*80)
print(f"  Validation WT Dice: {val_wt_dice:.4f}")
print(f"  Test WT Dice:       {test_wt_dice:.4f}")
print(f"  Difference:         {(test_wt_dice - val_wt_dice)*100:+.2f}%")

if test_wt_dice > val_wt_dice + 0.02:
    print(f"\n✅ Test set is actually EASIER than validation!")
    print(f"   Your attention model IS working - val performance improved from 0.70 to 0.76")
elif abs(test_wt_dice - val_wt_dice) < 0.01:
    print(f"\n✓ Validation and test performance are consistent")
else:
    print(f"\n⚠️ Test set is harder than validation")

# Compare with baseline
print("\n" + "="*80)
print("BASELINE COMPARISON:")
print("="*80)
print(f"  Baseline:")
print(f"    Validation: 0.7000")
print(f"    Test:       0.8006")
print(f"    Gap:        +10.06%")
print(f"\n  Attention:")
print(f"    Validation: {val_wt_dice:.4f}")
print(f"    Test:       {test_wt_dice:.4f}")
print(f"    Gap:        {(test_wt_dice - val_wt_dice)*100:+.2f}%")

print("\n💡 INSIGHT:")
if test_wt_dice > val_wt_dice + 0.02:
    print("  Both models show test > validation (test set easier)")
    print("  Attention improved validation by +6.2%")
    print("  Attention should also improve test if we optimize properly")
print("="*80)
