"""
Evaluate Ultimate Attention U-Net on Test Set
==============================================
Final evaluation to confirm target achievement
"""
import sys
import os
sys.path.insert(0, os.path.abspath('../'))

import torch
from torch.utils.data import DataLoader
import json
from pathlib import Path
import numpy as np

print("="*80)
print("ULTIMATE ATTENTION U-NET - TEST SET EVALUATION")
print("="*80)

# Import modules
from create_dataset import BraTSSegmentationDataset
from attention_unet_v2_improved import ImprovedAttentionUNet3D
from metrics import SegmentationMetrics, MetricsTracker

# Load model
print("\n[1/4] Loading best model...")
model = ImprovedAttentionUNet3D(
    in_channels=4,
    num_classes=4,
    base_channels=32,
    deep_supervision=True  # MUST match training config
)

checkpoint_path = '../../models/checkpoints/v2/attention_ultimate/best_model.pth'
checkpoint = torch.load(checkpoint_path, map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()  # In eval mode, only returns main output (no deep supervision outputs)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

print(f"  ✓ Loaded from epoch {checkpoint['epoch']}")
print(f"  ✓ Validation WT Dice: {checkpoint['val_wt_dice']:.4f}")
print(f"  ✓ Using device: {device}")

# Load test data
print("\n[2/4] Loading test data...")
with open('../../data/splits/data_splits.json') as f:
    splits = json.load(f)

data_root = Path('../../data/raw/brats_gli/training_data1_v2')
test_dirs = [data_root / pid for pid in splits['test']]

test_dataset = BraTSSegmentationDataset(
    patient_dirs=test_dirs,
    transform=None,
    crop_size=(128, 128, 128)
)

test_loader = DataLoader(
    test_dataset,
    batch_size=2,
    shuffle=False,
    num_workers=1
)

print(f"  ✓ Test set: {len(test_dataset)} patients")

# Evaluate
print("\n[3/4] Evaluating on test set...")
print("  (This will take 5-10 minutes...)")

metrics_calc = SegmentationMetrics(num_classes=4)
tracker = MetricsTracker(metrics_calc)

processed = 0
with torch.no_grad():
    for batch_idx, (images, masks) in enumerate(test_loader):
        images = images.to(device)
        masks_np = masks.numpy()
        
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
        preds_np = preds.cpu().numpy()
        
        for i in range(preds_np.shape[0]):
            tracker.add_sample(preds_np[i], masks_np[i])
            processed += 1
        
        if (batch_idx + 1) % 20 == 0:
            print(f"    Processed {batch_idx + 1}/{len(test_loader)} batches ({processed} samples)...")

print(f"  ✓ Evaluation complete! Processed {processed} samples")

# Display results
print("\n[4/4] Results:")
tracker.print_summary()

# Save results
os.makedirs('../../results/v2/attention_ultimate', exist_ok=True)
tracker.save_results('../../results/v2/attention_ultimate/test_metrics.csv')

summary = tracker.get_summary_statistics()
with open('../../results/v2/attention_ultimate/test_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

# Final comparison
print("\n" + "="*80)
print("COMPREHENSIVE RESULTS COMPARISON")
print("="*80)

results_table = [
    ["Model", "Val WT Dice", "Test WT Dice", "Improvement"],
    ["-"*15, "-"*12, "-"*13, "-"*12],
    ["Baseline", "0.7000", "0.8006", "Baseline"],
    ["Attention V1", "0.7621", "0.7966", "-0.40%"],
    ["Ultimate V2", f"{checkpoint['val_wt_dice']:.4f}", 
     f"{summary['mean']['WT_Dice']:.4f}", 
     f"{(summary['mean']['WT_Dice'] - 0.8006)*100:+.2f}%"]
]

for row in results_table:
    print(f"  {row[0]:<15} {row[1]:<12} {row[2]:<13} {row[3]:<12}")

print("="*80)

# Achievement check
test_wt = summary['mean']['WT_Dice']
print(f"\n🎯 TARGET ACHIEVEMENT CHECK:")
print(f"   Your Test WT Dice:  {test_wt:.4f}")
print(f"   Week 1 Target:      0.8400")

if test_wt >= 0.88:
    print(f"\n🎉🎉🎉 OUTSTANDING! You've reached the FINAL GOAL (0.88)!")
    print(f"   This exceeds all expectations!")
elif test_wt >= 0.86:
    print(f"\n✅✅ EXCELLENT! You've exceeded the Week 1 target!")
    print(f"   You're ahead of schedule!")
elif test_wt >= 0.84:
    print(f"\n✅ SUCCESS! You've met the Week 1 target (0.84)!")
    print(f"   Ready to proceed to Week 2-3 improvements!")
elif test_wt >= 0.82:
    print(f"\n✓ Very close! Just {(0.84 - test_wt)*100:.1f}% away from target.")
    print(f"   Test-Time Augmentation should push you over 0.84!")
else:
    print(f"\n⚠️ Below target. Gap: {(0.84 - test_wt)*100:.1f}%")
    print(f"   Recommend: Try ensemble or test-time augmentation")

print("\n" + "="*80)
print("DETAILED METRICS:")
print("="*80)
print(f"  WT Dice: {summary['mean']['WT_Dice']:.4f} ± {summary['std']['WT_Dice']:.4f}")
print(f"  TC Dice: {summary['mean']['TC_Dice']:.4f} ± {summary['std']['TC_Dice']:.4f}")
print(f"  ET Dice: {summary['mean']['ET_Dice']:.4f} ± {summary['std']['ET_Dice']:.4f}")
print("="*80)

print("\nResults saved:")
print("  📊 results/v2/attention_ultimate/test_metrics.csv")
print("  📈 results/v2/attention_ultimate/test_summary.json")
print("="*80)
