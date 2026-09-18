"""
Evaluate Attention U-Net on Test Set
====================================
Calculate comprehensive metrics for the trained Attention U-Net
"""
import sys
import os
sys.path.insert(0, os.path.abspath('../'))

import torch
from torch.utils.data import DataLoader
import json
from pathlib import Path

print("="*80)
print("EVALUATING ATTENTION U-NET ON TEST SET")
print("="*80)

# Import modules
from create_dataset import BraTSSegmentationDataset
from attention_unet import AttentionUNet3D
from metrics import SegmentationMetrics, MetricsTracker

# [1] Load model
print("\n[1/4] Loading Attention U-Net model...")
model = AttentionUNet3D(in_channels=4, num_classes=4, base_channels=32)

checkpoint_path = '../../models/checkpoints/v2/attention_unet/attention_unet_best.pth'
checkpoint = torch.load(checkpoint_path, map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

print(f"  ✓ Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
print(f"  ✓ Validation WT Dice: {checkpoint.get('val_wt_dice', 'unknown'):.4f}")
print(f"  ✓ Using device: {device}")

# [2] Load test data
print("\n[2/4] Loading test data...")
with open('../../data/splits/data_splits.json', 'r') as f:
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

# [3] Evaluate
print("\n[3/4] Calculating comprehensive metrics...")
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

# [4] Display and save results
print("\n[4/4] Saving results...")
tracker.print_summary()

os.makedirs('../../results/v2/attention_unet', exist_ok=True)
tracker.save_results('../../results/v2/attention_unet/test_metrics.csv')

summary = tracker.get_summary_statistics()
with open('../../results/v2/attention_unet/summary_statistics.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\n" + "="*80)
print("✅ ATTENTION U-NET EVALUATION COMPLETE!")
print("="*80)
print("\nResults saved:")
print("  📊 results/v2/attention_unet/test_metrics.csv")
print("  📈 results/v2/attention_unet/summary_statistics.json")

print("\n" + "="*80)
print("TEST SET PERFORMANCE:")
print("="*80)
print(f"  WT Dice: {summary['mean']['WT_Dice']:.4f} ± {summary['std']['WT_Dice']:.4f}")
print(f"  TC Dice: {summary['mean']['TC_Dice']:.4f} ± {summary['std']['TC_Dice']:.4f}")
print(f"  ET Dice: {summary['mean']['ET_Dice']:.4f} ± {summary['std']['ET_Dice']:.4f}")
print("="*80)

# Compare with baseline
baseline_wt = 0.8006
current_wt = summary['mean']['WT_Dice']
improvement = (current_wt - baseline_wt) * 100

print(f"\n📊 COMPARISON WITH BASELINE:")
print(f"   Baseline WT Dice:     {baseline_wt:.4f}")
print(f"   Attention U-Net Dice: {current_wt:.4f}")
print(f"   Improvement:          {improvement:+.2f}%")

if current_wt >= 0.84:
    print(f"\n✅ SUCCESS! Reached Week 1 target (0.84)")
    print(f"   Ready to move to Week 2!")
elif current_wt >= 0.82:
    print(f"\n✓ Good progress! Close to target.")
    print(f"   Consider: Train a bit longer or try combo loss")
else:
    print(f"\n⚠ Below expected improvement.")
    print(f"   Investigate: Check training curves, try different hyperparameters")

print("="*80)
