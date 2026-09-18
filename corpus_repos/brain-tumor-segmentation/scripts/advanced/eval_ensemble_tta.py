"""
Ensemble: Baseline + Attention (both with TTA)
==============================================
Combine predictions from both models for best results.
"""
import sys
import os
sys.path.insert(0, os.path.abspath('../'))

import torch
import json
from pathlib import Path

from create_dataset import BraTSSegmentationDataset
from unet_model import UNet3D
from attention_unet import AttentionUNet3D
from metrics import SegmentationMetrics, MetricsTracker

print("="*80)
print("ENSEMBLE: BASELINE + ATTENTION (WITH TTA)")
print("="*80)


def apply_tta(model, image, device):
    """Apply TTA to a model"""
    model.eval()
    predictions = []
    
    with torch.no_grad():
        pred = model(image.to(device))
        predictions.append(pred.cpu())
        
        image_h_flip = torch.flip(image, dims=[2])
        pred_h = model(image_h_flip.to(device))
        pred_h = torch.flip(pred_h, dims=[2])
        predictions.append(pred_h.cpu())
        
        image_v_flip = torch.flip(image, dims=[3])
        pred_v = model(image_v_flip.to(device))
        pred_v = torch.flip(pred_v, dims=[3])
        predictions.append(pred_v.cpu())
        
        image_both = torch.flip(image, dims=[2, 3])
        pred_both = model(image_both.to(device))
        pred_both = torch.flip(pred_both, dims=[2, 3])
        predictions.append(pred_both.cpu())
    
    return torch.mean(torch.stack(predictions), dim=0)


# Load both models
print("\n[1/4] Loading models...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Baseline
baseline = UNet3D(in_channels=4, num_classes=4, base_channels=32)
baseline_ckpt = torch.load('../../models/checkpoints/augmented_best_model.pth')
baseline.load_state_dict(baseline_ckpt['model_state_dict'])
baseline.eval().to(device)
print("  ✓ Baseline loaded")

# Attention
attention = AttentionUNet3D(in_channels=4, num_classes=4, base_channels=32)
attention_ckpt = torch.load('../../models/checkpoints/v2/attention_unet/attention_unet_best.pth')
attention.load_state_dict(attention_ckpt['model_state_dict'])
attention.eval().to(device)
print("  ✓ Attention loaded")

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
print(f"  ✓ Test set: {len(test_dataset)} patients")

# Ensemble evaluation
print("\n[3/4] Evaluating ensemble with TTA...")
print("  (This will take 25-30 minutes - 8 predictions per sample)")

metrics_calc = SegmentationMetrics(num_classes=4)
tracker = MetricsTracker(metrics_calc)

processed = 0
for idx in range(len(test_dataset)):
    image, mask = test_dataset[idx]
    image = image.unsqueeze(0)
    
    # Get TTA predictions from both models
    baseline_pred = apply_tta(baseline, image, device)
    attention_pred = apply_tta(attention, image, device)
    
    # Ensemble: Average both predictions
    ensemble_pred = (baseline_pred + attention_pred) / 2.0
    
    # Final prediction
    final_pred = torch.argmax(ensemble_pred, dim=1).squeeze(0).numpy()
    mask_np = mask.numpy()
    
    tracker.add_sample(final_pred, mask_np)
    processed += 1
    
    if processed % 10 == 0:
        print(f"    Processed {processed}/{len(test_dataset)} patients...")

print(f"  ✓ Ensemble evaluation complete!")

# Results
print("\n[4/4] Final Results...")
tracker.print_summary()

# Save
os.makedirs('../../results/v2/ensemble_tta', exist_ok=True)
tracker.save_results('../../results/v2/ensemble_tta/test_metrics.csv')

summary = tracker.get_summary_statistics()
with open('../../results/v2/ensemble_tta/summary_statistics.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\n" + "="*80)
print("✅ FINAL ENSEMBLE RESULTS!")
print("="*80)
print("\nProgression:")
print(f"  Baseline (no TTA):           0.8006")
print(f"  Attention (no TTA):          0.7966")
print(f"  Ensemble + TTA:              {summary['mean']['WT_Dice']:.4f}")
print(f"  Improvement over baseline:   {(summary['mean']['WT_Dice'] - 0.8006)*100:+.2f}%")

target = 0.84
if summary['mean']['WT_Dice'] >= target:
    print(f"\n🎉 SUCCESS! Reached target ({target:.2f}) - Week 1 COMPLETE!")
elif summary['mean']['WT_Dice'] >= 0.82:
    print(f"\n✅ Very close! Just {(target - summary['mean']['WT_Dice'])*100:.2f}% from target")
else:
    print(f"\n✓ Good progress. Gap: {(target - summary['mean']['WT_Dice'])*100:.2f}%")

print("="*80)
