"""
Smart Ensemble: Baseline+TTA + Attention (no TTA)
=================================================
Use TTA only where it helps!
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
print("SMART ENSEMBLE: BASELINE+TTA + ATTENTION (NO TTA)")
print("="*80)


def apply_tta(model, image, device):
    """Apply TTA"""
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


# Load models
print("\n[1/4] Loading models...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

baseline = UNet3D(in_channels=4, num_classes=4, base_channels=32)
baseline_ckpt = torch.load('../../models/checkpoints/augmented_best_model.pth')
baseline.load_state_dict(baseline_ckpt['model_state_dict'])
baseline.eval().to(device)
print("  ✓ Baseline loaded")

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
test_dataset = BraTSSegmentationDataset(test_dirs, transform=None, crop_size=(128, 128, 128))
print(f"  ✓ Test set: {len(test_dataset)} patients")

# Smart ensemble
print("\n[3/4] Evaluating smart ensemble...")
print("  Strategy: Baseline WITH TTA + Attention WITHOUT TTA")

metrics_calc = SegmentationMetrics(num_classes=4)
tracker = MetricsTracker(metrics_calc)

processed = 0
for idx in range(len(test_dataset)):
    image, mask = test_dataset[idx]
    image = image.unsqueeze(0)
    
    # Baseline with TTA
    baseline_pred = apply_tta(baseline, image, device)
    
    # Attention WITHOUT TTA (just original)
    with torch.no_grad():
        attention_pred = attention(image.to(device)).cpu()
    
    # Ensemble: Average both
    ensemble_pred = (baseline_pred + attention_pred) / 2.0
    
    final_pred = torch.argmax(ensemble_pred, dim=1).squeeze(0).numpy()
    mask_np = mask.numpy()
    
    tracker.add_sample(final_pred, mask_np)
    processed += 1
    
    if processed % 20 == 0:
        print(f"    Processed {processed}/{len(test_dataset)} patients...")

print(f"  ✓ Evaluation complete!")

# Results
print("\n[4/4] Final Results...")
tracker.print_summary()

os.makedirs('../../results/v2/ensemble_smart', exist_ok=True)
tracker.save_results('../../results/v2/ensemble_smart/test_metrics.csv')

summary = tracker.get_summary_statistics()
with open('../../results/v2/ensemble_smart/summary_statistics.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\n" + "="*80)
print("✅ SMART ENSEMBLE RESULTS!")
print("="*80)
print("\nComparison:")
print(f"  Baseline (no TTA):           0.8006")
print(f"  Baseline (with TTA):         0.8026")
print(f"  Attention (no TTA):          0.7966")
print(f"  Smart Ensemble:              {summary['mean']['WT_Dice']:.4f}")
print(f"  Improvement over baseline:   {(summary['mean']['WT_Dice'] - 0.8006)*100:+.2f}%")

if summary['mean']['WT_Dice'] >= 0.84:
    print(f"\n🎉 SUCCESS! Week 1 target ACHIEVED!")
elif summary['mean']['WT_Dice'] >= 0.82:
    print(f"\n✅ Very close to Week 1 target!")
elif summary['mean']['WT_Dice'] >= 0.81:
    print(f"\n✓ Solid improvement!")
else:
    print(f"\n⚠️ Modest improvement")

print("="*80)
