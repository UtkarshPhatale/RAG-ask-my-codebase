"""
Test-Time Augmentation for Baseline Model
=========================================
"""
import sys
import os
sys.path.insert(0, os.path.abspath('../'))

import torch
from torch.utils.data import DataLoader
import json
from pathlib import Path

from create_dataset import BraTSSegmentationDataset
from unet_model import UNet3D
from metrics import SegmentationMetrics, MetricsTracker

print("="*80)
print("BASELINE MODEL WITH TEST-TIME AUGMENTATION")
print("="*80)


def apply_tta(model, image, device):
    """Apply test-time augmentation"""
    model.eval()
    predictions = []
    
    with torch.no_grad():
        # 1. Original
        pred = model(image.to(device))
        predictions.append(pred.cpu())
        
        # 2. Horizontal flip
        image_h_flip = torch.flip(image, dims=[2])
        pred_h = model(image_h_flip.to(device))
        pred_h = torch.flip(pred_h, dims=[2])
        predictions.append(pred_h.cpu())
        
        # 3. Vertical flip
        image_v_flip = torch.flip(image, dims=[3])
        pred_v = model(image_v_flip.to(device))
        pred_v = torch.flip(pred_v, dims=[3])
        predictions.append(pred_v.cpu())
        
        # 4. Both flips
        image_both_flip = torch.flip(image, dims=[2, 3])
        pred_both = model(image_both_flip.to(device))
        pred_both = torch.flip(pred_both, dims=[2, 3])
        predictions.append(pred_both.cpu())
    
    avg_prediction = torch.mean(torch.stack(predictions), dim=0)
    return avg_prediction


# Load baseline model
print("\n[1/4] Loading Baseline U-Net...")
model = UNet3D(in_channels=4, num_classes=4, base_channels=32)
checkpoint = torch.load('../../models/checkpoints/augmented_best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(f"  ✓ Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
print(f"  ✓ Device: {device}")

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

# Evaluate with TTA
print("\n[3/4] Evaluating with Test-Time Augmentation...")
print("  (This will take 15-20 minutes)")

metrics_calc = SegmentationMetrics(num_classes=4)
tracker = MetricsTracker(metrics_calc)

processed = 0
for idx in range(len(test_dataset)):
    image, mask = test_dataset[idx]
    image = image.unsqueeze(0)
    
    avg_pred = apply_tta(model, image, device)
    final_pred = torch.argmax(avg_pred, dim=1).squeeze(0).numpy()
    mask_np = mask.numpy()
    
    tracker.add_sample(final_pred, mask_np)
    processed += 1
    
    if (processed) % 20 == 0:
        print(f"    Processed {processed}/{len(test_dataset)} patients...")

print(f"  ✓ Evaluation complete!")

# Display results
print("\n[4/4] Results...")
tracker.print_summary()

# Save results
os.makedirs('../../results/v2/baseline_tta', exist_ok=True)
tracker.save_results('../../results/v2/baseline_tta/test_metrics.csv')

summary = tracker.get_summary_statistics()
with open('../../results/v2/baseline_tta/summary_statistics.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\n" + "="*80)
print("✅ BASELINE WITH TTA - COMPLETE!")
print("="*80)
print("\nPerformance:")
print(f"  Without TTA: 0.8006")
print(f"  With TTA:    {summary['mean']['WT_Dice']:.4f}")
print(f"  Improvement: {(summary['mean']['WT_Dice'] - 0.8006)*100:+.2f}%")
print("="*80)
