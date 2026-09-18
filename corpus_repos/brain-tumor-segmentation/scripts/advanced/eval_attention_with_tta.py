"""
Test-Time Augmentation for Attention U-Net
==========================================
Apply multiple augmentations during inference and average predictions.

Expected improvement: +2-3% over single prediction
"""
import sys
import os
sys.path.insert(0, os.path.abspath('../'))

import torch
from torch.utils.data import DataLoader
import json
from pathlib import Path
import numpy as np

from create_dataset import BraTSSegmentationDataset
from attention_unet import AttentionUNet3D
from metrics import SegmentationMetrics, MetricsTracker

print("="*80)
print("ATTENTION U-NET WITH TEST-TIME AUGMENTATION")
print("="*80)


def apply_tta(model, image, device):
    """
    Apply test-time augmentation
    
    Augmentations:
    1. Original
    2. Horizontal flip
    3. Vertical flip  
    4. Both flips
    
    Returns: Averaged prediction
    """
    model.eval()
    predictions = []
    
    with torch.no_grad():
        # 1. Original
        pred = model(image.to(device))
        predictions.append(pred.cpu())
        
        # 2. Horizontal flip (dim 2)
        image_h_flip = torch.flip(image, dims=[2])
        pred_h = model(image_h_flip.to(device))
        pred_h = torch.flip(pred_h, dims=[2])  # Flip back
        predictions.append(pred_h.cpu())
        
        # 3. Vertical flip (dim 3)
        image_v_flip = torch.flip(image, dims=[3])
        pred_v = model(image_v_flip.to(device))
        pred_v = torch.flip(pred_v, dims=[3])  # Flip back
        predictions.append(pred_v.cpu())
        
        # 4. Both flips
        image_both_flip = torch.flip(image, dims=[2, 3])
        pred_both = model(image_both_flip.to(device))
        pred_both = torch.flip(pred_both, dims=[2, 3])  # Flip back
        predictions.append(pred_both.cpu())
    
    # Average all predictions
    avg_prediction = torch.mean(torch.stack(predictions), dim=0)
    
    return avg_prediction


# Load model
print("\n[1/4] Loading Attention U-Net...")
model = AttentionUNet3D(in_channels=4, num_classes=4, base_channels=32)
checkpoint = torch.load('../../models/checkpoints/v2/attention_unet/attention_unet_best.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(f"  ✓ Model loaded from epoch {checkpoint['epoch']}")
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
print("  (This will take 15-20 minutes due to 4× inference per sample)")

metrics_calc = SegmentationMetrics(num_classes=4)
tracker = MetricsTracker(metrics_calc)

processed = 0
for idx in range(len(test_dataset)):
    # Load one sample at a time for TTA
    image, mask = test_dataset[idx]
    image = image.unsqueeze(0)  # Add batch dimension
    
    # Apply TTA
    avg_pred = apply_tta(model, image, device)
    
    # Get final prediction
    final_pred = torch.argmax(avg_pred, dim=1).squeeze(0).numpy()
    mask_np = mask.numpy()
    
    # Calculate metrics
    tracker.add_sample(final_pred, mask_np)
    processed += 1
    
    if (processed) % 20 == 0:
        print(f"    Processed {processed}/{len(test_dataset)} patients...")

print(f"  ✓ Evaluation complete! Processed {processed} patients")

# Display results
print("\n[4/4] Results with Test-Time Augmentation...")
tracker.print_summary()

# Save results
os.makedirs('../../results/v2/attention_unet_tta', exist_ok=True)
tracker.save_results('../../results/v2/attention_unet_tta/test_metrics.csv')

summary = tracker.get_summary_statistics()
with open('../../results/v2/attention_unet_tta/summary_statistics.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\n" + "="*80)
print("✅ ATTENTION U-NET WITH TTA - COMPLETE!")
print("="*80)
print("\nResults saved:")
print("  📊 results/v2/attention_unet_tta/test_metrics.csv")
print("  📈 results/v2/attention_unet_tta/summary_statistics.json")

print("\n" + "="*80)
print("PERFORMANCE COMPARISON:")
print("="*80)
print(f"  Without TTA: 0.7966")
print(f"  With TTA:    {summary['mean']['WT_Dice']:.4f}")
print(f"  Improvement: {(summary['mean']['WT_Dice'] - 0.7966)*100:+.2f}%")

if summary['mean']['WT_Dice'] >= 0.82:
    print(f"\n✅ Excellent! TTA brought significant improvement")
elif summary['mean']['WT_Dice'] >= 0.80:
    print(f"\n✓ Good! TTA helped")
else:
    print(f"\n⚠️ TTA improvement was modest")

print("="*80)
