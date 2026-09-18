"""
Comprehensive Evaluation - FINAL WORKING VERSION
================================================
This version uses the correct BraTSSegmentationDataset initialization:
  - patient_dirs: list of full paths to patient directories
"""
import sys
import os
sys.path.insert(0, os.path.abspath('../'))

import torch
from torch.utils.data import DataLoader
import json
from pathlib import Path

print("="*80)
print("COMPREHENSIVE EVALUATION OF YOUR CURRENT BEST MODEL")
print("="*80)

# [0] Detect and load model
print("\n[0/5] Loading your model...")
from unet_model import UNet3D

model = UNet3D(in_channels=4, num_classes=4, base_channels=32)
print("  ✓ Model created")

checkpoint_path = '../../models/checkpoints/augmented_best_model.pth'
checkpoint = torch.load(checkpoint_path, map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(f"  ✓ Model loaded from checkpoint (epoch {checkpoint.get('epoch', 'unknown')})")
print(f"  ✓ Using device: {device}")

# [1] Prepare test data paths
print("\n[1/5] Preparing test data...")

# Load splits
with open('../../data/splits/data_splits.json', 'r') as f:
    splits = json.load(f)

# Get patient IDs
test_patient_ids = splits['test']
print(f"  ✓ Found {len(test_patient_ids)} test patients")

# Convert patient IDs to full directory paths (as Path objects)
data_root = Path('../../data/raw/brats_gli/training_data1_v2')
test_patient_dirs = []

for patient_id in test_patient_ids:
    patient_dir = data_root / patient_id
    if patient_dir.exists():
        test_patient_dirs.append(patient_dir)  # Keep as Path object, not string
    else:
        print(f"  ! Warning: Patient directory not found: {patient_dir}")

print(f"  ✓ Found {len(test_patient_dirs)} patient directories")

if len(test_patient_dirs) == 0:
    print("  ✗ No patient directories found!")
    print(f"  ! Check if path exists: {data_root}")
    exit(1)

# [2] Create dataset and loader
print("\n[2/5] Creating test dataset and loader...")

from create_dataset import BraTSSegmentationDataset

# Create dataset with the correct parameter name: patient_dirs
test_dataset = BraTSSegmentationDataset(
    patient_dirs=test_patient_dirs,
    transform=None,
    crop_size=(128, 128, 128)
)

test_loader = DataLoader(
    test_dataset,
    batch_size=2,
    shuffle=False,
    num_workers=1  # Reduced to avoid warnings
)

print(f"  ✓ Dataset created: {len(test_dataset)} samples")
print(f"  ✓ Loader created: {len(test_loader)} batches")

# [3] Evaluate with comprehensive metrics
print("\n[3/5] Calculating comprehensive metrics...")
print("  (This will take 5-10 minutes...)")

from metrics import SegmentationMetrics, MetricsTracker

metrics_calc = SegmentationMetrics(num_classes=4)
tracker = MetricsTracker(metrics_calc)

processed_samples = 0
with torch.no_grad():
    for batch_idx, (images, masks) in enumerate(test_loader):
        images = images.to(device)
        masks_np = masks.numpy()
        
        # Get predictions
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
        preds_np = preds.cpu().numpy()
        
        # Calculate metrics for each sample in batch
        for i in range(preds_np.shape[0]):
            tracker.add_sample(preds_np[i], masks_np[i])
            processed_samples += 1
        
        # Progress indicator
        if (batch_idx + 1) % 20 == 0:
            print(f"    Processed {batch_idx + 1}/{len(test_loader)} batches ({processed_samples} samples)...")

print(f"  ✓ Evaluation complete! Processed {processed_samples} samples")

# [4] Display results
print("\n[4/5] Displaying comprehensive results...")
tracker.print_summary()

# [5] Save results
print("\n[5/5] Saving results...")
os.makedirs('../../results/v2', exist_ok=True)

# Save per-sample metrics to CSV
csv_path = '../../results/v2/baseline_comprehensive_metrics.csv'
tracker.save_results(csv_path)
print(f"  ✓ Saved detailed metrics: {csv_path}")

# Save summary statistics to JSON
summary = tracker.get_summary_statistics()
json_path = '../../results/v2/baseline_summary_statistics.json'
with open(json_path, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"  ✓ Saved summary statistics: {json_path}")

# Create human-readable report
report_path = '../../results/v2/baseline_summary_report.txt'
with open(report_path, 'w') as f:
    f.write("="*80 + "\n")
    f.write("COMPREHENSIVE BASELINE EVALUATION REPORT\n")
    f.write("="*80 + "\n\n")
    
    f.write("Model Information:\n")
    f.write(f"  Name: Augmented U-Net (Safe Augmentation)\n")
    f.write(f"  Checkpoint: {checkpoint_path}\n")
    f.write(f"  Trained Epochs: {checkpoint.get('epoch', 'unknown')}\n")
    f.write(f"  Parameters: 22.6M (base_channels=32)\n")
    f.write(f"  Test Patients: {len(test_patient_dirs)}\n\n")
    
    f.write("="*80 + "\n")
    f.write("PRIMARY METRICS (BraTS Standard)\n")
    f.write("="*80 + "\n\n")
    
    metrics_to_report = [
        ('Whole Tumor (WT)', 'WT_Dice'),
        ('Tumor Core (TC)', 'TC_Dice'),
        ('Enhancing Tumor (ET)', 'ET_Dice')
    ]
    
    for label, key in metrics_to_report:
        f.write(f"{label}:\n")
        f.write(f"  Mean:   {summary['mean'][key]:.4f}\n")
        f.write(f"  Std:    {summary['std'][key]:.4f}\n")
        f.write(f"  Median: {summary['median'][key]:.4f}\n")
        f.write(f"  Min:    {summary['min'][key]:.4f}\n")
        f.write(f"  Max:    {summary['max'][key]:.4f}\n\n")
    
    f.write("="*80 + "\n")
    f.write("PER-CLASS DETAILED METRICS\n")
    f.write("="*80 + "\n\n")
    
    class_names = ['NCR', 'ED', 'ET']
    for class_name in class_names:
        f.write(f"{class_name} (Necrotic/Edema/Enhancing):\n")
        f.write(f"  Dice:        {summary['mean'][f'{class_name}_Dice']:.4f} ± {summary['std'][f'{class_name}_Dice']:.4f}\n")
        f.write(f"  Sensitivity: {summary['mean'][f'{class_name}_Sensitivity']:.4f} ± {summary['std'][f'{class_name}_Sensitivity']:.4f}\n")
        f.write(f"  Specificity: {summary['mean'][f'{class_name}_Specificity']:.4f} ± {summary['std'][f'{class_name}_Specificity']:.4f}\n")
        f.write(f"  Precision:   {summary['mean'][f'{class_name}_Precision']:.4f} ± {summary['std'][f'{class_name}_Precision']:.4f}\n")
        f.write(f"  F1 Score:    {summary['mean'][f'{class_name}_F1']:.4f} ± {summary['std'][f'{class_name}_F1']:.4f}\n\n")
    
    f.write("="*80 + "\n")
    f.write("IMPROVEMENT ANALYSIS\n")
    f.write("="*80 + "\n\n")
    
    current_wt = summary['mean']['WT_Dice']
    current_tc = summary['mean']['TC_Dice']
    current_et = summary['mean']['ET_Dice']
    
    target_wt = 0.88
    target_tc = 0.85
    target_et = 0.80
    
    gap_wt = (target_wt - current_wt) * 100
    gap_tc = (target_tc - current_tc) * 100
    gap_et = (target_et - current_et) * 100
    
    f.write("Current Performance vs Targets:\n\n")
    f.write(f"  WT Dice: {current_wt:.4f} → Target: {target_wt:.4f} (Gap: {gap_wt:+.2f}%)\n")
    f.write(f"  TC Dice: {current_tc:.4f} → Target: {target_tc:.4f} (Gap: {gap_tc:+.2f}%)\n")
    f.write(f"  ET Dice: {current_et:.4f} → Target: {target_et:.4f} (Gap: {gap_et:+.2f}%)\n\n")
    
    f.write("Improvement Roadmap (Expected Gains):\n")
    f.write("  Week 1 - Attention U-Net:        +3-5%  → WT: ~{:.3f}\n".format(current_wt + 0.04))
    f.write("  Week 2 - Advanced Loss + TTA:    +2-3%  → WT: ~{:.3f}\n".format(current_wt + 0.07))
    f.write("  Week 3 - Ensemble Methods:       +2-3%  → WT: ~{:.3f}\n".format(current_wt + 0.10))
    f.write("  Week 4 - Post-processing:        +1-2%  → WT: ~{:.3f}\n\n".format(current_wt + 0.12))
    
    f.write("="*80 + "\n")
    f.write("NEXT STEPS\n")
    f.write("="*80 + "\n\n")
    f.write("1. Review this report and identify improvement areas\n")
    f.write("2. Start Week 1: Implement Attention U-Net architecture\n")
    f.write("3. Expected after Week 1: 0.83-0.85 WT Dice\n")
    f.write("4. Follow improvement_roadmap.md for systematic progress\n")
    f.write("5. Target: 0.88+ WT Dice within 4 weeks\n")

print(f"  ✓ Saved human-readable report: {report_path}")

# Final summary display
print("\n" + "="*80)
print("✅ COMPREHENSIVE EVALUATION COMPLETE!")
print("="*80)
print("\nFiles saved in results/v2/:")
print(f"  📊 baseline_comprehensive_metrics.csv   ({processed_samples} samples)")
print(f"  📈 baseline_summary_statistics.json")
print(f"  📝 baseline_summary_report.txt")

print("\n" + "="*80)
print("KEY FINDINGS:")
print("="*80)
print(f"  WT Dice: {summary['mean']['WT_Dice']:.4f} ± {summary['std']['WT_Dice']:.4f}")
print(f"  TC Dice: {summary['mean']['TC_Dice']:.4f} ± {summary['std']['TC_Dice']:.4f}")
print(f"  ET Dice: {summary['mean']['ET_Dice']:.4f} ± {summary['std']['ET_Dice']:.4f}")
print("="*80)

current_wt = summary['mean']['WT_Dice']
target_wt = 0.88
gap = (target_wt - current_wt) * 100

print(f"\n📍 YOUR BASELINE:")
print(f"   Current WT Dice: {current_wt:.4f}")
print(f"   Target WT Dice:  {target_wt:.4f}")
print(f"   Gap to close:    {gap:.2f}%")

print(f"\n🎯 IMPROVEMENT PLAN:")
print(f"   Week 1 Target: {current_wt + 0.04:.4f} (+4%)")
print(f"   Week 2 Target: {current_wt + 0.07:.4f} (+7%)")  
print(f"   Week 3 Target: {current_wt + 0.10:.4f} (+10%)")
print(f"   Final Target:  {target_wt:.4f}")

print(f"\n📖 NEXT STEPS:")
print(f"   1. Read report:  cat {report_path}")
print(f"   2. Review CSV:   Open baseline_comprehensive_metrics.csv in Excel")
print(f"   3. Start Week 1: Implement Attention U-Net")
print(f"   4. Follow:       improvement_roadmap.md")

print("="*80)
print("\n✨ You now have comprehensive baseline metrics!")
print("   Ready to start systematic improvements! 🚀")
print("="*80)
