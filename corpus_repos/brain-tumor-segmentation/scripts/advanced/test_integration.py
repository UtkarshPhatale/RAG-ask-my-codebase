"""
Test: Verify New Scripts Work with Your Existing Data
======================================================
This will confirm everything is connected properly.
"""
import sys
import os

# Add parent directory to access your existing scripts
sys.path.insert(0, os.path.abspath('../'))

print("="*70)
print("INTEGRATION TEST - Verifying Setup")
print("="*70)

# Test 1: Import your existing dataset code
print("\n[1/5] Testing import of existing dataset...")
try:
    from create_dataset import BraTSSegmentationDataset
    print("  ✓ Successfully imported create_dataset.py")
except Exception as e:
    print(f"  ✗ Error: {e}")
    exit(1)

# Test 2: Import new metrics module
print("\n[2/5] Testing import of new metrics module...")
try:
    from metrics import SegmentationMetrics, MetricsTracker
    print("  ✓ Successfully imported metrics.py")
except Exception as e:
    print(f"  ✗ Error: {e}")
    exit(1)

# Test 3: Check data splits exist
print("\n[3/5] Checking data splits file...")
import json
splits_path = '../../data/splits/data_splits.json'
try:
    with open(splits_path, 'r') as f:
        splits = json.load(f)
    print(f"  ✓ Found data splits:")
    print(f"     - Train: {len(splits['train'])} patients")
    print(f"     - Val:   {len(splits['val'])} patients")
    print(f"     - Test:  {len(splits['test'])} patients")
except Exception as e:
    print(f"  ✗ Error: {e}")
    exit(1)

# Test 4: Check your trained models exist
print("\n[4/5] Checking existing trained models...")
baseline_path = '../../models/checkpoints/best_model.pth'
augmented_path = '../../models/checkpoints/augmented_best_model.pth'

if os.path.exists(baseline_path):
    print(f"  ✓ Found baseline model: {baseline_path}")
else:
    print(f"  ! Baseline model not found (this is okay)")

if os.path.exists(augmented_path):
    print(f"  ✓ Found augmented model: {augmented_path}")
    import torch
    checkpoint = torch.load(augmented_path, map_location='cpu')
    print(f"     - Trained to epoch: {checkpoint.get('epoch', 'unknown')}")
    if 'best_dice' in checkpoint:
        print(f"     - Best validation Dice: {checkpoint['best_dice']:.4f}")
else:
    print(f"  ! Augmented model not found at: {augmented_path}")

# Test 5: Verify we can create a metrics calculator
print("\n[5/5] Testing metrics calculator creation...")
try:
    metrics_calc = SegmentationMetrics(num_classes=4)
    print("  ✓ Created SegmentationMetrics calculator")
    print("  ✓ Ready to calculate comprehensive metrics!")
except Exception as e:
    print(f"  ✗ Error: {e}")
    exit(1)

# Success!
print("\n" + "="*70)
print("✅ ALL TESTS PASSED!")
print("="*70)
print("\nYour setup is ready! Next steps:")
print("  1. Run: python3 eval_current_best.py")
print("  2. This will evaluate your current model with comprehensive metrics")
print("  3. You'll get 30+ metrics instead of just Dice scores")
print("="*70)
