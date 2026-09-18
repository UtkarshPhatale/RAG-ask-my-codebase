"""
Check BraTSSegmentationDataset Parameters
=========================================
This will show us exactly how to initialize your dataset class.
"""
import sys
import os
sys.path.insert(0, os.path.abspath('../'))

print("="*70)
print("CHECKING BraTSSegmentationDataset CLASS")
print("="*70)

from create_dataset import BraTSSegmentationDataset
import inspect

# Get the __init__ signature
sig = inspect.signature(BraTSSegmentationDataset.__init__)

print("\nBraTSSegmentationDataset.__init__() signature:")
print("-"*70)
print(f"def __init__{sig}:")
print("-"*70)

# Get parameters
params = sig.parameters
print("\nParameters:")
for name, param in params.items():
    if name == 'self':
        continue
    default = param.default
    if default == inspect.Parameter.empty:
        print(f"  - {name}: (required)")
    else:
        print(f"  - {name}: default = {default}")

# Try to get source code
print("\n" + "="*70)
print("SOURCE CODE (first 30 lines):")
print("="*70)
try:
    source = inspect.getsource(BraTSSegmentationDataset.__init__)
    lines = source.split('\n')[:30]
    for i, line in enumerate(lines, 1):
        print(f"{i:3}: {line}")
except Exception as e:
    print(f"Could not retrieve source: {e}")

# Test different initialization patterns
print("\n" + "="*70)
print("TESTING INITIALIZATION PATTERNS:")
print("="*70)

import json

# Load splits for testing
with open('../../data/splits/data_splits.json', 'r') as f:
    splits = json.load(f)

test_patients = splits['test'][:2]  # Just test with 2 patients

# Pattern 1: root_dir + patient_ids
print("\nPattern 1: BraTSSegmentationDataset(root_dir='...', patient_ids=[...])")
try:
    dataset = BraTSSegmentationDataset(
        root_dir='../../data/raw/brats_gli/training_data1_v2',
        patient_ids=test_patients
    )
    print("  ✓ SUCCESS with root_dir + patient_ids")
    working_pattern = "root_dir"
except Exception as e:
    print(f"  ✗ Failed: {e}")
    working_pattern = None

# Pattern 2: data_path + patient_ids
if working_pattern is None:
    print("\nPattern 2: BraTSSegmentationDataset(data_path='...', patient_ids=[...])")
    try:
        dataset = BraTSSegmentationDataset(
            data_path='../../data/raw/brats_gli/training_data1_v2',
            patient_ids=test_patients
        )
        print("  ✓ SUCCESS with data_path + patient_ids")
        working_pattern = "data_path"
    except Exception as e:
        print(f"  ✗ Failed: {e}")

# Pattern 3: base_dir + patient_ids
if working_pattern is None:
    print("\nPattern 3: BraTSSegmentationDataset(base_dir='...', patient_ids=[...])")
    try:
        dataset = BraTSSegmentationDataset(
            base_dir='../../data/raw/brats_gli/training_data1_v2',
            patient_ids=test_patients
        )
        print("  ✓ SUCCESS with base_dir + patient_ids")
        working_pattern = "base_dir"
    except Exception as e:
        print(f"  ✗ Failed: {e}")

# Pattern 4: patient_ids only (data_dir hardcoded?)
if working_pattern is None:
    print("\nPattern 4: BraTSSegmentationDataset(patient_ids=[...])")
    try:
        dataset = BraTSSegmentationDataset(patient_ids=test_patients)
        print("  ✓ SUCCESS with only patient_ids")
        working_pattern = "patient_ids_only"
    except Exception as e:
        print(f"  ✗ Failed: {e}")

# Pattern 5: Just check all parameters directly from signature
if working_pattern is None:
    print("\nPattern 5: Testing with actual signature parameters...")
    try:
        # Get non-default parameters
        required_params = []
        optional_params = {}
        
        for name, param in params.items():
            if name == 'self':
                continue
            if param.default == inspect.Parameter.empty:
                required_params.append(name)
            else:
                optional_params[name] = param.default
        
        print(f"  Required parameters: {required_params}")
        print(f"  Optional parameters: {optional_params}")
        
        # Try to construct with required params
        if len(required_params) == 1 and required_params[0] == 'patient_ids':
            dataset = BraTSSegmentationDataset(patient_ids=test_patients)
            print("  ✓ SUCCESS with signature-based init")
            working_pattern = "signature"
        elif len(required_params) == 2:
            param1, param2 = required_params
            dataset = BraTSSegmentationDataset(
                **{param1: '../../data/raw/brats_gli/training_data1_v2',
                   param2: test_patients}
            )
            print("  ✓ SUCCESS with signature-based init")
            working_pattern = "signature"
    except Exception as e:
        print(f"  ✗ Failed: {e}")

print("\n" + "="*70)
if working_pattern:
    print(f"✅ WORKING INITIALIZATION PATTERN: {working_pattern}")
    print("="*70)
    
    if working_pattern == "root_dir":
        print("\nUse this in your scripts:")
        print("  dataset = BraTSSegmentationDataset(")
        print("      root_dir='../../data/raw/brats_gli/training_data1_v2',")
        print("      patient_ids=splits['test']")
        print("  )")
    elif working_pattern == "data_path":
        print("\nUse this in your scripts:")
        print("  dataset = BraTSSegmentationDataset(")
        print("      data_path='../../data/raw/brats_gli/training_data1_v2',")
        print("      patient_ids=splits['test']")
        print("  )")
    elif working_pattern == "base_dir":
        print("\nUse this in your scripts:")
        print("  dataset = BraTSSegmentationDataset(")
        print("      base_dir='../../data/raw/brats_gli/training_data1_v2',")
        print("      patient_ids=splits['test']")
        print("  )")
    elif working_pattern == "patient_ids_only":
        print("\nUse this in your scripts:")
        print("  dataset = BraTSSegmentationDataset(patient_ids=splits['test'])")
    elif working_pattern == "signature":
        print("\nUse this in your scripts:")
        print("  # Use the parameters shown above")
else:
    print("❌ COULD NOT FIND WORKING PATTERN")
    print("="*70)
    print("\nPlease share the __init__ signature printed above")
    print("so I can create the correct initialization code.")

print("="*70)
