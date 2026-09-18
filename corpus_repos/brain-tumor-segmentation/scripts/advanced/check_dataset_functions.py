"""
Check Available Functions in create_dataset.py
==============================================
This will show us what functions you actually have.
"""
import sys
import os
sys.path.insert(0, os.path.abspath('../'))

print("="*70)
print("CHECKING YOUR create_dataset.py")
print("="*70)

import create_dataset
import inspect

# Get all functions and classes
print("\nAvailable functions and classes:")
print("-"*70)

for name, obj in inspect.getmembers(create_dataset):
    if not name.startswith('_'):  # Skip private members
        if inspect.isfunction(obj):
            sig = inspect.signature(obj)
            print(f"  Function: {name}{sig}")
        elif inspect.isclass(obj):
            print(f"  Class: {name}")

# Check specifically for common function names
print("\n" + "="*70)
print("CHECKING FOR DATA LOADING FUNCTIONS:")
print("="*70)

common_names = [
    'create_data_loaders',
    'get_data_loaders', 
    'create_dataloaders',
    'get_dataloaders',
    'create_loaders',
    'get_loaders',
]

found_function = None
for name in common_names:
    if hasattr(create_dataset, name):
        print(f"  ✓ Found: {name}()")
        found_function = name
        break
else:
    print("  ✗ None of the common function names found")

# Check for dataset class
print("\n" + "="*70)
print("CHECKING FOR DATASET CLASS:")
print("="*70)

common_class_names = [
    'BraTSSegmentationDataset',
    'BraTSDataset',
    'SegmentationDataset',
]

found_class = None
for name in common_class_names:
    if hasattr(create_dataset, name):
        print(f"  ✓ Found: {name}")
        found_class = name
        break
else:
    print("  ✗ None of the common class names found")

print("\n" + "="*70)
print("RECOMMENDATION:")
print("="*70)

if found_function:
    print(f"\n✅ Use this function to create data loaders:")
    print(f"   from create_dataset import {found_function}")
elif found_class:
    print(f"\n✅ Use this class to create dataset:")
    print(f"   from create_dataset import {found_class}")
    print(f"   # Then create DataLoader manually")
else:
    print("\n⚠️  Need to inspect create_dataset.py manually")
    print("   Location: ../create_dataset.py")

print("="*70)
