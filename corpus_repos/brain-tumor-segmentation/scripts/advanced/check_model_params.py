"""
Quick Helper: Check Your UNet3D Model Definition
================================================
This will show us exactly how your model is defined.
"""
import sys
import os
sys.path.insert(0, os.path.abspath('../'))

print("="*70)
print("CHECKING YOUR UNET3D MODEL DEFINITION")
print("="*70)

# Import your model
from unet_model import UNet3D
import inspect

# Get the __init__ method
init_method = UNet3D.__init__

# Get signature
sig = inspect.signature(init_method)

print("\nYour UNet3D.__init__() signature:")
print("-"*70)
print(f"def __init__{sig}:")
print("-"*70)

# Get parameter names and defaults
params = sig.parameters
print("\nParameters:")
for name, param in params.items():
    if name == 'self':
        continue
    default = param.default
    if default == inspect.Parameter.empty:
        print(f"  - {name}: (no default)")
    else:
        print(f"  - {name}: default = {default}")

# Try to get the source code
print("\n" + "="*70)
print("ATTEMPTING TO SHOW __init__ SOURCE CODE:")
print("="*70)
try:
    source = inspect.getsource(UNet3D.__init__)
    print(source[:500])  # First 500 characters
    if len(source) > 500:
        print("\n... (truncated)")
except Exception as e:
    print(f"Could not retrieve source: {e}")

# Test different initialization approaches
print("\n" + "="*70)
print("TESTING MODEL INITIALIZATION:")
print("="*70)

# Test 1: with num_classes
print("\nTest 1: UNet3D(in_channels=4, num_classes=4, base_channels=32)")
try:
    model = UNet3D(in_channels=4, num_classes=4, base_channels=32)
    print("  ✓ SUCCESS with num_classes parameter")
    working_params = "num_classes"
except Exception as e:
    print(f"  ✗ Failed: {e}")
    working_params = None

# Test 2: with out_channels
if working_params is None:
    print("\nTest 2: UNet3D(in_channels=4, out_channels=4, base_channels=32)")
    try:
        model = UNet3D(in_channels=4, out_channels=4, base_channels=32)
        print("  ✓ SUCCESS with out_channels parameter")
        working_params = "out_channels"
    except Exception as e:
        print(f"  ✗ Failed: {e}")

# Test 3: without output param
if working_params is None:
    print("\nTest 3: UNet3D(in_channels=4, base_channels=32)")
    try:
        model = UNet3D(in_channels=4, base_channels=32)
        print("  ✓ SUCCESS without output parameter")
        working_params = "no_output_param"
    except Exception as e:
        print(f"  ✗ Failed: {e}")

# Test 4: with n_classes
if working_params is None:
    print("\nTest 4: UNet3D(in_channels=4, n_classes=4, base_channels=32)")
    try:
        model = UNet3D(in_channels=4, n_classes=4, base_channels=32)
        print("  ✓ SUCCESS with n_classes parameter")
        working_params = "n_classes"
    except Exception as e:
        print(f"  ✗ Failed: {e}")

print("\n" + "="*70)
if working_params:
    print(f"✅ WORKING INITIALIZATION METHOD: {working_params}")
    print("="*70)
    
    if working_params == "num_classes":
        print("\nUse this in your scripts:")
        print("  model = UNet3D(in_channels=4, num_classes=4, base_channels=32)")
    elif working_params == "out_channels":
        print("\nUse this in your scripts:")
        print("  model = UNet3D(in_channels=4, out_channels=4, base_channels=32)")
    elif working_params == "no_output_param":
        print("\nUse this in your scripts:")
        print("  model = UNet3D(in_channels=4, base_channels=32)")
    elif working_params == "n_classes":
        print("\nUse this in your scripts:")
        print("  model = UNet3D(in_channels=4, n_classes=4, base_channels=32)")
else:
    print("❌ COULD NOT FIND WORKING INITIALIZATION")
    print("="*70)
    print("\nPlease check your unet_model.py file manually.")
    print("Location: ../unet_model.py")
print("="*70)
