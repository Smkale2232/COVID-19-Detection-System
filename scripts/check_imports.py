#!/usr/bin/env python3
"""
Quick script to check all imports
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def check_import(module_name, package_name=None):
    """Check if a module can be imported"""
    try:
        if package_name:
            __import__(package_name)
            print(f"✓ {module_name} (from {package_name})")
        else:
            __import__(module_name)
            print(f"✓ {module_name}")
        return True
    except ImportError as e:
        print(f"✗ {module_name}: {e}")
        return False

def main():
    print("Checking imports...")
    print("=" * 50)
    
    # Core dependencies
    core_imports = [
        ('torch', None),
        ('torchvision', None),
        ('numpy', None),
        ('pandas', None),
        ('sklearn', None),
        ('PIL', 'PIL.Image'),
        ('tqdm', None),
    ]
    
    # Our modules
    our_modules = [
        ('utils.data_loader', None),
        ('models.architectures', None),
        ('training.simple_trainer', None),
    ]
    
    print("Core dependencies:")
    core_passed = 0
    for module, package in core_imports:
        if check_import(module, package):
            core_passed += 1
    
    print("\nOur modules:")
    our_passed = 0
    for module, package in our_modules:
        if check_import(module, package):
            our_passed += 1
    
    print(f"\n{'=' * 50}")
    print(f"Core dependencies: {core_passed}/{len(core_imports)}")
    print(f"Our modules: {our_passed}/{len(our_modules)}")
    
    if core_passed == len(core_imports) and our_passed == len(our_modules):
        print("🎉 All imports successful!")
    else:
        print("❌ Some imports failed.")

if __name__ == '__main__':
    main()