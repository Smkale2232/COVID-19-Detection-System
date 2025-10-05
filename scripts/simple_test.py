#!/usr/bin/env python3
"""
Simple test script to verify basic functionality
"""

import os
import sys
import torch
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

def test_basic_imports():
    """Test basic imports"""
    print("Testing basic imports...")
    try:
        import torch
        import torchvision
        import numpy as np
        import pandas as pd
        import sklearn
        print("✓ All basic imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_data_loading():
    """Test data loading"""
    print("\nTesting data loading...")
    try:
        from utils.data_loader import MedicalDataHandler
        
        data_handler = MedicalDataHandler("data/sample/metadata.csv", "data/sample/images/")
        df = data_handler.df
        print(f"✓ Data loaded successfully: {len(df)} samples")
        print(f"  Classes: {df['label'].value_counts().to_dict()}")
        return True
    except Exception as e:
        print(f"✗ Data loading error: {e}")
        return False

def test_model_creation():
    """Test model creation"""
    print("\nTesting model creation...")
    try:
        from models.architectures import create_model
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Test DenseNet
        model = create_model('densenet121', num_classes=3, pretrained=False)
        print(f"✓ DenseNet121 created: {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Test ResNet
        model = create_model('resnet50', num_classes=3, pretrained=False)
        print(f"✓ ResNet50 created: {sum(p.numel() for p in model.parameters()):,} parameters")
        
        return True
    except Exception as e:
        print(f"✗ Model creation error: {e}")
        return False

def main():
    print("COVID-19 Detection System - Simple Test")
    print("=" * 50)
    
    tests = [
        test_basic_imports,
        test_data_loading, 
        test_model_creation
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n{'=' * 50}")
    print(f"Tests passed: {passed}/{len(tests)}")
    
    if passed == len(tests):
        print("🎉 All tests passed! You can proceed with training.")
        print("\nNext steps:")
        print("1. python scripts/train.py --config config/covid_radiography.yaml --data_csv data/sample/metadata.csv --image_dir data/sample/images/ --output_dir outputs/sample_training")
        print("2. Check outputs/sample_training for results")
    else:
        print("❌ Some tests failed. Please fix the issues above.")

if __name__ == '__main__':
    main()