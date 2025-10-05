#!/usr/bin/env python3
"""
Minimal working example of COVID-19 detection
"""

import os
import sys
import torch
import pandas as pd
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def minimal_example():
    """Minimal working example"""
    print("COVID-19 Detection - Minimal Working Example")
    print("=" * 50)
    
    # 1. Check basic imports
    print("1. Checking basic imports...")
    try:
        import torch
        import torchvision
        import numpy as np
        import pandas as pd
        import sklearn
        from PIL import Image
        print("✓ All basic imports successful")
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    
    # 2. Check data
    print("\n2. Checking data...")
    metadata_path = "data/sample/metadata.csv"
    if not os.path.exists(metadata_path):
        print(f"✗ Metadata file not found: {metadata_path}")
        return False
    
    df = pd.read_csv(metadata_path)
    print(f"✓ Data loaded: {len(df)} samples")
    print(f"  Classes: {df['label'].value_counts().to_dict()}")
    
    # 3. Check our modules
    print("\n3. Checking our modules...")
    try:
        from utils.data_loader import ChestXRayDataset, get_medical_transforms
        from models.architectures import create_model
        print("✓ Our modules imported successfully")
    except ImportError as e:
        print(f"✗ Our module import error: {e}")
        return False
    
    # 4. Test dataset creation
    print("\n4. Testing dataset creation...")
    try:
        transform = get_medical_transforms(is_training=False)
        dataset = ChestXRayDataset(df, "data/sample/images/", transform)
        sample = dataset[0]
        print(f"✓ Dataset created: {len(dataset)} samples")
        print(f"  Sample: image shape {sample[0].shape}, label {sample[1]}")
    except Exception as e:
        print(f"✗ Dataset creation error: {e}")
        return False
    
    # 5. Test model creation
    print("\n5. Testing model creation...")
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = create_model('densenet121', num_classes=3, pretrained=False)
        model.to(device)
        
        # Test forward pass
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224).to(device)
            output = model(dummy_input)
            print(f"✓ Model created: {sum(p.numel() for p in model.parameters()):,} parameters")
            print(f"  Input {dummy_input.shape} -> Output {output.shape}")
    except Exception as e:
        print(f"✗ Model creation error: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 Minimal example completed successfully!")
    print("You can now run the training script.")
    return True

if __name__ == '__main__':
    minimal_example()