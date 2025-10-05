#!/usr/bin/env python3
"""
Quick evaluation of your trained model
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add project root to path - FIXED
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.architectures import create_model

# Load your amazing model
model_path = "outputs/full_training/final_model.pth"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("🚀 Loading your 98.46% accurate model...")
checkpoint = torch.load(model_path, map_location='cpu')
model = create_model('densenet121', num_classes=3, pretrained=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

print("✅ Model loaded successfully!")
print(f"🏆 Validation accuracy: {checkpoint.get('val_accuracy', 'Unknown'):.4f}")

# Test inference speed
print("\n🧪 Testing inference speed...")
dummy_input = torch.randn(1, 3, 224, 224).to(device)

# Warm up
for _ in range(10):
    with torch.no_grad():
        _ = model(dummy_input)

# Time inference
import time
times = []
for _ in range(100):
    start = time.time()
    with torch.no_grad():
        _ = model(dummy_input)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    times.append(time.time() - start)

avg_time = np.mean(times) * 1000  # Convert to milliseconds
print(f"⏱️ Average inference time: {avg_time:.1f}ms per image")
print(f"📊 Throughput: {1000/avg_time:.1f} images/second")

if device.type == 'cuda':
    print(f"🎯 GPU memory allocated: {torch.cuda.memory_allocated()/1024**2:.1f} MB")