#!/usr/bin/env python3
"""
Evaluate the trained COVID-19 model
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.architectures import create_model
from utils.data_loader import MedicalDataHandler, create_data_loaders
from utils.metrics import MedicalEvaluator

def evaluate_model(model_path, data_csv, image_dir):
    """Evaluate a trained model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    checkpoint = torch.load(model_path, map_location='cpu')
    model = create_model('densenet121', num_classes=3, pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Load data
    data_handler = MedicalDataHandler(data_csv, image_dir)
    splits = data_handler.create_stratified_splits(n_splits=1)
    
    # Create test loader
    _, _, test_loader = create_data_loaders(
        splits['train'], splits['val'], splits['test'],
        image_dir,
        batch_size=8,
        img_size=224,
        use_sampler=False
    )
    
    # Evaluate
    all_predictions = []
    all_targets = []
    all_probabilities = []
    
    with torch.no_grad():
        for data, target, _ in test_loader:
            data = data.to(device)
            output = model(data)
            probabilities = torch.softmax(output, dim=1)
            _, predicted = torch.max(output, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Calculate metrics
    evaluator = MedicalEvaluator()
    report = evaluator.generate_report(
        np.array(all_targets), 
        np.array(all_predictions), 
        np.array(all_probabilities)
    )
    
    # Create visualizations
    evaluator.plot_confusion_matrix(np.array(all_targets), np.array(all_preds))
    evaluator.plot_roc_curves(np.array(all_targets), np.array(all_probabilities))
    
    return report

if __name__ == '__main__':
    model_path = "outputs/windows_training/final_model.pth"
    data_csv = "data/sample/metadata.csv"
    image_dir = "data/sample/images/"
    
    print("Evaluating COVID-19 Detection Model")
    print("=" * 50)
    report = evaluate_model(model_path, data_csv, image_dir)