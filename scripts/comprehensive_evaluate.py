#!/usr/bin/env python3
"""
Comprehensive evaluation of your trained COVID-19 model
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time 
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.architectures import create_model
from utils.data_loader import MedicalDataHandler, get_medical_transforms, ChestXRayDataset
from torch.utils.data import DataLoader

def comprehensive_evaluation(model_path, data_csv, image_dir, output_dir):
    """Comprehensive model evaluation with visualizations"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("🔬 Comprehensive Model Evaluation")
    print("=" * 50)
    
    # Load model
    print("Loading model...")
    checkpoint = torch.load(model_path, map_location='cpu')
    model = create_model('densenet121', num_classes=3, pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Load test data
    data_handler = MedicalDataHandler(data_csv, image_dir)
    splits = data_handler.create_stratified_splits(n_splits=1)
    
    # Create test loader
    transform = get_medical_transforms(is_training=False)
    test_dataset = ChestXRayDataset(splits['test'], image_dir, transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    print(f"Testing on {len(test_dataset)} samples")
    
    # Run inference
    all_predictions = []
    all_targets = []
    all_probabilities = []
    all_paths = []
    
    with torch.no_grad():
        for data, target, paths in test_loader:
            data = data.to(device)
            output = model(data)
            probabilities = torch.softmax(output, dim=1)
            _, predicted = torch.max(output, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_paths.extend(paths)
    
    # Convert to numpy
    predictions = np.array(all_predictions)
    targets = np.array(all_targets)
    probabilities = np.array(all_probabilities)
    
    # Calculate metrics
    accuracy = (predictions == targets).mean()
    class_names = ['COVID-19', 'Viral Pneumonia', 'Normal']
    
    print(f"\n📊 Overall Test Accuracy: {accuracy:.4f}")
    print(f"🏆 Best Validation Accuracy: {checkpoint.get('val_accuracy', 'Unknown'):.4f}")
    
    # Classification report
    print("\n📋 Classification Report:")
    print(classification_report(targets, predictions, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(targets, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Per-class accuracy
    print("\n🎯 Per-Class Accuracy:")
    for i, class_name in enumerate(class_names):
        class_mask = targets == i
        if class_mask.sum() > 0:
            class_accuracy = (predictions[class_mask] == targets[class_mask]).mean()
            print(f"  {class_name}: {class_accuracy:.4f} ({class_mask.sum()} samples)")
    
    # Save detailed results
    results_df = pd.DataFrame({
        'image_path': all_paths,
        'true_label': [class_names[i] for i in targets],
        'predicted_label': [class_names[i] for i in predictions],
        'correct': predictions == targets,
        'covid_prob': probabilities[:, 0],
        'pneumonia_prob': probabilities[:, 1],
        'normal_prob': probabilities[:, 2]
    })
    
    results_path = os.path.join(output_dir, 'detailed_predictions.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\n💾 Detailed results saved to: {results_path}")
    
    return accuracy, results_df

def test_inference_speed(model_path):
    """Test model inference speed"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(model_path, map_location='cpu')
    model = create_model('densenet121', num_classes=3, pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print("\n⚡ Inference Speed Test:")
    
    # Test with batch sizes
    batch_sizes = [1, 4, 8, 16]
    
    for batch_size in batch_sizes:
        dummy_input = torch.randn(batch_size, 3, 224, 224).to(device)
        
        # Warm up
        for _ in range(10):
            with torch.no_grad():
                _ = model(dummy_input)
        
        # Time inference
        times = []
        for _ in range(50):
            start = time.time()
            with torch.no_grad():
                _ = model(dummy_input)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append(time.time() - start)
        
        avg_time = np.mean(times) * 1000  # ms
        throughput = batch_size * 1000 / avg_time  # images/second
        
        print(f"  Batch size {batch_size}: {avg_time:.1f}ms per batch, {throughput:.1f} img/sec")

if __name__ == '__main__':
    # Use your best model
    model_path = "outputs/memory_training/final_model.pth"
    data_csv = "data/processed/metadata.csv"
    image_dir = "data/processed/images/"
    output_dir = "outputs/evaluation_results"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Run comprehensive evaluation
    accuracy, results_df = comprehensive_evaluation(model_path, data_csv, image_dir, output_dir)
    
    # Test inference speed
    test_inference_speed(model_path)
    
    print(f"\n🎉 Evaluation completed! Your model achieves {accuracy:.2%} test accuracy!")