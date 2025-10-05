#!/usr/bin/env python3
"""
Evaluation script for COVID-19 Chest X-Ray classification
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.architectures import create_model
from models.ensemble import COVIDEnsemble
from utils.data_loader import MedicalDataHandler, ChestXRayDataset, get_medical_transforms
from utils.metrics import MedicalEvaluator

def load_model(model_path, config, device):
    """Load trained model"""
    model = create_model(
        config['model']['backbone'],
        num_classes=config['model']['num_classes'],
        pretrained=False,
        dropout_rate=config['model']['dropout_rate']
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model

def evaluate_single_model(model, test_loader, device, class_names):
    """Evaluate a single model"""
    model.eval()
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
    
    # Calculate metrics
    evaluator = MedicalEvaluator(class_names)
    report = evaluator.generate_report(
        np.array(all_targets), 
        np.array(all_predictions), 
        np.array(all_probabilities)
    )
    
    # Create visualizations
    evaluator.plot_confusion_matrix(
        np.array(all_targets), 
        np.array(all_predictions)
    )
    
    evaluator.plot_roc_curves(
        np.array(all_targets), 
        np.array(all_probabilities)
    )
    
    evaluator.plot_precision_recall_curves(
        np.array(all_targets), 
        np.array(all_probabilities)
    )
    
    return report, all_predictions, all_probabilities, all_paths

def evaluate_ensemble(model_configs, test_loader, device, class_names):
    """Evaluate ensemble model"""
    ensemble = COVIDEnsemble(model_configs, device)
    
    all_predictions = []
    all_targets = []
    all_probabilities = []
    all_uncertainties = []
    all_paths = []
    
    for data, target, paths in test_loader:
        data = data.to(device)
        
        # Get ensemble predictions with uncertainty
        ensemble_result = ensemble.predict_with_confidence(
            data, 
            confidence_threshold=0.75,
            n_mc_samples=10
        )
        
        all_predictions.extend(ensemble_result['predictions'])
        all_probabilities.extend(ensemble_result['probabilities'])
        all_uncertainties.extend(ensemble_result['uncertainties'])
        all_targets.extend(target.numpy())
        all_paths.extend(paths)
    
    # Calculate metrics
    evaluator = MedicalEvaluator(class_names)
    report = evaluator.generate_report(
        np.array(all_targets), 
        np.array(all_predictions), 
        np.array(all_probabilities)
    )
    
    # Add uncertainty analysis to report
    uncertainty_analysis = {
        'mean_uncertainty': np.mean(all_uncertainties),
        'std_uncertainty': np.std(all_uncertainties),
        'high_uncertainty_threshold': np.percentile(all_uncertainties, 75),
        'num_low_confidence': np.sum(np.array(all_uncertainties) > np.percentile(all_uncertainties, 75))
    }
    report['uncertainty_analysis'] = uncertainty_analysis
    
    print("\n=== UNCERTAINTY ANALYSIS ===")
    print(f"Mean uncertainty: {uncertainty_analysis['mean_uncertainty']:.4f}")
    print(f"High uncertainty cases: {uncertainty_analysis['num_low_confidence']}/{len(all_uncertainties)}")
    
    return report, all_predictions, all_probabilities, all_uncertainties, all_paths

def main():
    parser = argparse.ArgumentParser(description='Evaluate COVID-19 Chest X-Ray Classifier')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--data_csv', type=str, required=True,
                       help='Path to data CSV file')
    parser.add_argument('--image_dir', type=str, required=True,
                       help='Path to image directory')
    parser.add_argument('--model_path', type=str, 
                       help='Path to single model checkpoint')
    parser.add_argument('--ensemble_dir', type=str,
                       help='Directory containing ensemble models')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize data handler
    data_handler = MedicalDataHandler(args.data_csv, args.image_dir)
    splits = data_handler.create_stratified_splits()
    
    # Create test dataset
    _, test_transform = get_medical_transforms()
    test_dataset = ChestXRayDataset(splits['test'], args.image_dir, test_transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=config['data']['batch_size'],
        shuffle=False, 
        num_workers=4
    )
    
    class_names = ['COVID-19', 'Viral Pneumonia', 'Normal']
    
    if args.ensemble_dir:
        print("Evaluating ensemble model...")
        # Load ensemble models
        model_configs = []
        for model_file in os.listdir(args.ensemble_dir):
            if model_file.endswith('.pth'):
                model_configs.append({
                    'backbone': config['model']['backbone'],
                    'num_classes': config['model']['num_classes'],
                    'checkpoint_path': os.path.join(args.ensemble_dir, model_file)
                })
        
        report, predictions, probabilities, uncertainties, paths = evaluate_ensemble(
            model_configs, test_loader, device, class_names
        )
        
        # Save detailed results
        results_df = pd.DataFrame({
            'image_path': paths,
            'true_label': [class_names[i] for i in test_dataset.dataframe['label'].map(test_dataset.class_map)],
            'predicted_label': [class_names[i] for i in predictions],
            'covid_probability': probabilities[:, 0],
            'pneumonia_probability': probabilities[:, 1],
            'normal_probability': probabilities[:, 2],
            'uncertainty': uncertainties
        })
        results_df.to_csv(os.path.join(args.output_dir, 'ensemble_predictions.csv'), index=False)
        
    elif args.model_path:
        print("Evaluating single model...")
        model = load_model(args.model_path, config, device)
        
        report, predictions, probabilities, paths = evaluate_single_model(
            model, test_loader, device, class_names
        )
        
        # Save detailed results
        results_df = pd.DataFrame({
            'image_path': paths,
            'true_label': [class_names[i] for i in test_dataset.dataframe['label'].map(test_dataset.class_map)],
            'predicted_label': [class_names[i] for i in predictions],
            'covid_probability': probabilities[:, 0],
            'pneumonia_probability': probabilities[:, 1],
            'normal_probability': probabilities[:, 2]
        })
        results_df.to_csv(os.path.join(args.output_dir, 'single_model_predictions.csv'), index=False)
    
    else:
        raise ValueError("Either --model_path or --ensemble_dir must be provided")
    
    # Save evaluation report
    with open(os.path.join(args.output_dir, 'evaluation_report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nEvaluation completed! Results saved to {args.output_dir}")

if __name__ == '__main__':
    main()