#!/usr/bin/env python3
"""
Complete pipeline for COVID-19 Chest X-Ray detection
"""

import os
import sys
import argparse
import yaml
import torch
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from scripts.train import main as train_main
from scripts.evaluate import main as evaluate_main
from scripts.predict import main as predict_main

def run_complete_pipeline(config_path, data_csv, image_dir, output_dir):
    """Run complete training, evaluation, and prediction pipeline"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update paths in config
    config['paths']['output_dir'] = output_dir
    config['paths']['model_dir'] = os.path.join(output_dir, 'models')
    config['paths']['log_dir'] = os.path.join(output_dir, 'logs')
    
    # Save updated config
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    
    print("=" * 60)
    print("COVID-19 CHEST X-RAY DETECTION PIPELINE")
    print("=" * 60)
    
    # Step 1: Training with Cross-Validation
    print("\n1. TRAINING WITH CROSS-VALIDATION")
    print("-" * 40)
    
    train_args = argparse.Namespace(
        config=config_path,
        data_csv=data_csv,
        image_dir=image_dir,
        cross_validation=True,
        num_folds=5
    )
    
    # Modify sys.argv for train script
    old_argv = sys.argv
    try:
        sys.argv = ['train.py'] + [
            '--config', config_path,
            '--data_csv', data_csv,
            '--image_dir', image_dir,
            '--cross_validation',
            '--num_folds', '5'
        ]
        train_main()
    except SystemExit:
        pass  # argparse exits after help
    finally:
        sys.argv = old_argv
    
    # Step 2: Ensemble Evaluation
    print("\n2. ENSEMBLE EVALUATION")
    print("-" * 40)
    
    ensemble_dir = os.path.join(output_dir, 'models')
    evaluate_args = argparse.Namespace(
        config=os.path.join(output_dir, 'config.yaml'),
        data_csv=data_csv,
        image_dir=image_dir,
        ensemble_dir=ensemble_dir,
        output_dir=os.path.join(output_dir, 'evaluation')
    )
    
    try:
        sys.argv = ['evaluate.py'] + [
            '--config', os.path.join(output_dir, 'config.yaml'),
            '--data_csv', data_csv,
            '--image_dir', image_dir,
            '--ensemble_dir', ensemble_dir,
            '--output_dir', os.path.join(output_dir, 'evaluation')
        ]
        evaluate_main()
    except SystemExit:
        pass
    finally:
        sys.argv = old_argv
    
    # Step 3: Generate Sample Predictions
    print("\n3. SAMPLE PREDICTIONS")
    print("-" * 40)
    
    # Use test images for demonstration
    test_predictions_dir = os.path.join(output_dir, 'predictions')
    os.makedirs(test_predictions_dir, exist_ok=True)
    
    # Create a sample of test images for prediction demonstration
    data_handler = MedicalDataHandler(data_csv, image_dir)
    splits = data_handler.create_stratified_splits()
    test_samples = splits['test'].sample(min(10, len(splits['test'])))
    
    # Save sample paths
    sample_csv = os.path.join(test_predictions_dir, 'sample_images.csv')
    test_samples.to_csv(sample_csv, index=False)
    
    predict_args = argparse.Namespace(
        config=os.path.join(output_dir, 'config.yaml'),
        input=image_dir,  # Using full directory for batch prediction
        ensemble_dir=ensemble_dir,
        output=os.path.join(test_predictions_dir, 'batch_predictions.csv'),
        use_uncertainty=True
    )
    
    try:
        sys.argv = ['predict.py'] + [
            '--config', os.path.join(output_dir, 'config.yaml'),
            '--input', image_dir,
            '--ensemble_dir', ensemble_dir,
            '--output', os.path.join(test_predictions_dir, 'batch_predictions.csv'),
            '--use_uncertainty'
        ]
        predict_main()
    except SystemExit:
        pass
    finally:
        sys.argv = old_argv
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"Results saved to: {output_dir}")
    print("\nGenerated outputs:")
    print(f"  - Models: {os.path.join(output_dir, 'models')}")
    print(f"  - Evaluation: {os.path.join(output_dir, 'evaluation')}")
    print(f"  - Predictions: {os.path.join(output_dir, 'predictions')}")
    print(f"  - Logs: {os.path.join(output_dir, 'logs')}")

def main():
    parser = argparse.ArgumentParser(description='COVID-19 Detection Pipeline')
    parser.add_argument('--config', type=str, default='config/default.yaml',
                       help='Path to configuration file')
    parser.add_argument('--data_csv', type=str, required=True,
                       help='Path to data CSV file')
    parser.add_argument('--image_dir', type=str, required=True,
                       help='Path to image directory')
    parser.add_argument('--output_dir', type=str, default='./pipeline_output',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    run_complete_pipeline(
        args.config,
        args.data_csv,
        args.image_dir,
        args.output_dir
    )

if __name__ == '__main__':
    main()