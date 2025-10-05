#!/usr/bin/env python3
"""
Training script for COVID-19 Chest X-Ray classification
"""

import os
import sys
import argparse
import yaml
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.architectures import create_model
from utils.data_loader import MedicalDataHandler, create_data_loaders
from training.trainer import COVIDTrainer, CrossValidationTrainer
from training.cross_validation import save_cross_validation_results

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_environment(config):
    """Setup training environment"""
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Create output directories
    os.makedirs(config['paths']['output_dir'], exist_ok=True)
    os.makedirs(config['paths']['model_dir'], exist_ok=True)
    os.makedirs(config['paths']['log_dir'], exist_ok=True)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"CUDA version: {torch.version.cuda}")
    
    return device

def main():
    parser = argparse.ArgumentParser(description='Train COVID-19 Chest X-Ray Classifier')
    parser.add_argument('--config', type=str, default='config/default.yaml',
                       help='Path to configuration file')
    parser.add_argument('--data_csv', type=str, required=True,
                       help='Path to data CSV file')
    parser.add_argument('--image_dir', type=str, required=True,
                       help='Path to image directory')
    parser.add_argument('--cross_validation', action='store_true',
                       help='Use cross-validation')
    parser.add_argument('--num_folds', type=int, default=5,
                       help='Number of folds for cross-validation')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup environment
    device = setup_environment(config)
    
    # Initialize data handler
    data_handler = MedicalDataHandler(args.data_csv, args.image_dir)
    data_handler.analyze_dataset()
    
    if args.cross_validation:
        print("Running cross-validation...")
        # Run cross-validation
        cv_trainer = CrossValidationTrainer(
            data_handler=data_handler,
            model_factory=create_model,
            device=device,
            config=config
        )
        
        cv_results = cv_trainer.run_cross_validation(n_splits=args.num_folds)
        
        # Save results
        summary = save_cross_validation_results(
            cv_results, 
            os.path.join(config['paths']['output_dir'], 'cross_validation')
        )
        
        print("\nCross-Validation Completed!")
        print(f"Mean Accuracy: {summary['mean_accuracy']:.4f} ± {summary['std_accuracy']:.4f}")
        
    else:
        print("Training single model...")
        # Create train/val/test splits
        splits = data_handler.create_stratified_splits()
        
        # Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders(
            splits['train'], splits['val'], splits['test'],
            args.image_dir,
            batch_size=config['data']['batch_size'],
            img_size=config['data']['image_size'][0]
        )
        
        # Create model
        model = create_model(
            config['model']['backbone'],
            num_classes=config['model']['num_classes'],
            pretrained=config['model']['pretrained'],
            dropout_rate=config['model']['dropout_rate']
        ).to(device)
        
        # Get class weights
        class_weights = data_handler.get_class_weights()
        
        # Create trainer
        trainer = COVIDTrainer(
            model=model,
            device=device,
            class_weights=class_weights,
            optimizer_config=config['training'],
            scheduler_config=config.get('scheduler')
        )
        
        # Train model
        result = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=config['training']['epochs'],
            early_stopping_patience=config['training']['early_stopping_patience'],
            checkpoint_dir=config['paths']['model_dir']
        )
        
        print(f"\nTraining completed!")
        print(f"Best validation accuracy: {result['best_val_acc']:.4f}")
        
        # Save final model
        final_model_path = os.path.join(config['paths']['model_dir'], 'final_model.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'class_weights': class_weights,
            'val_accuracy': result['best_val_acc'],
            'history': result['history']
        }, final_model_path)
        
        print(f"Final model saved to: {final_model_path}")

if __name__ == '__main__':
    main()