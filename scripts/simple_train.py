#!/usr/bin/env python3
"""
Simple training script without cross-validation
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

# Try importing from simple_trainer first, fall back to trainer
try:
    from training.simple_trainer import SimpleCOVIDTrainer as COVIDTrainer
    print("Using SimpleCOVIDTrainer")
except ImportError:
    try:
        from training.trainer import COVIDTrainer
        print("Using COVIDTrainer")
    except ImportError as e:
        print(f"Error importing trainer: {e}")
        sys.exit(1)

def safe_get_config(config, keys, default=None):
    """Safely get config value with type conversion"""
    value = config
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    
    # Convert to appropriate type
    if isinstance(value, str):
        try:
            if '.' in value or 'e' in value.lower():
                return float(value)
            else:
                return int(value)
        except ValueError:
            return value
    return value

def main():
    parser = argparse.ArgumentParser(description='Simple COVID-19 Training')
    parser.add_argument('--config', type=str, default='config/covid_radiography.yaml',
                       help='Path to configuration file')
    parser.add_argument('--data_csv', type=str, required=True,
                       help='Path to data CSV file')
    parser.add_argument('--image_dir', type=str, required=True,
                       help='Path to image directory')
    parser.add_argument('--output_dir', type=str, default='./outputs/simple_training',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize data handler
    data_handler = MedicalDataHandler(args.data_csv, args.image_dir)
    data_handler.analyze_dataset()
    
    # Create simple train/val/test splits (no cross-validation)
    splits = data_handler.create_stratified_splits(n_splits=1)  # Single split
    
    print(f"Train samples: {len(splits['train'])}")
    print(f"Val samples: {len(splits['val'])}")
    print(f"Test samples: {len(splits['test'])}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        splits['train'], splits['val'], splits['test'],
        args.image_dir,
        batch_size=safe_get_config(config, ['data', 'batch_size'], 16),
        img_size=safe_get_config(config, ['data', 'image_size', 0], 224),
        use_sampler=True
    )
    
    # Create model
    model = create_model(
        safe_get_config(config, ['model', 'backbone'], 'densenet121'),
        num_classes=safe_get_config(config, ['model', 'num_classes'], 3),
        pretrained=safe_get_config(config, ['model', 'pretrained'], True),
        dropout_rate=safe_get_config(config, ['model', 'dropout_rate'], 0.5)
    ).to(device)
    
    # Get class weights
    class_weights = data_handler.get_class_weights()
    print(f"Class weights: {class_weights}")
    
    # Create trainer with safe config values
    learning_rate = safe_get_config(config, ['training', 'learning_rate'], 0.0001)
    print(f"Learning rate: {learning_rate}")
    
    trainer = COVIDTrainer(
        model=model,
        device=device,
        class_weights=class_weights,
        learning_rate=learning_rate
    )
    
    # Train with fewer epochs for testing
    epochs = min(safe_get_config(config, ['training', 'epochs'], 10), 5)  # Reduced for testing
    
    print(f"\nStarting training for {epochs} epochs...")
    result = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        checkpoint_dir=args.output_dir
    )
    
    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {result['best_val_acc']:.4f}")
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, 'final_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'class_weights': class_weights,
        'val_accuracy': result['best_val_acc'],
        'history': result['history']
    }, final_model_path)
    
    print(f"Model saved to: {final_model_path}")

if __name__ == '__main__':
    main()