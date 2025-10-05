#!/usr/bin/env python3
"""
Windows-optimized training script with command-line arguments
"""

import os
import sys
import argparse
import yaml
import torch
import pandas as pd
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.architectures import create_model
from utils.data_loader import MedicalDataHandler, get_medical_transforms, ChestXRayDataset
from torch.utils.data import DataLoader

class WindowsCOVIDTrainer:
    """Windows-optimized trainer to avoid multiprocessing issues"""
    
    def __init__(self, model, device, class_weights, learning_rate=1e-4):
        self.model = model
        self.device = device
        self.class_weights = class_weights.to(device)
        
        # Setup optimizer and loss
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.criterion = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        
        # Training history
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
    
    def train_epoch(self, dataloader, epoch):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        print(f'Epoch {epoch+1} Training...')
        
        for batch_idx, (data, target, _) in enumerate(dataloader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(output, 1)
            correct_predictions += (predicted == target).sum().item()
            total_samples += target.size(0)
            
            if (batch_idx + 1) % 20 == 0:  # Print less frequently for larger datasets
                current_acc = correct_predictions / total_samples
                print(f'  Batch {batch_idx + 1}/{len(dataloader)}: Loss = {loss.item():.4f}, Acc = {current_acc:.4f}')
        
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = correct_predictions / total_samples
        
        return epoch_loss, epoch_acc
    
    def validate(self, dataloader):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        print('Validating...')
        
        with torch.no_grad():
            for data, target, _ in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                running_loss += loss.item()
                _, predicted = torch.max(output, 1)
                correct_predictions += (predicted == target).sum().item()
                total_samples += target.size(0)
        
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = correct_predictions / total_samples
        
        return epoch_loss, epoch_acc
    
    def train(self, train_loader, val_loader, epochs=10, checkpoint_dir='./checkpoints'):
        """Complete training loop"""
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        best_val_acc = 0.0
        best_model_path = None
        
        print("Starting training...")
        print(f"Training on {len(train_loader.dataset)} samples")
        print(f"Validating on {len(val_loader.dataset)} samples")
        
        for epoch in range(epochs):
            # Training phase
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            
            # Validation phase
            val_loss, val_acc = self.validate(val_loader)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
            print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                
                # Save model checkpoint
                best_model_path = os.path.join(checkpoint_dir, f'best_model_epoch_{epoch+1}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                }, best_model_path)
                print(f'  → New best model saved with val_acc: {val_acc:.4f}')
            
            print('-' * 60)
        
        return {
            'best_val_acc': best_val_acc,
            'best_model_path': best_model_path,
            'history': self.history
        }

def create_windows_data_loaders(train_df, val_df, test_df, image_dir, batch_size=16, img_size=224):
    """Create data loaders optimized for Windows"""
    train_transform = get_medical_transforms(img_size, is_training=True)
    val_transform = get_medical_transforms(img_size, is_training=False)
    
    train_dataset = ChestXRayDataset(train_df, image_dir, train_transform)
    val_dataset = ChestXRayDataset(val_df, image_dir, val_transform)
    test_dataset = ChestXRayDataset(test_df, image_dir, val_transform)
    
    # Windows-optimized: num_workers=0
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Essential for Windows
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    return train_loader, val_loader, test_loader

def main():
    parser = argparse.ArgumentParser(description='Windows COVID-19 Training')
    parser.add_argument('--config', type=str, default='config/covid_radiography.yaml',
                       help='Path to configuration file')
    parser.add_argument('--data_csv', type=str, required=True,
                       help='Path to data CSV file')
    parser.add_argument('--image_dir', type=str, required=True,
                       help='Path to image directory')
    parser.add_argument('--output_dir', type=str, default='./outputs/windows_training',
                       help='Output directory')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                       help='Learning rate')
    
    args = parser.parse_args()
    
    # Load configuration
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}
        print(f"Config file {args.config} not found, using command-line arguments")
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize data handler
    print("Loading dataset...")
    data_handler = MedicalDataHandler(args.data_csv, args.image_dir)
    class_distribution = data_handler.analyze_dataset()
    
    # Create simple train/val/test splits
    splits = data_handler.create_stratified_splits(n_splits=1)
    
    print(f"Train samples: {len(splits['train'])}")
    print(f"Val samples: {len(splits['val'])}")
    print(f"Test samples: {len(splits['test'])}")
    
    # Create Windows-optimized data loaders
    train_loader, val_loader, test_loader = create_windows_data_loaders(
        splits['train'], splits['val'], splits['test'],
        args.image_dir,
        batch_size=args.batch_size,
        img_size=224
    )
    
    # Create model
    print("Creating model...")
    model = create_model(
        'densenet121',
        num_classes=3,
        pretrained=True,
        dropout_rate=0.5
    ).to(device)
    
    # Get class weights (handle class imbalance)
    class_weights = data_handler.get_class_weights()
    print(f"Class weights: {class_weights}")
    
    # Create trainer
    trainer = WindowsCOVIDTrainer(
        model=model,
        device=device,
        class_weights=class_weights,
        learning_rate=args.learning_rate
    )
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    
    result = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        checkpoint_dir=args.output_dir
    )
    
    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {result['best_val_acc']:.4f}")
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, 'final_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'val_accuracy': result['best_val_acc'],
        'history': result['history'],
        'class_weights': class_weights,
        'training_args': vars(args)
    }, final_model_path)
    
    print(f"Model saved to: {final_model_path}")
    
    # Save training history
    history_df = pd.DataFrame(result['history'])
    history_path = os.path.join(args.output_dir, 'training_history.csv')
    history_df.to_csv(history_path, index=False)
    print(f"Training history saved to: {history_path}")

if __name__ == '__main__':
    main()