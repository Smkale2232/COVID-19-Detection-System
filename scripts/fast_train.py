#!/usr/bin/env python3
"""
Fast GPU-optimized training script - simplified and actually faster
"""

import os
import sys
import argparse
import torch
import pandas as pd
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.architectures import create_model
from utils.data_loader import MedicalDataHandler, get_medical_transforms, ChestXRayDataset
from torch.utils.data import DataLoader

class FastCOVIDTrainer:
    """Fast trainer - minimal overhead, maximum GPU usage"""
    
    def __init__(self, model, device, class_weights, learning_rate=1e-4):
        self.model = model
        self.device = device
        self.class_weights = class_weights.to(device)
        
        # Simple optimizer - no fancy stuff
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'epoch_times': []
        }
    
    def train_epoch(self, dataloader, epoch):
        """Fast training - minimal printing, maximum GPU"""
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        epoch_start = time.time()
        
        # Minimal progress indicator
        if epoch % 5 == 0:  # Only print every 5 epochs
            print(f'Epoch {epoch+1} Training...')
        
        for batch_idx, (data, target, _) in enumerate(dataloader):
            # Fast data transfer
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
        
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = correct_predictions / total_samples
        epoch_time = time.time() - epoch_start
        
        return epoch_loss, epoch_acc, epoch_time
    
    def validate(self, dataloader):
        """Fast validation"""
        self.model.eval()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for data, target, _ in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                running_loss += loss.item()
                _, predicted = torch.max(output, 1)
                correct_predictions += (predicted == target).sum().item()
                total_samples += target.size(0)
        
        return running_loss / len(dataloader), correct_predictions / total_samples
    
    def train(self, train_loader, val_loader, epochs=10, checkpoint_dir='./checkpoints'):
        """Fast training loop"""
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        best_val_acc = 0.0
        best_model_path = None
        
        print("🚀 Starting FAST training...")
        print(f"Training on {len(train_loader.dataset)} samples")
        print(f"Validating on {len(val_loader.dataset)} samples")
        print("Progress: [", end='')
        
        for epoch in range(epochs):
            # Training
            train_loss, train_acc, epoch_time = self.train_epoch(train_loader, epoch)
            
            # Validation (less frequent to save time)
            if epoch % 2 == 0 or epoch == epochs - 1:  # Validate every 2 epochs
                val_loss, val_acc = self.validate(val_loader)
            else:
                val_loss, val_acc = 0, 0  # Skip validation this epoch
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['epoch_times'].append(epoch_time)
            
            # Simple progress bar
            print('=', end='', flush=True)
            
            # Save best model (only when we validate)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_path = os.path.join(checkpoint_dir, f'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'val_acc': val_acc,
                }, best_model_path)
        
        print('] 100%')
        return {
            'best_val_acc': best_val_acc,
            'best_model_path': best_model_path,
            'history': self.history
        }

def create_fast_data_loaders(train_df, val_df, test_df, image_dir, batch_size=64, img_size=224):
    """Create fast data loaders - larger batches, minimal overhead"""
    train_transform = get_medical_transforms(img_size, is_training=True)
    val_transform = get_medical_transforms(img_size, is_training=False)
    
    train_dataset = ChestXRayDataset(train_df, image_dir, train_transform)
    val_dataset = ChestXRayDataset(val_df, image_dir, val_transform)
    test_dataset = ChestXRayDataset(test_df, image_dir, val_transform)
    
    # Fast settings - larger batches, no workers
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,  # Much larger batches
        shuffle=True,
        num_workers=0,
        pin_memory=False  # Simpler is faster sometimes
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    return train_loader, val_loader

def main():
    parser = argparse.ArgumentParser(description='FAST COVID-19 Training')
    parser.add_argument('--data_csv', type=str, required=True,
                       help='Path to data CSV file')
    parser.add_argument('--image_dir', type=str, required=True,
                       help='Path to image directory')
    parser.add_argument('--output_dir', type=str, default='./outputs/fast_training',
                       help='Output directory')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,  # Much larger
                       help='Batch size for training')
    
    args = parser.parse_args()
    
    # Simple device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print("Loading data...")
    data_handler = MedicalDataHandler(args.data_csv, args.image_dir)
    splits = data_handler.create_stratified_splits(n_splits=1)
    
    print(f"Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")
    
    # Create fast data loaders
    train_loader, val_loader = create_fast_data_loaders(
        splits['train'], splits['val'], splits['test'],
        args.image_dir,
        batch_size=args.batch_size
    )
    
    # Create model
    print("Creating model...")
    model = create_model(
        'densenet121',
        num_classes=3,
        pretrained=True,
        dropout_rate=0.5
    ).to(device)
    
    # Simple class weights
    class_weights = data_handler.get_class_weights()
    
    # Create fast trainer
    trainer = FastCOVIDTrainer(
        model=model,
        device=device,
        class_weights=class_weights,
        learning_rate=0.001  # Slightly higher for faster convergence
    )
    
    print(f"\n🚀 Starting FAST training for {args.epochs} epochs...")
    print(f"📦 Batch size: {args.batch_size} (larger = faster)")
    start_time = time.time()
    
    result = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        checkpoint_dir=args.output_dir
    )
    
    total_time = time.time() - start_time
    print(f"\n✅ Training completed in {total_time:.1f}s ({total_time/args.epochs:.1f}s per epoch)")
    print(f"🏆 Best validation accuracy: {result['best_val_acc']:.4f}")
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'val_accuracy': result['best_val_acc'],
    }, os.path.join(args.output_dir, 'final_model.pth'))
    
    print(f"💾 Model saved")

if __name__ == '__main__':
    main()