#!/usr/bin/env python3
"""
Memory-optimized training for GTX 1650 (4GB VRAM)
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

class MemoryOptimizedTrainer:
    """Trainer optimized for 4GB GPU memory"""
    
    def __init__(self, model, device, class_weights, learning_rate=1e-4):
        self.model = model
        self.device = device
        self.class_weights = class_weights.to(device)
        
        # Simple optimizer
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'epoch_times': []
        }
    
    def train_epoch(self, dataloader, epoch):
        """Memory-efficient training"""
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        epoch_start = time.time()
        
        # Clear GPU cache at start of epoch
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
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
            
            # Print progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                current_acc = correct_predictions / total_samples
                print(f'  Batch {batch_idx + 1}/{len(dataloader)}: Loss = {loss.item():.4f}, Acc = {current_acc:.4f}')
                
            # Clear memory periodically
            if self.device.type == 'cuda' and (batch_idx + 1) % 20 == 0:
                torch.cuda.empty_cache()
        
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = correct_predictions / total_samples
        epoch_time = time.time() - epoch_start
        
        return epoch_loss, epoch_acc, epoch_time
    
    def validate(self, dataloader):
        """Memory-efficient validation"""
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
                
                # Clear memory periodically during validation
                if self.device.type == 'cuda' and total_samples % 512 == 0:
                    torch.cuda.empty_cache()
        
        return running_loss / len(dataloader), correct_predictions / total_samples
    
    def train(self, train_loader, val_loader, epochs=10, checkpoint_dir='./checkpoints'):
        """Training loop with memory management"""
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        best_val_acc = 0.0
        best_model_path = None
        
        print("🚀 Starting memory-optimized training...")
        print(f"Training on {len(train_loader.dataset)} samples")
        print(f"Validating on {len(val_loader.dataset)} samples")
        
        for epoch in range(epochs):
            # Clear GPU cache at start of each epoch
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            # Training phase
            train_loss, train_acc, epoch_time = self.train_epoch(train_loader, epoch)
            
            # Validation phase
            val_loss, val_acc = self.validate(val_loader)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['epoch_times'].append(epoch_time)
            
            print(f'Epoch {epoch+1}/{epochs} completed in {epoch_time:.2f}s')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
            print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_path = os.path.join(checkpoint_dir, f'best_model_epoch_{epoch+1}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'val_acc': val_acc,
                }, best_model_path)
                print(f'  💾 New best model saved with val_acc: {val_acc:.4f}')
            
            print('-' * 60)
            
            # Clear GPU cache at end of epoch
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        return {
            'best_val_acc': best_val_acc,
            'best_model_path': best_model_path,
            'history': self.history
        }

def create_memory_optimized_loaders(train_df, val_df, test_df, image_dir, batch_size=16, img_size=224):
    """Create data loaders optimized for 4GB GPU"""
    train_transform = get_medical_transforms(img_size, is_training=True)
    val_transform = get_medical_transforms(img_size, is_training=False)
    
    train_dataset = ChestXRayDataset(train_df, image_dir, train_transform)
    val_dataset = ChestXRayDataset(val_df, image_dir, val_transform)
    test_dataset = ChestXRayDataset(test_df, image_dir, val_transform)
    
    # Memory-optimized settings for 4GB GPU
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,  # Smaller batches for 4GB GPU
        shuffle=True,
        num_workers=0,
        pin_memory=False,  # Disable pin_memory to save RAM
        persistent_workers=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False
    )
    
    return train_loader, val_loader

def main():
    parser = argparse.ArgumentParser(description='Memory-Optimized COVID-19 Training')
    parser.add_argument('--data_csv', type=str, required=True,
                       help='Path to data CSV file')
    parser.add_argument('--image_dir', type=str, required=True,
                       help='Path to image directory')
    parser.add_argument('--output_dir', type=str, default='./outputs/memory_optimized_training',
                       help='Output directory')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,  # Safe for 4GB
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                       help='Learning rate')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print("Loading data...")
    data_handler = MedicalDataHandler(args.data_csv, args.image_dir)
    splits = data_handler.create_stratified_splits(n_splits=1)
    
    print(f"Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")
    
    # Create memory-optimized data loaders
    train_loader, val_loader = create_memory_optimized_loaders(
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
    
    # Get class weights
    class_weights = data_handler.get_class_weights()
    
    # Create memory-optimized trainer
    trainer = MemoryOptimizedTrainer(
        model=model,
        device=device,
        class_weights=class_weights,
        learning_rate=args.learning_rate
    )
    
    print(f"\n🚀 Starting memory-optimized training for {args.epochs} epochs...")
    print(f"📦 Batch size: {args.batch_size} (safe for 4GB GPU)")
    print(f"📚 Learning rate: {args.learning_rate}")
    print("🎯 Features: Periodic GPU cache clearing, memory monitoring")
    
    start_time = time.time()
    result = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        checkpoint_dir=args.output_dir
    )
    
    total_time = time.time() - start_time
    print(f"\n✅ Training completed in {total_time:.1f}s")
    print(f"🏆 Best validation accuracy: {result['best_val_acc']:.4f}")
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'val_accuracy': result['best_val_acc'],
        'history': result['history'],
    }, os.path.join(args.output_dir, 'final_model.pth'))
    
    print(f"💾 Model saved to: {args.output_dir}")
    
    # Save training history
    history_df = pd.DataFrame(result['history'])
    history_path = os.path.join(args.output_dir, 'training_history.csv')
    history_df.to_csv(history_path, index=False)
    print(f"📊 Training history saved")

if __name__ == '__main__':
    main()