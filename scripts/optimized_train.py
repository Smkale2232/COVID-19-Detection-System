#!/usr/bin/env python3
"""
GPU-optimized training script for COVID-19 detection
"""

import os
import sys
import argparse
import yaml
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

def check_gpu_usage():
    """Check if GPU is available and being used"""
    if torch.cuda.is_available():
        print(f"🎯 GPU: {torch.cuda.get_device_name()}")
        print(f"🎯 CUDA Version: {torch.version.cuda}")
        print(f"🎯 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        return True
    else:
        print("❌ No GPU available, using CPU")
        return False

class OptimizedCOVIDTrainer:
    """GPU-optimized trainer for faster training"""
    
    def __init__(self, model, device, class_weights, learning_rate=1e-4):
        self.model = model
        self.device = device
        self.class_weights = class_weights.to(device)
        
        # Mixed precision training for GPU acceleration
        self.scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
        
        # Setup optimizer and loss
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.criterion = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        
        # Training history
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'epoch_times': []
        }
    
    def train_epoch(self, dataloader, epoch):
        """Train for one epoch with GPU optimization"""
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        epoch_start = time.time()
        
        print(f'Epoch {epoch+1} Training...')
        
        for batch_idx, (data, target, _) in enumerate(dataloader):
            data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            
            # Mixed precision training for GPU
            if self.device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    output = self.model(data)
                    loss = self.criterion(output, target)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(output, 1)
            correct_predictions += (predicted == target).sum().item()
            total_samples += target.size(0)
            
            if (batch_idx + 1) % 50 == 0:  # Print less frequently
                current_acc = correct_predictions / total_samples
                print(f'  Batch {batch_idx + 1}/{len(dataloader)}: Loss = {loss.item():.4f}, Acc = {current_acc:.4f}')
        
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = correct_predictions / total_samples
        epoch_time = time.time() - epoch_start
        
        return epoch_loss, epoch_acc, epoch_time
    
    def validate(self, dataloader):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        print('Validating...')
        
        with torch.no_grad():
            for data, target, _ in dataloader:
                data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
                
                # Mixed precision for validation too
                if self.device.type == 'cuda':
                    with torch.cuda.amp.autocast():
                        output = self.model(data)
                        loss = self.criterion(output, target)
                else:
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
        """Complete training loop with timing"""
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        best_val_acc = 0.0
        best_model_path = None
        
        print("🚀 Starting optimized training...")
        print(f"📊 Training on {len(train_loader.dataset)} samples")
        print(f"📊 Validating on {len(val_loader.dataset)} samples")
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
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
            
            print(f'⏱️ Epoch {epoch+1}/{epochs} completed in {epoch_time:.2f}s')
            print(f'  📈 Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
            print(f'  📊 Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                
                best_model_path = os.path.join(checkpoint_dir, f'best_model_epoch_{epoch+1}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                }, best_model_path)
                print(f'  💾 New best model saved with val_acc: {val_acc:.4f}')
            
            print('-' * 60)
        
        # Calculate average epoch time
        avg_epoch_time = sum(self.history['epoch_times']) / len(self.history['epoch_times'])
        print(f"⏱️ Average epoch time: {avg_epoch_time:.2f}s")
        
        return {
            'best_val_acc': best_val_acc,
            'best_model_path': best_model_path,
            'history': self.history
        }

def create_optimized_data_loaders(train_df, val_df, test_df, image_dir, batch_size=32, img_size=224):
    """Create optimized data loaders for GPU training"""
    train_transform = get_medical_transforms(img_size, is_training=True)
    val_transform = get_medical_transforms(img_size, is_training=False)
    
    train_dataset = ChestXRayDataset(train_df, image_dir, train_transform)
    val_dataset = ChestXRayDataset(val_df, image_dir, val_transform)
    test_dataset = ChestXRayDataset(test_df, image_dir, val_transform)
    
    # Optimized settings for GPU
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Windows compatibility
        pin_memory=True,  # Faster GPU transfer
        persistent_workers=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False
    )
    
    return train_loader, val_loader, test_loader

def main():
    parser = argparse.ArgumentParser(description='Optimized COVID-19 Training')
    parser.add_argument('--data_csv', type=str, required=True,
                       help='Path to data CSV file')
    parser.add_argument('--image_dir', type=str, required=True,
                       help='Path to image directory')
    parser.add_argument('--output_dir', type=str, default='./outputs/optimized_training',
                       help='Output directory')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training (larger for GPU)')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                       help='Learning rate')
    
    args = parser.parse_args()
    
    # Check GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu_available = check_gpu_usage()
    print(f"🎯 Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize data handler
    print("📂 Loading dataset...")
    data_handler = MedicalDataHandler(args.data_csv, args.image_dir)
    class_distribution = data_handler.analyze_dataset()
    
    # Create splits
    splits = data_handler.create_stratified_splits(n_splits=1)
    
    print(f"📊 Train samples: {len(splits['train'])}")
    print(f"📊 Val samples: {len(splits['val'])}")
    print(f"📊 Test samples: {len(splits['test'])}")
    
    # Create optimized data loaders
    train_loader, val_loader, test_loader = create_optimized_data_loaders(
        splits['train'], splits['val'], splits['test'],
        args.image_dir,
        batch_size=args.batch_size,
        img_size=224
    )
    
    # Create model
    print("🛠️ Creating model...")
    model = create_model(
        'densenet121',
        num_classes=3,
        pretrained=True,
        dropout_rate=0.5
    ).to(device)
    
    # Get class weights
    class_weights = data_handler.get_class_weights()
    print(f"⚖️ Class weights: {class_weights}")
    
    # Create optimized trainer
    trainer = OptimizedCOVIDTrainer(
        model=model,
        device=device,
        class_weights=class_weights,
        learning_rate=args.learning_rate
    )
    
    print(f"\n🚀 Starting optimized training for {args.epochs} epochs...")
    print(f"📦 Batch size: {args.batch_size}")
    print(f"📚 Learning rate: {args.learning_rate}")
    print(f"🎯 Using {'GPU with mixed precision' if gpu_available else 'CPU'}")
    
    result = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        checkpoint_dir=args.output_dir
    )
    
    print(f"\n🎉 Training completed!")
    print(f"🏆 Best validation accuracy: {result['best_val_acc']:.4f}")
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, 'final_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'val_accuracy': result['best_val_acc'],
        'history': result['history'],
        'class_weights': class_weights,
        'training_args': vars(args)
    }, final_model_path)
    
    print(f"💾 Model saved to: {final_model_path}")
    
    # Save training history
    history_df = pd.DataFrame(result['history'])
    history_path = os.path.join(args.output_dir, 'training_history.csv')
    history_df.to_csv(history_path, index=False)
    print(f"📊 Training history saved to: {history_path}")

if __name__ == '__main__':
    main()