import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import time
from tqdm import tqdm

class SimpleCOVIDTrainer:
    """Simple trainer class for COVID-19 classification"""
    
    def __init__(self, model, device, class_weights, learning_rate=1e-4):
        self.model = model
        self.device = device
        self.class_weights = class_weights.to(device)
        
        # Ensure learning_rate is float
        if isinstance(learning_rate, str):
            learning_rate = float(learning_rate)
        
        # Setup optimizer and loss
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        
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
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1} Training')
        
        for batch_idx, (data, target, _) in enumerate(pbar):
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
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{correct_predictions/total_samples:.4f}'
            })
        
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = correct_predictions / total_samples
        
        return epoch_loss, epoch_acc
    
    def validate(self, dataloader):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for data, target, _ in tqdm(dataloader, desc='Validation'):
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
            start_time = time.time()
            
            # Training phase
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            
            # Validation phase
            val_loss, val_acc = self.validate(val_loader)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            epoch_time = time.time() - start_time
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
            print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            print(f'  Time: {epoch_time:.2f}s')
            
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