import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import time
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add this import
from utils.data_loader import create_data_loaders

class COVIDTrainer:
    """Trainer class for COVID-19 classification models"""
    
    def __init__(self, model, device, class_weights, optimizer_config, scheduler_config=None):
        self.model = model
        self.device = device
        self.class_weights = class_weights.to(device)
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer(optimizer_config)
        self.scheduler = self._setup_scheduler(scheduler_config) if scheduler_config else None
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        
        # Training history
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'learning_rates': []
        }
    
    def _setup_optimizer(self, config):
        """Setup optimizer based on configuration"""
        optimizer_name = config.get('name', 'AdamW')
        lr = config.get('lr', 1e-4)
        weight_decay = config.get('weight_decay', 1e-4)
        
        if optimizer_name == 'AdamW':
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'Adam':
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'SGD':
            momentum = config.get('momentum', 0.9)
            return optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, 
                           weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    def _setup_scheduler(self, config):
        """Setup learning rate scheduler"""
        scheduler_name = config.get('name', 'CosineAnnealing')
        
        if scheduler_name == 'CosineAnnealing':
            T_max = config.get('T_max', 10)
            return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=T_max)
        elif scheduler_name == 'ReduceLROnPlateau':
            patience = config.get('patience', 5)
            factor = config.get('factor', 0.5)
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', patience=patience, factor=factor
            )
        elif scheduler_name == 'StepLR':
            step_size = config.get('step_size', 10)
            gamma = config.get('gamma', 0.1)
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_name}")
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Tuple[float, float]:
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
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
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
    
    def validate(self, dataloader: DataLoader) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for data, target, _ in tqdm(dataloader, desc='Validation'):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                running_loss += loss.item()
                _, predicted = torch.max(output, 1)
                correct_predictions += (predicted == target).sum().item()
                total_samples += target.size(0)
                
                probabilities = torch.softmax(output, dim=1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = correct_predictions / total_samples
        
        return epoch_loss, epoch_acc, np.array(all_predictions), np.array(all_targets), np.array(all_probabilities)
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              epochs: int, early_stopping_patience: int = 10,
              checkpoint_dir: str = './checkpoints') -> Dict:
        """Complete training loop"""
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        best_val_acc = 0.0
        patience_counter = 0
        best_model_path = None
        
        print("Starting training...")
        print(f"Training on {len(train_loader.dataset)} samples")
        print(f"Validating on {len(val_loader.dataset)} samples")
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Training phase
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            
            # Validation phase
            val_loss, val_acc, _, _, _ = self.validate(val_loader)
            
            # Update scheduler
            current_lr = self.optimizer.param_groups[0]['lr']
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(current_lr)
            
            epoch_time = time.time() - start_time
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
            print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            print(f'  LR: {current_lr:.2e}, Time: {epoch_time:.2f}s')
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                
                # Save model checkpoint
                best_model_path = os.path.join(checkpoint_dir, f'best_model_epoch_{epoch+1}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                    'history': self.history
                }, best_model_path)
                print(f'  → New best model saved with val_acc: {val_acc:.4f}')
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break
            
            print('-' * 60)
        
        # Load best model for return
        if best_model_path and os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best model from epoch {checkpoint['epoch'] + 1}")
        
        return {
            'best_val_acc': best_val_acc,
            'best_model_path': best_model_path,
            'history': self.history
        }


class CrossValidationTrainer:
    """Trainer for cross-validation experiments"""
    
    def __init__(self, data_handler, model_factory, device, config):
        self.data_handler = data_handler
        self.model_factory = model_factory
        self.device = device
        self.config = config
        
    def run_cross_validation(self, n_splits=5):
        """Run stratified k-fold cross-validation"""
        splits = self.data_handler.create_stratified_splits(n_splits=n_splits)
        cv_results = {}
        
        for fold, split in enumerate(splits['cv_splits']):
            print(f"\n{'='*50}")
            print(f"Training Fold {fold + 1}/{n_splits}")
            print(f"{'='*50}")
            
            # Create data loaders for this fold
            train_loader, val_loader, _ = create_data_loaders(
                split['train'], split['val'], splits['test'],
                self.data_handler.image_dir,
                batch_size=self.config['data']['batch_size'],
                img_size=self.config['data']['image_size'][0],
                use_sampler=True
            )
            
            # Create model
            model = self.model_factory(
                self.config['model']['backbone'],
                num_classes=self.config['model']['num_classes'],
                pretrained=self.config['model']['pretrained'],
                dropout_rate=self.config['model']['dropout_rate']
            ).to(self.device)
            
            # Get class weights
            class_weights = self.data_handler.get_class_weights()
            
            # Create trainer
            trainer = COVIDTrainer(
                model=model,
                device=self.device,
                class_weights=class_weights,
                optimizer_config=self.config['training'],
                scheduler_config=self.config.get('scheduler')
            )
            
            # Train model with fewer epochs for testing
            epochs = min(self.config['training']['epochs'], 5)  # Reduced for testing
            
            fold_result = trainer.train(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=epochs,
                early_stopping_patience=3,
                checkpoint_dir=os.path.join(self.config['paths']['output_dir'], f'fold_{fold}')
            )
            
            cv_results[f'fold_{fold}'] = {
                'model': model,
                'trainer': trainer,
                'result': fold_result,
                'val_accuracy': fold_result['best_val_acc']
            }
            
            print(f"Fold {fold + 1} completed. Best Val Acc: {fold_result['best_val_acc']:.4f}")
        
        # Aggregate results
        val_accuracies = [result['val_accuracy'] for result in cv_results.values()]
        print(f"\nCross-Validation Results:")
        print(f"Mean Val Accuracy: {np.mean(val_accuracies):.4f} ± {np.std(val_accuracies):.4f}")
        print(f"Fold Accuracies: {[f'{acc:.4f}' for acc in val_accuracies]}")
        
        return cv_results