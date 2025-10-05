import os
import pandas as pd  # Add this import
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split, StratifiedKFold
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms

class ChestXRayDataset(Dataset):
    """COVID-19 Chest X-Ray Dataset"""
    
    def __init__(self, dataframe, image_dir, transform=None, 
                 class_map=None, is_training=True):
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.is_training = is_training
        
        # Default class mapping
        self.class_map = class_map or {
            'COVID-19': 0, 
            'Viral Pneumonia': 1, 
            'Normal': 2
        }
        self.inverse_class_map = {v: k for k, v in self.class_map.items()}
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        img_info = self.dataframe.iloc[idx]
        img_path = os.path.join(self.image_dir, img_info['filename'])
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image as fallback
            image = Image.new('RGB', (224, 224), color='black')
        
        # Get label
        label = self.class_map[img_info['label']]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label, img_path  # Return path for debugging
    
    def get_class_distribution(self):
        return self.dataframe['label'].value_counts()

class MedicalDataHandler:
    """Handler for medical data with proper splitting and balancing"""
    
    def __init__(self, csv_path, image_dir, random_state=42):
        self.image_dir = image_dir
        self.random_state = random_state
        self.class_map = {'COVID-19': 0, 'Viral Pneumonia': 1, 'Normal': 2}
        
        if csv_path is not None:
            self.df = pd.read_csv(csv_path)
        else:
            self.df = None
        
    def set_dataframe(self, dataframe):
        """Set dataframe manually"""
        self.df = dataframe
        
    def analyze_dataset(self):
        """Analyze dataset characteristics"""
        if self.df is None:
            print("No dataset loaded")
            return None
            
        print("Dataset Analysis:")
        print(f"Total samples: {len(self.df)}")
        
        class_counts = self.df['label'].value_counts()
        for label, count in class_counts.items():
            percentage = count / len(self.df) * 100
            print(f"{label}: {count} ({percentage:.2f}%)")
        
        return class_counts
    
    def create_stratified_splits(self, n_splits=5, test_size=0.15, val_size=0.15):
        """Create stratified train/val/test splits"""
        if self.df is None:
            raise ValueError("No dataset loaded. Call set_dataframe() first.")
            
        # First split: separate test set
        train_val_df, test_df = train_test_split(
            self.df,
            test_size=test_size,
            stratify=self.df['label'],
            random_state=self.random_state
        )
        
        # Second split: separate validation set from train+val
        val_ratio = val_size / (1 - test_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_ratio,
            stratify=train_val_df['label'],
            random_state=self.random_state
        )
        
        # Only create k-fold splits if n_splits > 1
        cv_splits = []
        if n_splits > 1:
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, 
                                 random_state=self.random_state)
            
            for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['label'])):
                cv_splits.append({
                    'fold': fold,
                    'train': train_df.iloc[train_idx],
                    'val': train_df.iloc[val_idx]
                })
        else:
            # For single split, just use the original train/val split
            cv_splits.append({
                'fold': 0,
                'train': train_df,
                'val': val_df
            })
        
        splits = {
            'train': train_df,
            'val': val_df,
            'test': test_df,
            'cv_splits': cv_splits
        }
        
        return splits
    
    def get_class_weights(self):
        """Calculate class weights for imbalanced data"""
        if self.df is None:
            return torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
            
        class_counts = self.df['label'].value_counts().sort_index()
        total_samples = len(self.df)
        num_classes = len(class_counts)
        
        # Using inverse frequency weighting
        weights = total_samples / (num_classes * class_counts)
        weights_tensor = torch.tensor(weights.values, dtype=torch.float32)
        
        return weights_tensor
    
    def get_sampler(self, dataset):
        """Create weighted sampler for imbalanced training"""
        if self.df is None:
            return None
            
        labels = [dataset.class_map[label] for label in dataset.dataframe['label']]
        class_weights = self.get_class_weights()
        
        sample_weights = [class_weights[label] for label in labels]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        
        return sampler

def get_medical_transforms(img_size=224, is_training=True):
    """Get medical-appropriate data transformations"""
    if is_training:
        # Medical images: no horizontal flips due to anatomical consistency
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), 
                                  scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    return transform

def create_data_loaders(train_df, val_df, test_df, image_dir, batch_size=16, img_size=224, use_sampler=True):
    """Create data loaders for train, validation, and test sets"""
    train_transform = get_medical_transforms(img_size, is_training=True)
    val_transform = get_medical_transforms(img_size, is_training=False)
    
    train_dataset = ChestXRayDataset(train_df, image_dir, train_transform)
    val_dataset = ChestXRayDataset(val_df, image_dir, val_transform)
    test_dataset = ChestXRayDataset(test_df, image_dir, val_transform)
    
    # Create data handlers for sampling (only if we have a CSV for class weights)
    train_sampler = None
    if use_sampler:
        try:
            # Create a temporary data handler just for getting class weights
            temp_handler = MedicalDataHandler(None, image_dir)
            # Manually set the dataframe for weight calculation
            temp_handler.df = train_df
            train_sampler = temp_handler.get_sampler(train_dataset)
        except Exception as e:
            print(f"Warning: Could not create weighted sampler: {e}. Using default sampling.")
    
    # Use num_workers=0 for Windows to avoid multiprocessing issues
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=0,  # Changed from 4 to 0
        pin_memory=False  # Changed from True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Changed from 4 to 0
        pin_memory=False  # Changed from True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Changed from 4 to 0
        pin_memory=False  # Changed from True
    )
    
    return train_loader, val_loader, test_loader