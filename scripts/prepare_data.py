#!/usr/bin/env python3
"""
Data preparation script for COVID-19 Radiography Database
"""

import os
import pandas as pd
import shutil
from pathlib import Path
import argparse

def prepare_covid_radiography_data(kaggle_dir, output_dir):
    """
    Prepare COVID-19 Radiography Database for our pipeline
    Fixed version that correctly identifies class labels
    """
    
    # Create output directories
    images_dir = Path(output_dir) / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Source directory: {kaggle_dir}")
    print(f"Output directory: {output_dir}")
    
    # Class mappings based on directory names
    class_mappings = {
        'COVID': 'COVID-19',
        'Normal': 'Normal', 
        'Viral Pneumonia': 'Viral Pneumonia'
        # Note: We're skipping 'Lung_Opacity' as it's not in our 3-class problem
    }
    
    metadata = []
    image_id = 0
    total_copied = 0
    
    print("\nProcessing images...")
    
    # Process each class directory
    for class_dir, class_label in class_mappings.items():
        images_path = Path(kaggle_dir) / class_dir / 'images'
        
        if not images_path.exists():
            print(f"  Warning: {images_path} does not exist. Skipping {class_label}.")
            continue
        
        # Get all image files
        image_files = [f for f in os.listdir(images_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"  Processing {class_label}: {len(image_files)} images")
        
        # Process images in this directory
        for img_file in image_files:
            source_path = images_path / img_file
            
            # Create new filename to avoid conflicts
            new_filename = f"{class_label.replace(' ', '_').lower()}_{image_id:06d}.png"
            dest_path = images_dir / new_filename
            
            try:
                # Copy image
                shutil.copy2(source_path, dest_path)
                
                # Add to metadata
                metadata.append({
                    'filename': new_filename,
                    'label': class_label,
                    'original_path': str(source_path),
                    'image_id': image_id
                })
                
                image_id += 1
                total_copied += 1
                
            except Exception as e:
                print(f"    Error copying {source_path}: {e}")
    
    # Create metadata DataFrame
    if metadata:
        df = pd.DataFrame(metadata)
        
        # Save metadata
        metadata_path = Path(output_dir) / 'metadata.csv'
        df.to_csv(metadata_path, index=False)
        
        print(f"\nData preparation completed!")
        print(f"Total images processed: {total_copied}")
        print(f"Classes distribution:")
        print(df['label'].value_counts())
        print(f"\nMetadata saved to: {metadata_path}")
        print(f"Images copied to: {images_dir}")
        
        return df
    else:
        print("\nERROR: No images were found or processed!")
        print("Please check the directory structure and file paths.")
        return None

def create_sample_dataset(full_dataset_dir, sample_dir, samples_per_class=100):
    """Create a smaller sample dataset for testing"""
    metadata_path = Path(full_dataset_dir) / 'metadata.csv'
    images_dir = Path(full_dataset_dir) / 'images'
    
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    
    df = pd.read_csv(metadata_path)
    
    # Sample from each class
    sampled_df = df.groupby('label', group_keys=False).apply(
        lambda x: x.sample(min(len(x), samples_per_class), random_state=42)
    ).reset_index(drop=True)
    
    # Create sample directory
    sample_images_dir = Path(sample_dir) / 'images'
    sample_images_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy sampled images
    copied_count = 0
    for _, row in sampled_df.iterrows():
        src_path = images_dir / row['filename']
        dst_path = sample_images_dir / row['filename']
        try:
            shutil.copy2(src_path, dst_path)
            copied_count += 1
        except Exception as e:
            print(f"Error copying {src_path}: {e}")
    
    # Save sample metadata
    sample_metadata_path = Path(sample_dir) / 'metadata.csv'
    sampled_df.to_csv(sample_metadata_path, index=False)
    
    print(f"Sample dataset created with {copied_count} images")
    print(f"Sample metadata: {sample_metadata_path}")
    print("Class distribution in sample:")
    print(sampled_df['label'].value_counts())

def main():
    parser = argparse.ArgumentParser(description='Prepare COVID-19 Radiography Dataset')
    parser.add_argument('--kaggle_dir', type=str, required=True,
                       help='Path to downloaded Kaggle dataset directory')
    parser.add_argument('--output_dir', type=str, default='./data/processed',
                       help='Output directory for processed data')
    parser.add_argument('--create_sample', action='store_true',
                       help='Create a small sample dataset for testing')
    parser.add_argument('--samples_per_class', type=int, default=100,
                       help='Number of samples per class for sample dataset')
    
    args = parser.parse_args()
    
    # Check if source directory exists
    if not os.path.exists(args.kaggle_dir):
        print(f"ERROR: Source directory '{args.kaggle_dir}' does not exist!")
        print("Please check the path and try again.")
        return
    
    # Prepare full dataset
    df = prepare_covid_radiography_data(args.kaggle_dir, args.output_dir)
    
    # Create sample dataset if requested and we have data
    if args.create_sample and df is not None:
        sample_dir = Path(args.output_dir).parent / 'sample'
        create_sample_dataset(args.output_dir, sample_dir, args.samples_per_class)

if __name__ == '__main__':
    main()