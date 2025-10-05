#!/usr/bin/env python3
"""
Prediction script for COVID-19 Chest X-Ray classification
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.architectures import create_model
from models.ensemble import COVIDEnsemble

class COVIDPredictor:
    """COVID-19 prediction interface"""
    
    def __init__(self, model_configs, device, img_size=224):
        self.device = device
        self.img_size = img_size
        self.class_names = ['COVID-19', 'Viral Pneumonia', 'Normal']
        
        if isinstance(model_configs, list):
            # Ensemble mode
            self.ensemble = COVIDEnsemble(model_configs, device)
            self.is_ensemble = True
        else:
            # Single model mode
            self.model = self._load_single_model(model_configs)
            self.is_ensemble = False
        
        # Preprocessing transform
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _load_single_model(self, config):
        """Load single model"""
        model = create_model(
            config['backbone'],
            num_classes=config.get('num_classes', 3),
            pretrained=False,
            dropout_rate=config.get('dropout_rate', 0.5)
        ).to(self.device)
        
        checkpoint = torch.load(config['checkpoint_path'], map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        return model
    
    def predict_image(self, image_path, use_uncertainty=True):
        """Predict single image"""
        # Load and preprocess image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            return {
                'error': f"Could not load image: {e}",
                'image_path': image_path
            }
        
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        if self.is_ensemble:
            if use_uncertainty:
                # Use MC Dropout for uncertainty estimation
                result = self.ensemble.predict_with_confidence(
                    input_tensor, 
                    n_mc_samples=10
                )
                
                prediction = {
                    'predicted_class': self.class_names[result['predictions'][0]],
                    'probabilities': {
                        self.class_names[i]: float(prob) 
                        for i, prob in enumerate(result['probabilities'][0])
                    },
                    'confidence': float(result['confidence_scores'][0]),
                    'uncertainty': float(result['uncertainties'][0]),
                    'needs_review': bool(result['needs_review'][0]),
                    'image_path': image_path
                }
            else:
                preds, probs, _ = self.ensemble.predict(input_tensor)
                prediction = {
                    'predicted_class': self.class_names[preds[0]],
                    'probabilities': {
                        self.class_names[i]: float(prob) 
                        for i, prob in enumerate(probs[0])
                    },
                    'confidence': float(np.max(probs[0])),
                    'image_path': image_path
                }
        else:
            # Single model prediction
            with torch.no_grad():
                output = self.model(input_tensor)
                probabilities = torch.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                prediction = {
                    'predicted_class': self.class_names[predicted.item()],
                    'probabilities': {
                        self.class_names[i]: float(prob) 
                        for i, prob in enumerate(probabilities[0])
                    },
                    'confidence': float(confidence.item()),
                    'image_path': image_path
                }
        
        return prediction
    
    def predict_batch(self, image_dir, output_csv=None):
        """Predict all images in a directory"""
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_paths = []
        
        for ext in image_extensions:
            image_paths.extend(Path(image_dir).glob(f'*{ext}'))
            image_paths.extend(Path(image_dir).glob(f'*{ext.upper()}'))
        
        print(f"Found {len(image_paths)} images in {image_dir}")
        
        results = []
        for image_path in image_paths:
            print(f"Processing: {image_path.name}")
            result = self.predict_image(str(image_path))
            results.append(result)
        
        # Create results dataframe
        df_data = []
        for result in results:
            if 'error' in result:
                print(f"Error processing {result['image_path']}: {result['error']}")
                continue
            
            row = {
                'image_path': result['image_path'],
                'predicted_class': result['predicted_class'],
                'confidence': result['confidence'],
            }
            
            # Add probabilities for each class
            for class_name, prob in result['probabilities'].items():
                row[f'prob_{class_name.lower().replace(" ", "_")}'] = prob
            
            # Add uncertainty if available
            if 'uncertainty' in result:
                row['uncertainty'] = result['uncertainty']
                row['needs_review'] = result['needs_review']
            
            df_data.append(row)
        
        results_df = pd.DataFrame(df_data)
        
        if output_csv:
            results_df.to_csv(output_csv, index=False)
            print(f"Predictions saved to: {output_csv}")
        
        return results_df

def main():
    parser = argparse.ArgumentParser(description='Predict COVID-19 from Chest X-Ray images')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--input', type=str, required=True,
                       help='Input image or directory')
    parser.add_argument('--model_path', type=str,
                       help='Path to model checkpoint (single model)')
    parser.add_argument('--ensemble_dir', type=str,
                       help='Directory containing ensemble models')
    parser.add_argument('--output', type=str,
                       help='Output CSV file for batch predictions')
    parser.add_argument('--use_uncertainty', action='store_true',
                       help='Use uncertainty estimation')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize predictor
    if args.ensemble_dir:
        # Load ensemble models
        model_configs = []
        for model_file in os.listdir(args.ensemble_dir):
            if model_file.endswith('.pth'):
                model_configs.append({
                    'backbone': config['model']['backbone'],
                    'num_classes': config['model']['num_classes'],
                    'checkpoint_path': os.path.join(args.ensemble_dir, model_file)
                })
        predictor = COVIDPredictor(model_configs, device)
        print(f"Loaded ensemble with {len(model_configs)} models")
        
    elif args.model_path:
        # Load single model
        model_config = {
            'backbone': config['model']['backbone'],
            'num_classes': config['model']['num_classes'],
            'checkpoint_path': args.model_path
        }
        predictor = COVIDPredictor(model_config, device)
        print("Loaded single model")
    
    else:
        raise ValueError("Either --model_path or --ensemble_dir must be provided")
    
    # Make predictions
    if os.path.isfile(args.input):
        # Single image prediction
        result = predictor.predict_image(args.input, args.use_uncertainty)
        
        print("\n=== PREDICTION RESULT ===")
        print(f"Image: {result['image_path']}")
        print(f"Predicted Class: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.4f}")
        
        print("\nClass Probabilities:")
        for class_name, prob in result['probabilities'].items():
            print(f"  {class_name}: {prob:.4f}")
        
        if 'uncertainty' in result:
            print(f"Uncertainty: {result['uncertainty']:.4f}")
            print(f"Needs Review: {'Yes' if result['needs_review'] else 'No'}")
    
    elif os.path.isdir(args.input):
        # Batch prediction
        results_df = predictor.predict_batch(args.input, args.output)
        
        # Print summary
        print("\n=== BATCH PREDICTION SUMMARY ===")
        print(f"Total images processed: {len(results_df)}")
        
        class_counts = results_df['predicted_class'].value_counts()
        for class_name, count in class_counts.items():
            percentage = count / len(results_df) * 100
            print(f"{class_name}: {count} ({percentage:.1f}%)")
        
        if 'needs_review' in results_df.columns:
            needs_review = results_df['needs_review'].sum()
            print(f"Images needing review: {needs_review} ({needs_review/len(results_df)*100:.1f}%)")
    
    else:
        raise ValueError(f"Input path does not exist: {args.input}")

if __name__ == '__main__':
    main()