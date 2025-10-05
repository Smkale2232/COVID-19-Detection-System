import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional
from .architectures import create_model

class COVIDEnsemble:
    """Ensemble model for COVID-19 detection"""
    
    def __init__(self, model_configs: List[Dict], device: torch.device):
        self.models = []
        self.device = device
        self.model_configs = model_configs
        
        for config in model_configs:
            model = create_model(
                config['backbone'],
                num_classes=config.get('num_classes', 3),
                pretrained=config.get('pretrained', False),
                dropout_rate=config.get('dropout_rate', 0.5)
            )
            
            if 'checkpoint_path' in config:
                checkpoint = torch.load(config['checkpoint_path'], map_location='cpu')
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
            
            model.to(device)
            model.eval()
            self.models.append(model)
    
    def predict(self, x: torch.Tensor, method: str = 'soft_voting', 
                n_mc_samples: int = 1) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Make ensemble predictions
        
        Args:
            x: Input tensor
            method: 'soft_voting' or 'hard_voting'
            n_mc_samples: Number of Monte Carlo samples for uncertainty estimation
        
        Returns:
            predictions: Ensemble predictions
            probabilities: Class probabilities
            uncertainties: Prediction uncertainties (if n_mc_samples > 1)
        """
        all_predictions = []
        all_probabilities = []
        
        with torch.no_grad():
            for model in self.models:
                model_predictions = []
                model_probabilities = []
                
                for _ in range(n_mc_samples):
                    if n_mc_samples > 1:
                        model.enable_dropout()  # MC Dropout
                    
                    output = model(x)
                    probabilities = torch.softmax(output, dim=1)
                    _, predicted = torch.max(output, 1)
                    
                    model_predictions.append(predicted.cpu().numpy())
                    model_probabilities.append(probabilities.cpu().numpy())
                
                # Average over MC samples for this model
                model_avg_probs = np.mean(model_probabilities, axis=0)
                all_probabilities.append(model_avg_probs)
                
                if n_mc_samples == 1:
                    all_predictions.append(model_predictions[0])
                
                model.eval()
        
        all_probabilities = np.array(all_probabilities)  # [n_models, batch_size, n_classes]
        
        if method == 'soft_voting':
            # Average probabilities across models
            ensemble_probs = np.mean(all_probabilities, axis=0)
            ensemble_preds = np.argmax(ensemble_probs, axis=1)
            
        elif method == 'hard_voting':
            # Majority vote
            all_predictions = np.array(all_predictions)
            ensemble_preds = []
            for i in range(all_predictions.shape[1]):
                votes = all_predictions[:, i]
                ensemble_preds.append(np.bincount(votes).argmax())
            ensemble_preds = np.array(ensemble_preds)
            ensemble_probs = None
        
        # Calculate uncertainty if using MC Dropout
        uncertainties = None
        if n_mc_samples > 1:
            uncertainties = -np.sum(ensemble_probs * np.log(ensemble_probs + 1e-8), axis=1)
        
        return ensemble_preds, ensemble_probs, uncertainties
    
    def predict_with_confidence(self, x: torch.Tensor, confidence_threshold: float = 0.75,
                               n_mc_samples: int = 10) -> Dict:
        """
        Predict with confidence scores and uncertainty estimation
        """
        preds, probs, uncertainties = self.predict(x, method='soft_voting', 
                                                  n_mc_samples=n_mc_samples)
        
        confidence_scores = np.max(probs, axis=1)
        low_confidence_mask = confidence_scores < confidence_threshold
        
        return {
            'predictions': preds,
            'probabilities': probs,
            'confidence_scores': confidence_scores,
            'uncertainties': uncertainties,
            'low_confidence_mask': low_confidence_mask,
            'needs_review': low_confidence_mask | (uncertainties > np.percentile(uncertainties, 75))
        }