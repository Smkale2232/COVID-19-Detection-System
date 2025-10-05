import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List
from .trainer import CrossValidationTrainer

def save_cross_validation_results(cv_results: Dict, output_dir: str):
    """Save cross-validation results to files"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save summary
    summary = {
        'fold_accuracies': [result['val_accuracy'] for result in cv_results.values()],
        'mean_accuracy': np.mean([result['val_accuracy'] for result in cv_results.values()]),
        'std_accuracy': np.std([result['val_accuracy'] for result in cv_results.values()]),
        'num_folds': len(cv_results)
    }
    
    with open(os.path.join(output_dir, 'cv_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save detailed results per fold
    for fold_name, result in cv_results.items():
        fold_dir = os.path.join(output_dir, fold_name)
        os.makedirs(fold_dir, exist_ok=True)
        
        # Save training history
        history_df = pd.DataFrame(result['trainer'].history)
        history_df.to_csv(os.path.join(fold_dir, 'training_history.csv'), index=False)
        
        # Save model checkpoint
        torch.save({
            'model_state_dict': result['model'].state_dict(),
            'fold': fold_name,
            'val_accuracy': result['val_accuracy']
        }, os.path.join(fold_dir, 'model.pth'))
    
    print(f"Cross-validation results saved to {output_dir}")
    return summary