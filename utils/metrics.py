import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (roc_auc_score, precision_recall_curve, 
                           confusion_matrix, classification_report,
                           average_precision_score, roc_curve, auc)
from sklearn.preprocessing import label_binarize
import torch
from typing import Dict, List, Tuple, Optional

class MedicalEvaluator:
    """Comprehensive medical evaluation metrics"""
    
    def __init__(self, class_names: List[str] = None):
        self.class_names = class_names or ['COVID-19', 'Viral Pneumonia', 'Normal']
        self.label_map = {i: name for i, name in enumerate(self.class_names)}
        
    def calculate_comprehensive_metrics(self, y_true: np.ndarray, 
                                      y_pred: np.ndarray, 
                                      y_prob: np.ndarray) -> Dict:
        """Calculate comprehensive medical metrics"""
        from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        recall_per_class = recall_score(y_true, y_pred, average=None)
        precision_per_class = precision_score(y_true, y_pred, average=None)
        f1_per_class = f1_score(y_true, y_pred, average=None)
        
        # Macro and weighted averages
        macro_recall = recall_score(y_true, y_pred, average='macro')
        weighted_recall = recall_score(y_true, y_pred, average='weighted')
        macro_precision = precision_score(y_true, y_pred, average='macro')
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        
        # COVID-19 specific metrics (assuming class 0 is COVID-19)
        covid_recall = recall_per_class[0]
        covid_precision = precision_per_class[0]
        covid_f1 = f1_per_class[0]
        
        # AUC scores
        try:
            if y_prob.shape[1] == 2:  # Binary case
                auc_score = roc_auc_score(y_true, y_prob[:, 1])
                auc_scores = {'macro': auc_score}
            else:  # Multiclass
                y_true_bin = label_binarize(y_true, classes=range(len(self.class_names)))
                auc_scores = {}
                for i in range(len(self.class_names)):
                    auc_scores[self.class_names[i]] = roc_auc_score(y_true_bin[:, i], y_prob[:, i])
                auc_scores['macro'] = roc_auc_score(y_true_bin, y_prob, average='macro')
                auc_scores['weighted'] = roc_auc_score(y_true_bin, y_prob, average='weighted')
        except Exception as e:
            print(f"Error calculating AUC: {e}")
            auc_scores = {}
        
        metrics = {
            'accuracy': accuracy,
            'per_class_recall': recall_per_class,
            'per_class_precision': precision_per_class,
            'per_class_f1': f1_per_class,
            'macro_recall': macro_recall,
            'weighted_recall': weighted_recall,
            'macro_precision': macro_precision,
            'macro_f1': macro_f1,
            'covid_recall': covid_recall,
            'covid_precision': covid_precision,
            'covid_f1': covid_f1,
            'auc_scores': auc_scores
        }
        
        return metrics
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            normalize: bool = True, save_path: str = None):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = 'Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'
            
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                   xticklabels=self.class_names, 
                   yticklabels=self.class_names)
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return cm
    
    def plot_roc_curves(self, y_true: np.ndarray, y_prob: np.ndarray, 
                       save_path: str = None):
        """Plot ROC curves for all classes"""
        if y_prob.shape[1] == 2:
            # Binary case
            fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.3f})')
        else:
            # Multiclass case
            y_true_bin = label_binarize(y_true, classes=range(len(self.class_names)))
            
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            
            plt.figure(figsize=(10, 8))
            colors = ['blue', 'red', 'green', 'purple', 'orange']
            
            for i in range(len(self.class_names)):
                fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
                plt.plot(fpr[i], tpr[i], color=colors[i % len(colors)], lw=2,
                        label=f'{self.class_names[i]} (AUC = {roc_auc[i]:.3f})')
            
            # Micro-average ROC
            fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            plt.plot(fpr["micro"], tpr["micro"], 
                    label=f'Micro-average (AUC = {roc_auc["micro"]:.3f})',
                    color='deeppink', linestyle=':', linewidth=4)
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curves')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return roc_auc
    
    def plot_precision_recall_curves(self, y_true: np.ndarray, y_prob: np.ndarray,
                                   save_path: str = None):
        """Plot precision-recall curves"""
        if y_prob.shape[1] == 2:
            precision, recall, _ = precision_recall_curve(y_true, y_prob[:, 1])
            avg_precision = average_precision_score(y_true, y_prob[:, 1])
            
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, lw=2,
                    label=f'AP = {avg_precision:.3f}')
        else:
            y_true_bin = label_binarize(y_true, classes=range(len(self.class_names)))
            
            precision = dict()
            recall = dict()
            average_precision = dict()
            
            plt.figure(figsize=(10, 8))
            colors = ['blue', 'red', 'green', 'purple', 'orange']
            
            for i in range(len(self.class_names)):
                precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
                average_precision[i] = average_precision_score(y_true_bin[:, i], y_prob[:, i])
                plt.plot(recall[i], precision[i], color=colors[i % len(colors)], lw=2,
                        label=f'{self.class_names[i]} (AP = {average_precision[i]:.3f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall Curves')
        plt.legend(loc="best")
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return average_precision
    
    def find_optimal_threshold(self, y_true: np.ndarray, y_prob: np.ndarray,
                             target_class: int = 0, min_precision: float = 0.7):
        """Find optimal threshold for high recall while maintaining precision"""
        from sklearn.metrics import precision_recall_curve
        
        y_true_binary = (y_true == target_class).astype(int)
        precision, recall, thresholds = precision_recall_curve(
            y_true_binary, y_prob[:, target_class]
        )
        
        # Find threshold that gives maximum recall with minimum precision constraint
        valid_indices = [i for i, p in enumerate(precision) 
                        if p >= min_precision and i < len(thresholds)]
        
        if valid_indices:
            best_idx = valid_indices[np.argmax(recall[valid_indices])]
            best_threshold = thresholds[best_idx]
            best_recall = recall[best_idx]
            best_precision = precision[best_idx]
        else:
            # If no threshold meets precision constraint, use default
            best_threshold = 0.5
            best_recall = recall[np.argmin(np.abs(thresholds - 0.5))]
            best_precision = precision[np.argmin(np.abs(thresholds - 0.5))]
        
        return best_threshold, best_recall, best_precision
    
    def generate_report(self, y_true: np.ndarray, y_pred: np.ndarray, 
                       y_prob: np.ndarray, save_path: str = None) -> Dict:
        """Generate comprehensive evaluation report"""
        # Calculate metrics
        metrics = self.calculate_comprehensive_metrics(y_true, y_pred, y_prob)
        
        # Generate classification report
        class_report = classification_report(y_true, y_pred, 
                                           target_names=self.class_names,
                                           output_dict=True)
        
        # Create report
        report = {
            'metrics': metrics,
            'classification_report': class_report,
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
        
        # Print summary
        print("=== COMPREHENSIVE MEDICAL EVALUATION REPORT ===")
        print(f"\nOverall Accuracy: {metrics['accuracy']:.4f}")
        print(f"Macro-average F1: {metrics['macro_f1']:.4f}")
        
        print(f"\nCOVID-19 Detection Performance:")
        print(f"  Recall (Sensitivity): {metrics['covid_recall']:.4f}")
        print(f"  Precision: {metrics['covid_precision']:.4f}")
        print(f"  F1-Score: {metrics['covid_f1']:.4f}")
        
        if 'auc_scores' in metrics and 'macro' in metrics['auc_scores']:
            print(f"  Macro AUC: {metrics['auc_scores']['macro']:.4f}")
        
        # Save report if path provided
        if save_path:
            import json
            with open(save_path, 'w') as f:
                # Convert numpy types to Python types for JSON serialization
                report_serializable = json.loads(json.dumps(report, default=str))
                json.dump(report_serializable, f, indent=2)
        
        return report