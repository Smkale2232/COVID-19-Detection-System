import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional, Dict, Any

class BaseCOVIDClassifier(nn.Module):
    """Base class for COVID-19 classification models"""
    
    def __init__(self, num_classes: int = 3, dropout_rate: float = 0.5):
        super(BaseCOVIDClassifier, self).__init__()
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
    def _modify_classifier(self, in_features: int):
        """Replace the classifier head for transfer learning"""
        return nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate / 2),
            nn.Linear(512, self.num_classes)
        )
    
    def enable_dropout(self):
        """Enable dropout layers for MC Dropout"""
        for m in self.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

class DenseNetCOVID(BaseCOVIDClassifier):
    """DenseNet-based COVID classifier"""
    
    def __init__(self, variant: str = "densenet121", num_classes: int = 3, 
                 pretrained: bool = True, dropout_rate: float = 0.5):
        super().__init__(num_classes, dropout_rate)
        
        if variant == "densenet121":
            self.backbone = models.densenet121(pretrained=pretrained)
            in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        elif variant == "densenet169":
            self.backbone = models.densenet169(pretrained=pretrained)
            in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported DenseNet variant: {variant}")
            
        self.classifier = self._modify_classifier(in_features)
        
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

class ResNetCOVID(BaseCOVIDClassifier):
    """ResNet-based COVID classifier"""
    
    def __init__(self, variant: str = "resnet50", num_classes: int = 3,
                 pretrained: bool = True, dropout_rate: float = 0.5):
        super().__init__(num_classes, dropout_rate)
        
        if variant == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif variant == "resnet101":
            self.backbone = models.resnet101(pretrained=pretrained)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif variant == "resnet34":
            self.backbone = models.resnet34(pretrained=pretrained)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            raise ValueError(f"Unsupported ResNet variant: {variant}")
            
        self.classifier = self._modify_classifier(in_features)
        
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

class EfficientNetCOVID(BaseCOVIDClassifier):
    """EfficientNet-based COVID classifier using torchvision"""
    
    def __init__(self, variant: str = "efficientnet_b0", num_classes: int = 3,
                 pretrained: bool = True, dropout_rate: float = 0.5):
        super().__init__(num_classes, dropout_rate)
        
        # Use torchvision's EfficientNet instead of timm
        if variant == "efficientnet_b0":
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            in_features = self.backbone.classifier[1].in_features
            # Replace classifier with identity
            self.backbone.classifier = nn.Identity()
        elif variant == "efficientnet_b1":
            self.backbone = models.efficientnet_b1(pretrained=pretrained)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported EfficientNet variant: {variant}")
            
        self.classifier = self._modify_classifier(in_features)
        
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

def create_model(model_name: str, **kwargs) -> nn.Module:
    """Factory function to create models"""
    model_registry = {
        'densenet121': DenseNetCOVID,
        'densenet169': DenseNetCOVID,
        'resnet50': ResNetCOVID,
        'resnet101': ResNetCOVID,
        'resnet34': ResNetCOVID,
        'efficientnet_b0': EfficientNetCOVID,
        'efficientnet_b1': EfficientNetCOVID,
    }
    
    if model_name not in model_registry:
        raise ValueError(f"Model {model_name} not supported. Available: {list(model_registry.keys())}")
    
    return model_registry[model_name](variant=model_name, **kwargs)