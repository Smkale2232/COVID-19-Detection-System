#!/usr/bin/env python3
"""
Make predictions on single images
"""

import torch
from PIL import Image
import torchvision.transforms as transforms
from models.architectures import create_model

def predict_image(model_path, image_path):
    """Predict a single image"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    checkpoint = torch.load(model_path, map_location='cpu')
    model = create_model('densenet121', num_classes=3, pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Preprocess image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    class_names = ['COVID-19', 'Viral Pneumonia', 'Normal']
    
    print(f"Prediction: {class_names[predicted.item()]}")
    print(f"Confidence: {confidence.item():.4f}")
    print("\nClass Probabilities:")
    for i, prob in enumerate(probabilities[0]):
        print(f"  {class_names[i]}: {prob.item():.4f}")
    
    return class_names[predicted.item()], confidence.item()

if __name__ == '__main__':
    model_path = "outputs/windows_training/final_model.pth"
    # Test with a sample image
    image_path = "data/sample/images/covid-19_000000.png"  # Change this path
    predict_image(model_path, image_path)