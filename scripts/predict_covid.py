#!/usr/bin/env python3
"""
Simple COVID-19 prediction interface
"""

import os
import sys
import torch
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.architectures import create_model

class COVIDPredictor:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = ['COVID-19', 'Viral Pneumonia', 'Normal']
        
        # Load model
        checkpoint = torch.load(model_path, map_location='cpu')
        self.model = create_model('densenet121', num_classes=3, pretrained=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f"✅ Model loaded with {checkpoint.get('val_accuracy', 0):.2%} validation accuracy")
    
    def is_valid_image_file(self, file_path):
        """Check if file is a valid image"""
        valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
        file_ext = Path(file_path).suffix.lower()
        return file_ext in valid_extensions
    
    def predict(self, image_path):
        """Predict a single image"""
        try:
            # Validate file exists and is an image
            if not os.path.exists(image_path):
                print("❌ File not found. Please check the path.")
                return None, 0
            
            if not self.is_valid_image_file(image_path):
                print("❌ Not a valid image file. Supported formats: PNG, JPG, JPEG, BMP, TIFF")
                return None, 0
            
            # Open and process image
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(input_tensor)
                probabilities = torch.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            
            prediction = self.class_names[predicted.item()]
            confidence = confidence.item()
            
            print(f"\n🩺 COVID-19 Detection Result:")
            print(f"📁 Image: {os.path.basename(image_path)}")
            print(f"🔍 Prediction: {prediction}")
            print(f"🎯 Confidence: {confidence:.2%}")
            
            print("\n📊 Class Probabilities:")
            for i, (class_name, prob) in enumerate(zip(self.class_names, probabilities[0])):
                print(f"  {class_name}: {prob.item():.2%}")
            
            # Medical advice based on prediction
            if prediction == 'COVID-19' and confidence > 0.7:
                print("\n⚠️  MEDICAL ALERT: High confidence COVID-19 detected!")
                print("   Please consult healthcare professional immediately.")
            elif prediction == 'Viral Pneumonia' and confidence > 0.7:
                print("\n⚠️  MEDICAL ALERT: Viral Pneumonia detected!")
                print("   Please consult healthcare professional.")
            elif confidence < 0.6:
                print("\n⚠️  UNCERTAIN: Low confidence prediction.")
                print("   Consider clinical evaluation or retest.")
            else:
                print("\n✅ Prediction appears normal or low risk.")
                print("   Continue monitoring as recommended by healthcare provider.")
            
            return prediction, confidence
            
        except Exception as e:
            print(f"❌ Error processing image: {e}")
            print("   Please ensure this is a valid chest X-ray image.")
            return None, 0

def list_sample_images():
    """List available sample images for testing"""
    sample_dir = Path("data/sample/images")
    if sample_dir.exists():
        print("\n📂 Available sample images for testing:")
        image_files = list(sample_dir.glob("*.png"))
        for i, img_path in enumerate(image_files[:10]):  # Show first 10
            print(f"   {i+1}. {img_path.name}")
        if len(image_files) > 10:
            print(f"   ... and {len(image_files) - 10} more")
        return sample_dir
    return None

def main():
    model_path = "outputs/memory_training/final_model.pth"
    
    if not os.path.exists(model_path):
        print("❌ Model not found. Please train the model first.")
        return
    
    predictor = COVIDPredictor(model_path)
    
    print("🦠 COVID-19 Chest X-Ray Detection System")
    print("=" * 50)
    print("This AI tool analyzes chest X-rays for COVID-19, Viral Pneumonia, and Normal cases.")
    print("NOTE: This is for research/demonstration purposes only.")
    print("Always consult healthcare professionals for medical diagnosis.\n")
    
    # Show sample images
    sample_dir = list_sample_images()
    if sample_dir:
        print(f"\n💡 Tip: You can use sample images like: {sample_dir / 'covid-19_000000.png'}")
    
    print("\nEnter the path to a chest X-ray image (or 'quit' to exit):")
    
    while True:
        image_path = input("\n📁 Image path: ").strip().strip('"')
        
        if image_path.lower() in ['quit', 'exit', 'q']:
            print("👋 Thank you for using COVID-19 Detection System!")
            break
        
        # Handle relative paths
        if not os.path.isabs(image_path):
            image_path = os.path.abspath(image_path)
        
        predictor.predict(image_path)

if __name__ == '__main__':
    main()