#!/usr/bin/env python3
"""
Fixed COVID-19 prediction interface with better path handling
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
    
    def resolve_path(self, input_path):
        """Resolve and validate file path"""
        path = Path(input_path)
        
        # If relative path, make it relative to project root
        if not path.is_absolute():
            path = Path(__file__).parent.parent / path
        
        # Try different extensions if needed
        if not path.exists():
            # Try with different extensions
            for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
                alt_path = path.with_suffix(ext)
                if alt_path.exists():
                    return alt_path
        
        return path
    
    def is_valid_image_file(self, file_path):
        """Check if file is a valid image"""
        valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
        file_ext = Path(file_path).suffix.lower()
        return file_ext in valid_extensions
    
    def predict(self, image_path):
        """Predict a single image"""
        try:
            # Resolve and validate path
            resolved_path = self.resolve_path(image_path)
            
            if not resolved_path.exists():
                print(f"❌ File not found: {resolved_path}")
                print(f"   Tried: {resolved_path.absolute()}")
                return None, 0
            
            if not self.is_valid_image_file(resolved_path):
                print("❌ Not a valid image file. Supported formats: PNG, JPG, JPEG, BMP, TIFF")
                return None, 0
            
            print(f"📁 Processing: {resolved_path.name}")
            
            # Open and process image
            image = Image.open(resolved_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(input_tensor)
                probabilities = torch.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            
            prediction = self.class_names[predicted.item()]
            confidence = confidence.item()
            
            print(f"\n🩺 COVID-19 Detection Result:")
            print(f"📁 Image: {resolved_path.name}")
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

def find_working_sample_images():
    """Find actual working sample images"""
    sample_dir = Path("data/sample/images")
    if sample_dir.exists():
        image_files = list(sample_dir.glob("*.png"))
        if image_files:
            print("\n📂 Working sample images found:")
            for i, img_path in enumerate(image_files[:5]):  # Show first 5
                print(f"   {i+1}. {img_path.name}")
            return image_files
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
    
    # Find working sample images
    sample_images = find_working_sample_images()
    if sample_images:
        first_image = sample_images[0]
        print(f"💡 Try this working image: {first_image.name}")
        print(f"   Full path: {first_image.absolute()}")
    
    print("\nEnter the path to a chest X-ray image (or 'quit' to exit):")
    
    while True:
        image_path = input("\n📁 Image path: ").strip().strip('"')
        
        if image_path.lower() in ['quit', 'exit', 'q']:
            print("👋 Thank you for using COVID-19 Detection System!")
            break
        
        # If user enters just a filename, try in sample directory
        if '/' not in image_path and '\\' not in image_path:
            sample_path = Path("data/sample/images") / image_path
            if sample_path.exists():
                image_path = str(sample_path)
                print(f"🔍 Found in sample directory: {image_path}")
        
        predictor.predict(image_path)

if __name__ == '__main__':
    main()