"""
Underwater Waste Detection - Inference Script
Detects waste in underwater images and calculates threat scores
"""

import argparse
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# Threat scoring system
THREAT_SCORES = {
    'plastic': {'score': 9, 'decomposition': '450-1000 years'},
    'metal': {'score': 7, 'decomposition': '50-200 years'},
    'glass': {'score': 5, 'decomposition': '1 million years'},
    'paper': {'score': 3, 'decomposition': '2-6 weeks'},
    'organic': {'score': 2, 'decomposition': '1-4 weeks'},
}

def get_threat_level(score):
    """Convert threat score to level"""
    if score >= 8:
        return "🔴 CRITICAL"
    elif score >= 6:
        return "🟠 HIGH"
    elif score >= 4:
        return "🟡 MODERATE"
    else:
        return "🟢 LOW"

def detect_waste(image_path, model_path='model/best.pt', save=True):
    """
    Detect waste in image and calculate threat score
    
    Args:
        image_path: Path to input image
        model_path: Path to trained model
        save: Whether to save output image
    """
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    
    # Run detection
    print(f"Detecting waste in {image_path}...")
    results = model(image_path)
    result = results[0]
    
    # Analyze detections
    boxes = result.boxes
    classes = boxes.cls.cpu().numpy().astype(int)
    confidences = boxes.conf.cpu().numpy()
    class_names = model.names
    
    # Calculate threat
    total_threat = 0
    waste_counts = {}
    
    for cls, conf in zip(classes, confidences):
        waste_type = class_names[cls]
        waste_counts[waste_type] = waste_counts.get(waste_type, 0) + 1
        
        threat_info = THREAT_SCORES.get(waste_type, {'score': 5})
        total_threat += threat_info['score'] * conf
    
    avg_threat = total_threat / len(classes) if len(classes) > 0 else 0
    
    # Print results
    print("\n" + "="*60)
    print("🌊 DETECTION RESULTS")
    print("="*60)
    print(f"Total waste items: {len(classes)}")
    print(f"\nWaste breakdown:")
    for waste_type, count in waste_counts.items():
        print(f"  • {waste_type.capitalize()}: {count}")
    print(f"\n⚠️ Threat Score: {avg_threat:.2f}/10")
    print(f"⚠️ Threat Level: {get_threat_level(avg_threat)}")
    print("="*60)
    
    # Save annotated image
    if save:
        output_path = Path('output') / f"detected_{Path(image_path).name}"
        output_path.parent.mkdir(exist_ok=True)
        
        annotated = result.plot()
        cv2.imwrite(str(output_path), annotated)
        print(f"\n✅ Saved result to: {output_path}")
    
    return result, avg_threat

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Underwater Waste Detection')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--model', type=str, default='model/best.pt', help='Path to model')
    parser.add_argument('--save', action='store_true', help='Save output image')
    
    args = parser.parse_args()
    
    detect_waste(args.image, args.model, args.save)
