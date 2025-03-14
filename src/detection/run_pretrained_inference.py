from ultralytics import YOLO
import os
import random
from datetime import datetime

def run_inference():
    # Load the pre-trained model
    model = YOLO('yolov8n.pt')
    
    # Path to validation images
    val_path = "/Users/humamkhurram/Desktop/AI Automotive/data/processed/images/val"
    
    # Create output directory for results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"pretrained_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of validation images
    val_images = [f for f in os.listdir(val_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Randomly select 10 images for demonstration
    sample_images = random.sample(val_images, min(10, len(val_images)))
    
    print(f"Running inference on {len(sample_images)} sample images...")
    
    # Run inference on sample images
    results = []
    for img_name in sample_images:
        img_path = os.path.join(val_path, img_name)
        result = model(img_path, save=True, project=output_dir, name="inference")
        results.extend(result)
    
    print(f"\nInference complete! Results saved in '{output_dir}/inference' directory")
    print("\nDetection Statistics:")
    
    # Calculate and display statistics
    class_counts = {}
    total_detections = 0
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if conf > 0.3:  # Consider only detections with confidence > 0.3
                class_name = model.names[cls]
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
                total_detections += 1
    
    print(f"\nTotal detections: {total_detections}")
    print("\nDetections per class:")
    for class_name, count in class_counts.items():
        print(f"{class_name}: {count}")

if __name__ == "__main__":
    run_inference() 