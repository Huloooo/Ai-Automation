from ultralytics import YOLO
import cv2
import os
import sys

def run_detection(image_path):
    """Run vehicle detection on a single image"""
    # Load the model
    model = YOLO('yolov8n.pt')
    
    # Run inference
    results = model(image_path)
    
    # Process results
    for result in results:
        # Get the image
        im_array = result.plot()  # Plot results image
        
        # Save the annotated image
        output_path = f"detection_results_{os.path.basename(image_path)}"
        cv2.imwrite(output_path, im_array)
        
        # Print detections
        boxes = result.boxes
        print("\nDetections:")
        for box in boxes:
            # Get class name and confidence
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])
            
            # Only show vehicle-related detections with confidence > 0.3
            vehicle_classes = ['car', 'motorcycle', 'bus', 'truck', 'bicycle']
            if class_name in vehicle_classes and confidence > 0.3:
                print(f"{class_name}: {confidence:.2%} confidence")
        
        print(f"\nResults saved to: {output_path}")

def main():
    print("Vehicle Detection using YOLOv8")
    print("------------------------------")
    
    if len(sys.argv) < 2:
        print("Usage: python run_detection.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        sys.exit(1)
    
    run_detection(image_path)

if __name__ == "__main__":
    main() 