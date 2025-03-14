import cv2
import numpy as np
from ultralytics import YOLO
import time

class VehiclePedestrianDetector:
    def __init__(self):
        # Initialize YOLOv8 model
        self.model = YOLO('yolov8n.pt')
        # Classes we're interested in (vehicles and pedestrians)
        self.target_classes = [0, 1, 2, 3, 5, 7]  # person, bicycle, car, motorcycle, bus, truck
        
    def process_frame(self, frame):
        # Run YOLOv8 inference
        results = self.model(frame, classes=self.target_classes)
        
        # Process results
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                # Get confidence
                conf = box.conf[0].cpu().numpy()
                # Get class
                cls = int(box.cls[0].cpu().numpy())
                
                # Draw bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
                # Add label
                label = f'{result.names[cls]} {conf:.2f}'
                cv2.putText(frame, label, (int(x1), int(y1) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame

def main():
    # Initialize detector
    detector = VehiclePedestrianDetector()
    
    # Initialize video capture (0 for webcam)
    cap = cv2.VideoCapture(0)
    
    # Set frame width and height
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # FPS calculation variables
    fps = 0
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process frame
        processed_frame = detector.process_frame(frame)
        
        # Calculate FPS
        frame_count += 1
        if frame_count >= 30:
            end_time = time.time()
            fps = frame_count / (end_time - start_time)
            frame_count = 0
            start_time = time.time()
        
        # Display FPS
        cv2.putText(processed_frame, f'FPS: {fps:.1f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show frame
        cv2.imshow('Vehicle and Pedestrian Detection', processed_frame)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main() 