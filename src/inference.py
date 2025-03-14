import cv2
import numpy as np
from pathlib import Path
import logging
from models.detection_model import DetectionModel
from utils.logger import setup_logger
import time
from typing import Optional, Tuple

class RealTimeDetector:
    def __init__(self, 
                 model_path: str,
                 conf_threshold: float = 0.25,
                 iou_threshold: float = 0.45):
        """
        Initialize real-time detector.
        
        Args:
            model_path (str): Path to trained model
            conf_threshold (float): Confidence threshold
            iou_threshold (float): IoU threshold
        """
        self.logger = logging.getLogger(__name__)
        self.model = DetectionModel(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, list]:
        """
        Process a single frame.
        
        Args:
            frame (np.ndarray): Input frame
            
        Returns:
            Tuple[np.ndarray, list]: Processed frame and detections
        """
        # Perform detection
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )
        
        # Draw detections
        annotated_frame = results[0].plot()
        
        # Update FPS
        self.frame_count += 1
        if self.frame_count % 30 == 0:  # Update FPS every 30 frames
            elapsed_time = time.time() - self.start_time
            self.fps = self.frame_count / elapsed_time
        
        # Add FPS to frame
        cv2.putText(annotated_frame, f'FPS: {self.fps:.1f}',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return annotated_frame, results[0].boxes.data.cpu().numpy()
    
    def run(self, 
            source: int = 0,
            output_path: Optional[str] = None,
            display: bool = True):
        """
        Run real-time detection.
        
        Args:
            source (int): Camera index or video path
            output_path (str, optional): Path to save output video
            display (bool): Whether to display the video
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            self.logger.error(f"Failed to open video source: {source}")
            return
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Initialize video writer if output path is provided
        writer = None
        if output_path:
            writer = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps,
                (width, height)
            )
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                processed_frame, detections = self.process_frame(frame)
                
                # Write frame if output path is provided
                if writer:
                    writer.write(processed_frame)
                
                # Display frame
                if display:
                    cv2.imshow('Real-time Detection', processed_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
        except KeyboardInterrupt:
            self.logger.info("Stopping detection...")
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()

def main():
    # Setup logging
    setup_logger('logs/inference.log')
    logging.info("Starting inference...")
    
    # Initialize detector
    detector = RealTimeDetector(
        model_path='outputs/vehicle_pedestrian_detection/weights/best.pt',
        conf_threshold=0.25,
        iou_threshold=0.45
    )
    
    # Run detection
    detector.run(
        source=0,  # Use default camera
        output_path='outputs/detection_output.mp4',
        display=True
    )
    
    logging.info("Inference completed")

if __name__ == "__main__":
    main() 