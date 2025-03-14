import cv2
import torch
from pathlib import Path
from models.detection_model import DetectionModel
import logging
from utils.logger import setup_logger

def test_model():
    # Setup logging
    setup_logger('logs/test_model.log')
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize model
        logger.info("Initializing model...")
        model = DetectionModel(model_path='models/weights/yolo11n.pt')
        
        # Get model info
        model_info = model.get_model_info()
        logger.info(f"Model loaded successfully:")
        logger.info(f"Model type: {model_info['model_type']}")
        logger.info(f"Task: {model_info['task']}")
        logger.info(f"Total parameters: {model_info['total_params']:,}")
        
        # Check CUDA availability
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
        
        # Test inference on a sample image or webcam frame
        logger.info("\nTesting inference...")
        cap = cv2.VideoCapture(0)  # Use webcam
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            # Run inference
            results = model.predict(
                source=frame,
                conf=0.25,
                iou=0.45,
                verbose=False
            )
            
            # Process results
            detections = results[0].boxes.data.cpu().numpy()
            logger.info(f"Number of detections: {len(detections)}")
            
            # Save test image with detections
            output_dir = Path('outputs/test')
            output_dir.mkdir(parents=True, exist_ok=True)
            
            annotated_frame = results[0].plot()
            cv2.imwrite(str(output_dir / 'test_detection.jpg'), annotated_frame)
            logger.info(f"Test image saved to: {output_dir / 'test_detection.jpg'}")
            
        return True
        
    except Exception as e:
        logger.error(f"Error testing model: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_model()
    print("Model test completed successfully!" if success else "Model test failed!") 