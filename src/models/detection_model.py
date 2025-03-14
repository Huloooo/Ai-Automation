import torch
from ultralytics import YOLO
import cv2
import numpy as np
from typing import List, Tuple, Dict
import logging

class DetectionModel:
    """YOLOv8 model for object detection."""
    
    def __init__(self, model_path: str = None):
        """
        Initialize the model.
        
        Args:
            model_path (str, optional): Path to pre-trained model
        """
        self.logger = logging.getLogger(__name__)
        
        if model_path:
            self.logger.info(f"Loading pre-trained model from {model_path}")
            self.model = YOLO(model_path)
        else:
            self.logger.info("Initializing new YOLOv8n model")
            self.model = YOLO('yolov8n.pt')
    
    def train(self, **kwargs):
        """Train the model with given parameters."""
        return self.model.train(**kwargs)
    
    def predict(self, source, **kwargs):
        """Run inference on images or video."""
        return self.model.predict(source=source, **kwargs)
    
    def evaluate(self, **kwargs):
        """Evaluate model performance."""
        return self.model.val(**kwargs)
    
    def export(self, format='onnx', **kwargs):
        """Export the model to different formats."""
        return self.model.export(format=format, **kwargs)
    
    def get_model_info(self) -> Dict:
        """
        Get model information.
        
        Returns:
            Dict: Model information including type and parameters
        """
        info = {
            'model_type': self.model.type,
            'task': self.model.task,
            'total_params': sum(p.numel() for p in self.model.parameters())
        }
        return info 