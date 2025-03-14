import os
import shutil
import cv2
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict, Tuple, Optional
import json
from tqdm import tqdm

class DatasetConverter:
    """Utility class to convert various dataset formats to YOLO format."""
    
    def __init__(self, class_mapping: Dict[str, int]):
        """
        Initialize the converter with class mapping.
        
        Args:
            class_mapping (Dict[str, int]): Mapping from class names to class indices
        """
        self.class_mapping = class_mapping
        self.logger = logging.getLogger(__name__)
    
    def convert_bbox(self, bbox: List[float], img_width: int, img_height: int) -> List[float]:
        """
        Convert bounding box to YOLO format (normalized coordinates).
        
        Args:
            bbox (List[float]): [x_min, y_min, x_max, y_max]
            img_width (int): Image width
            img_height (int): Image height
            
        Returns:
            List[float]: [x_center, y_center, width, height] normalized
        """
        x_min, y_min, x_max, y_max = bbox
        
        # Convert to normalized coordinates
        x_center = (x_min + x_max) / 2.0 / img_width
        y_center = (y_min + y_max) / 2.0 / img_height
        width = (x_max - x_min) / img_width
        height = (y_max - y_min) / img_height
        
        return [x_center, y_center, width, height]
    
    def convert_coco_annotation(self, annotation: dict, img_width: int, img_height: int) -> Optional[str]:
        """
        Convert COCO format annotation to YOLO format.
        
        Args:
            annotation (dict): COCO annotation
            img_width (int): Image width
            img_height (int): Image height
            
        Returns:
            Optional[str]: YOLO format annotation line or None if class not in mapping
        """
        category_id = annotation['category_id']
        category_name = annotation.get('category_name', str(category_id))
        
        # Skip if class not in mapping
        if category_name not in self.class_mapping:
            return None
            
        class_idx = self.class_mapping[category_name]
        bbox = annotation['bbox']  # [x_min, y_min, width, height]
        
        # Convert COCO bbox to [x_min, y_min, x_max, y_max]
        x_min, y_min, width, height = bbox
        x_max = x_min + width
        y_max = y_min + height
        
        # Convert to YOLO format
        yolo_bbox = self.convert_bbox([x_min, y_min, x_max, y_max], img_width, img_height)
        
        # Create YOLO annotation line
        return f"{class_idx} {' '.join([f'{x:.6f}' for x in yolo_bbox])}"
    
    def convert_voc_annotation(self, annotation: dict, img_width: int, img_height: int) -> Optional[str]:
        """
        Convert Pascal VOC format annotation to YOLO format.
        
        Args:
            annotation (dict): VOC annotation
            img_width (int): Image width
            img_height (int): Image height
            
        Returns:
            Optional[str]: YOLO format annotation line or None if class not in mapping
        """
        class_name = annotation['name']
        
        # Skip if class not in mapping
        if class_name not in self.class_mapping:
            return None
            
        class_idx = self.class_mapping[class_name]
        bbox = annotation['bndbox']
        
        # Convert VOC bbox to [x_min, y_min, x_max, y_max]
        x_min = float(bbox['xmin'])
        y_min = float(bbox['ymin'])
        x_max = float(bbox['xmax'])
        y_max = float(bbox['ymax'])
        
        # Convert to YOLO format
        yolo_bbox = self.convert_bbox([x_min, y_min, x_max, y_max], img_width, img_height)
        
        # Create YOLO annotation line
        return f"{class_idx} {' '.join([f'{x:.6f}' for x in yolo_bbox])}"
    
    def process_image(self, 
                     image_path: str, 
                     output_image_path: str,
                     output_label_path: str,
                     annotations: List[dict],
                     format: str = 'coco') -> bool:
        """
        Process a single image and its annotations.
        
        Args:
            image_path (str): Path to source image
            output_image_path (str): Path to save processed image
            output_label_path (str): Path to save YOLO label
            annotations (List[dict]): List of annotations for the image
            format (str): Annotation format ('coco' or 'voc')
            
        Returns:
            bool: True if processing successful, False otherwise
        """
        try:
            # Read and verify image
            img = cv2.imread(image_path)
            if img is None:
                self.logger.error(f"Failed to read image: {image_path}")
                return False
                
            height, width = img.shape[:2]
            
            # Convert annotations
            yolo_annotations = []
            for ann in annotations:
                if format == 'coco':
                    yolo_ann = self.convert_coco_annotation(ann, width, height)
                else:  # VOC format
                    yolo_ann = self.convert_voc_annotation(ann, width, height)
                    
                if yolo_ann is not None:
                    yolo_annotations.append(yolo_ann)
            
            # Save image and annotations
            os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
            os.makedirs(os.path.dirname(output_label_path), exist_ok=True)
            
            cv2.imwrite(output_image_path, img)
            
            with open(output_label_path, 'w') as f:
                f.write('\n'.join(yolo_annotations))
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing {image_path}: {str(e)}")
            return False
    
    def convert_dataset(self,
                       source_dir: str,
                       output_dir: str,
                       annotation_file: Optional[str] = None,
                       format: str = 'coco'):
        """
        Convert a dataset to YOLO format.
        
        Args:
            source_dir (str): Source directory containing images
            output_dir (str): Output directory for converted dataset
            annotation_file (str, optional): Path to annotation file (for COCO)
            format (str): Dataset format ('coco' or 'voc')
        """
        self.logger.info(f"Converting {format.upper()} dataset to YOLO format...")
        
        # Create output directories
        images_dir = os.path.join(output_dir, 'images')
        labels_dir = os.path.join(output_dir, 'labels')
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        
        if format == 'coco':
            # Load COCO annotations
            with open(annotation_file, 'r') as f:
                coco_data = json.load(f)
            
            # Create image id to annotations mapping
            image_annotations = {}
            for ann in coco_data['annotations']:
                img_id = ann['image_id']
                if img_id not in image_annotations:
                    image_annotations[img_id] = []
                image_annotations[img_id].append(ann)
            
            # Process each image
            for img_info in tqdm(coco_data['images'], desc="Converting images"):
                img_id = img_info['id']
                img_file = img_info['file_name']
                
                source_image = os.path.join(source_dir, img_file)
                output_image = os.path.join(images_dir, img_file)
                output_label = os.path.join(labels_dir, Path(img_file).stem + '.txt')
                
                annotations = image_annotations.get(img_id, [])
                self.process_image(source_image, output_image, output_label, annotations, format)
                
        else:  # VOC format
            # Process each image in source directory
            for img_file in tqdm(os.listdir(source_dir), desc="Converting images"):
                if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                    
                base_name = Path(img_file).stem
                source_image = os.path.join(source_dir, img_file)
                source_annotation = os.path.join(source_dir, 'Annotations', f"{base_name}.xml")
                
                output_image = os.path.join(images_dir, img_file)
                output_label = os.path.join(labels_dir, f"{base_name}.txt")
                
                # Load VOC annotation
                if os.path.exists(source_annotation):
                    with open(source_annotation, 'r') as f:
                        annotation_data = f.read()
                    # Parse XML and convert to dict format
                    # Note: You'll need to implement XML parsing based on your needs
                    annotations = []  # Parse XML to list of annotations
                    
                    self.process_image(source_image, output_image, output_label, annotations, format)
        
        self.logger.info("Dataset conversion completed!")

def main():
    # Example usage
    class_mapping = {
        'person': 0,
        'car': 1,
        'motorcycle': 2,
        'bus': 3,
        'truck': 4,
        'bicycle': 5,
        'van': 6,
        'other-vehicle': 7
    }
    
    converter = DatasetConverter(class_mapping)
    
    # Convert COCO format dataset
    converter.convert_dataset(
        source_dir='data/raw/images',
        output_dir='data/processed',
        annotation_file='data/raw/annotations.json',
        format='coco'
    )

if __name__ == "__main__":
    main() 