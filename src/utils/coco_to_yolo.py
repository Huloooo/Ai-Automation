import json
import os
from pathlib import Path
import shutil
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# COCO categories we're interested in
CATEGORIES = {
    1: 0,   # person
    2: 1,   # bicycle
    3: 2,   # car
    4: 3,   # motorcycle
    6: 4,   # bus
    8: 5,   # truck
    7: 6,   # truck (will be mapped to van)
}

def convert_bbox_coco_to_yolo(img_width, img_height, bbox):
    """Convert COCO bbox to YOLO format."""
    x, y, w, h = bbox
    
    # Convert COCO bbox (x, y, width, height) to YOLO format (x_center, y_center, width, height)
    x_center = (x + w/2) / img_width
    y_center = (y + h/2) / img_height
    width = w / img_width
    height = h / img_height
    
    return [x_center, y_center, width, height]

def convert_coco_to_yolo(coco_path, output_path, split='val'):
    """Convert COCO annotations to YOLO format."""
    logger.info(f"Converting {split} annotations from COCO to YOLO format...")
    
    # Create output directories
    images_out = Path(output_path) / 'images' / split
    labels_out = Path(output_path) / 'labels' / split
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)
    
    # Load COCO annotations
    with open(coco_path, 'r') as f:
        coco = json.load(f)
    
    # Create image_id to annotations mapping
    image_annotations = {}
    for ann in coco['annotations']:
        if ann['category_id'] in CATEGORIES:
            image_id = ann['image_id']
            if image_id not in image_annotations:
                image_annotations[image_id] = []
            image_annotations[image_id].append(ann)
    
    # Process each image
    for img in tqdm(coco['images'], desc=f"Processing {split} images"):
        img_id = img['id']
        if img_id in image_annotations:
            # Get image dimensions
            img_width = img['width']
            img_height = img['height']
            
            # Create YOLO annotation file
            label_file = labels_out / f"{Path(img['file_name']).stem}.txt"
            
            # Convert annotations
            with open(label_file, 'w') as f:
                for ann in image_annotations[img_id]:
                    category_id = CATEGORIES[ann['category_id']]
                    bbox = convert_bbox_coco_to_yolo(img_width, img_height, ann['bbox'])
                    f.write(f"{category_id} {' '.join([str(x) for x in bbox])}\n")
            
            # Copy image file
            src_img = Path(coco_path).parent.parent / 'images' / f"{split}2014" / img['file_name']
            dst_img = images_out / img['file_name']
            if src_img.exists():
                shutil.copy2(src_img, dst_img)
            else:
                logger.warning(f"Image not found: {src_img}")
    
    logger.info(f"Conversion completed for {split} split!")

def main():
    """Main function to convert COCO dataset to YOLO format."""
    # Define paths
    coco_root = Path('data/raw')
    output_root = Path('data/processed')
    
    # Convert validation set
    val_ann_path = coco_root / 'annotations' / 'instances_val2014.json'
    if val_ann_path.exists():
        convert_coco_to_yolo(str(val_ann_path), str(output_root), 'val')
    else:
        logger.error(f"Validation annotations not found at {val_ann_path}")
    
    # Convert training set
    train_ann_path = coco_root / 'annotations' / 'instances_train2014.json'
    if train_ann_path.exists():
        convert_coco_to_yolo(str(train_ann_path), str(output_root), 'train')
    else:
        logger.error(f"Training annotations not found at {train_ann_path}")

if __name__ == "__main__":
    main() 