import os
import shutil
from pathlib import Path
import random
import logging
from typing import Tuple, List

def create_dataset_structure(base_path: str, train_ratio: float = 0.7, val_ratio: float = 0.2) -> None:
    """
    Create the dataset directory structure.
    
    Args:
        base_path (str): Base directory for the dataset
        train_ratio (float): Ratio of training data
        val_ratio (float): Ratio of validation data
    """
    # Create main directories
    dirs = ['images/train', 'images/val', 'images/test', 'labels/train', 'labels/val', 'labels/test']
    for dir_path in dirs:
        Path(os.path.join(base_path, dir_path)).mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Created dataset structure in {base_path}")

def split_dataset(source_dir: str, 
                 dest_dir: str,
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.2,
                 seed: int = 42) -> None:
    """
    Split dataset into train/val/test sets.
    
    Args:
        source_dir (str): Source directory containing images and labels
        dest_dir (str): Destination directory for split dataset
        train_ratio (float): Ratio of training data
        val_ratio (float): Ratio of validation data
        seed (int): Random seed for reproducibility
    """
    random.seed(seed)
    
    # Get all image files
    image_files = [f for f in os.listdir(os.path.join(source_dir, 'images')) 
                  if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Shuffle files
    random.shuffle(image_files)
    
    # Calculate split indices
    n_files = len(image_files)
    n_train = int(n_files * train_ratio)
    n_val = int(n_files * val_ratio)
    
    # Split files
    train_files = image_files[:n_train]
    val_files = image_files[n_train:n_train + n_val]
    test_files = image_files[n_train + n_val:]
    
    # Move files to respective directories
    splits = {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }
    
    for split_name, files in splits.items():
        for img_file in files:
            # Move image
            src_img = os.path.join(source_dir, 'images', img_file)
            dst_img = os.path.join(dest_dir, 'images', split_name, img_file)
            shutil.copy2(src_img, dst_img)
            
            # Move corresponding label
            label_file = os.path.splitext(img_file)[0] + '.txt'
            src_label = os.path.join(source_dir, 'labels', label_file)
            if os.path.exists(src_label):
                dst_label = os.path.join(dest_dir, 'labels', split_name, label_file)
                shutil.copy2(src_label, dst_label)
    
    logging.info(f"Dataset split completed. Train: {len(train_files)}, "
                f"Val: {len(val_files)}, Test: {len(test_files)}")

def verify_dataset(dataset_path: str) -> Tuple[bool, List[str]]:
    """
    Verify dataset structure and files.
    
    Args:
        dataset_path (str): Path to dataset directory
        
    Returns:
        Tuple[bool, List[str]]: (is_valid, error_messages)
    """
    errors = []
    is_valid = True
    
    # Check directory structure
    required_dirs = ['images/train', 'images/val', 'images/test',
                    'labels/train', 'labels/val', 'labels/test']
    
    for dir_path in required_dirs:
        if not os.path.exists(os.path.join(dataset_path, dir_path)):
            errors.append(f"Missing directory: {dir_path}")
            is_valid = False
    
    # Check for matching images and labels
    for split in ['train', 'val', 'test']:
        img_dir = os.path.join(dataset_path, 'images', split)
        label_dir = os.path.join(dataset_path, 'labels', split)
        
        if not os.path.exists(img_dir) or not os.path.exists(label_dir):
            continue
        
        images = set(os.path.splitext(f)[0] for f in os.listdir(img_dir)
                    if f.endswith(('.jpg', '.jpeg', '.png')))
        labels = set(os.path.splitext(f)[0] for f in os.listdir(label_dir)
                    if f.endswith('.txt'))
        
        # Check for missing labels
        missing_labels = images - labels
        if missing_labels:
            errors.append(f"Missing labels for images in {split}: {missing_labels}")
            is_valid = False
        
        # Check for missing images
        missing_images = labels - images
        if missing_images:
            errors.append(f"Missing images for labels in {split}: {missing_images}")
            is_valid = False
    
    return is_valid, errors 