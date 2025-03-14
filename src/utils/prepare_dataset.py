import os
import shutil
from pathlib import Path
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_directories():
    """Create necessary directories if they don't exist."""
    dirs = [
        'data/processed/images/train',
        'data/processed/images/val',
        'data/processed/labels/train',
        'data/processed/labels/val'
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {d}")

def move_files(src_dir: str, dest_dir: str, files: list):
    """Move files from source to destination directory."""
    for file in tqdm(files, desc=f"Moving files to {dest_dir}"):
        src_path = os.path.join(src_dir, file)
        dest_path = os.path.join(dest_dir, file)
        if os.path.exists(src_path):
            shutil.copy2(src_path, dest_path)
        else:
            logger.warning(f"Source file not found: {src_path}")

def prepare_dataset():
    """Prepare the dataset by moving files to the correct locations."""
    logger.info("Starting dataset preparation...")
    
    # Setup directories
    setup_directories()
    
    # Define source and destination directories
    raw_img_dir = 'data/raw/images'
    raw_label_dir = 'data/raw/labels'
    
    # Get list of files
    train_images = [f for f in os.listdir(os.path.join(raw_img_dir, 'train')) if f.endswith(('.jpg', '.jpeg', '.png'))]
    val_images = [f for f in os.listdir(os.path.join(raw_img_dir, 'val')) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Move training files
    logger.info("Moving training files...")
    move_files(
        os.path.join(raw_img_dir, 'train'),
        'data/processed/images/train',
        train_images
    )
    
    # Move validation files
    logger.info("Moving validation files...")
    move_files(
        os.path.join(raw_img_dir, 'val'),
        'data/processed/images/val',
        val_images
    )
    
    # Move corresponding label files
    logger.info("Moving label files...")
    for img_file in train_images:
        label_file = img_file.rsplit('.', 1)[0] + '.txt'
        if os.path.exists(os.path.join(raw_label_dir, 'train', label_file)):
            shutil.copy2(
                os.path.join(raw_label_dir, 'train', label_file),
                os.path.join('data/processed/labels/train', label_file)
            )
    
    for img_file in val_images:
        label_file = img_file.rsplit('.', 1)[0] + '.txt'
        if os.path.exists(os.path.join(raw_label_dir, 'val', label_file)):
            shutil.copy2(
                os.path.join(raw_label_dir, 'val', label_file),
                os.path.join('data/processed/labels/val', label_file)
            )
    
    logger.info("Dataset preparation completed!")

if __name__ == "__main__":
    prepare_dataset() 