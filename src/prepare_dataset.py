import os
import shutil
from pathlib import Path
import logging
from utils.logger import setup_logger
from utils.dataset_utils import create_dataset_structure, split_dataset, verify_dataset

def prepare_dataset(
    source_dir: str,
    output_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    seed: int = 42
):
    """
    Prepare and organize the dataset.
    
    Args:
        source_dir (str): Directory containing source images and labels
        output_dir (str): Directory to store organized dataset
        train_ratio (float): Ratio of training data
        val_ratio (float): Ratio of validation data
        seed (int): Random seed for reproducibility
    """
    logger = logging.getLogger(__name__)
    
    # Create output directory structure
    create_dataset_structure(output_dir)
    logger.info(f"Created dataset structure in {output_dir}")
    
    # Split and organize dataset
    split_dataset(
        source_dir=source_dir,
        dest_dir=output_dir,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed
    )
    logger.info("Split and organized dataset")
    
    # Verify dataset
    is_valid, error_msg = verify_dataset(output_dir)
    if is_valid:
        logger.info("Dataset verification successful")
    else:
        logger.error(f"Dataset verification failed: {error_msg}")
    
    return is_valid

def main():
    # Setup logging
    setup_logger('logs/dataset_preparation.log')
    logging.info("Starting dataset preparation...")
    
    # Define paths
    source_dir = "data/raw"
    output_dir = "data/processed"
    
    # Prepare dataset
    success = prepare_dataset(
        source_dir=source_dir,
        output_dir=output_dir,
        train_ratio=0.7,
        val_ratio=0.2,
        seed=42
    )
    
    if success:
        logging.info("Dataset preparation completed successfully")
    else:
        logging.error("Dataset preparation failed")
    
    logging.info("Dataset preparation process finished")

if __name__ == "__main__":
    main() 