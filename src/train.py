import os
import yaml
import logging
from pathlib import Path
from models.detection_model import DetectionModel
from utils.logger import setup_logger
from utils.metrics import calculate_metrics
import torch
from torch.utils.tensorboard import SummaryWriter

def load_config(config_path: str) -> dict:
    """
    Load training configuration from YAML file.
    
    Args:
        config_path (str): Path to configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_training(config: dict) -> tuple:
    """
    Setup training environment and directories.
    
    Args:
        config (dict): Training configuration
        
    Returns:
        tuple: (output_dir, log_dir, writer)
    """
    # Create output directory
    output_dir = Path(config['output_dir']) / config['experiment_name']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log directory
    log_dir = output_dir / 'logs'
    log_dir.mkdir(exist_ok=True)
    
    # Setup tensorboard writer
    writer = SummaryWriter(log_dir / 'tensorboard')
    
    return output_dir, log_dir, writer

def train_model(config: dict):
    """
    Train the detection model.
    
    Args:
        config (dict): Training configuration
    """
    logger = logging.getLogger(__name__)
    
    # Setup training environment
    output_dir, log_dir, writer = setup_training(config)
    logger.info(f"Training setup completed. Output directory: {output_dir}")
    
    # Initialize model
    model = DetectionModel()
    logger.info("Model initialized")
    
    # Log model information
    model_info = model.get_model_info()
    logger.info(f"Model architecture: {model_info['architecture']}")
    logger.info(f"Total parameters: {model_info['total_params']:,}")
    
    # Train model
    try:
        results = model.train(
            data=config['data_yaml'],
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            imgsz=config['image_size'],
            project=str(output_dir),
            name='weights',
            **config['training_params']
        )
        
        # Calculate and log metrics
        metrics = calculate_metrics(results)
        logger.info("Training completed successfully")
        logger.info("Final metrics:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value}")
            
        # Log metrics to tensorboard
        for metric, value in metrics.items():
            writer.add_scalar(f'Metrics/{metric}', value, config['epochs'])
        
        # Export model
        model.export(format=config['export_format'])
        logger.info(f"Model exported in {config['export_format']} format")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise
    
    finally:
        writer.close()
    
    return results

def main():
    # Setup logging
    setup_logger('logs/training.log')
    logging.info("Starting model training...")
    
    # Load configuration
    config = load_config('config/training_config.yaml')
    
    # Train model
    results = train_model(config)
    
    logging.info("Training process completed")

if __name__ == "__main__":
    main() 