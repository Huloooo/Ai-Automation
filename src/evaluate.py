import os
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from models.detection_model import DetectionModel
from utils.logger import setup_logger
from utils.metrics import calculate_metrics, calculate_class_metrics, calculate_detection_speed

def evaluate_model(
    model_path: str,
    data_yaml: str,
    output_dir: str,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45
):
    """
    Evaluate model performance and generate visualizations.
    
    Args:
        model_path (str): Path to trained model
        data_yaml (str): Path to data configuration file
        output_dir (str): Directory to save evaluation results
        conf_threshold (float): Confidence threshold
        iou_threshold (float): IoU threshold
    """
    logger = logging.getLogger(__name__)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize model
    model = DetectionModel(model_path)
    logger.info(f"Loaded model from {model_path}")
    
    # Evaluate model
    results = model.evaluate(
        data=data_yaml,
        conf=conf_threshold,
        iou=iou_threshold
    )
    
    # Calculate metrics
    metrics = calculate_metrics(results)
    class_metrics = calculate_class_metrics(results)
    speed_metrics = calculate_detection_speed(results)
    
    # Log metrics
    logger.info("Overall Metrics:")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value}")
    
    logger.info("\nPer-class Metrics:")
    for class_name, class_metric in class_metrics.items():
        logger.info(f"\n{class_name}:")
        for metric, value in class_metric.items():
            logger.info(f"{metric}: {value}")
    
    logger.info("\nSpeed Metrics:")
    for metric, value in speed_metrics.items():
        logger.info(f"{metric}: {value}")
    
    # Create visualizations
    plot_metrics(metrics, class_metrics, output_dir)
    
    return metrics, class_metrics, speed_metrics

def plot_metrics(metrics: dict, class_metrics: dict, output_dir: str):
    """
    Create and save metric visualizations.
    
    Args:
        metrics (dict): Overall metrics
        class_metrics (dict): Per-class metrics
        output_dir (str): Directory to save plots
    """
    # Set style
    plt.style.use('seaborn')
    
    # Plot overall metrics
    plt.figure(figsize=(10, 6))
    plt.bar(metrics.keys(), metrics.values())
    plt.title('Overall Model Metrics')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overall_metrics.png'))
    plt.close()
    
    # Plot per-class metrics
    for metric in ['precision', 'recall', 'f1-score', 'mAP50']:
        plt.figure(figsize=(12, 6))
        class_names = list(class_metrics.keys())
        values = [class_metrics[class_name][metric] for class_name in class_names]
        plt.bar(class_names, values)
        plt.title(f'Per-class {metric.capitalize()}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'per_class_{metric}.png'))
        plt.close()
    
    # Create confusion matrix heatmap
    if 'confusion_matrix' in metrics:
        plt.figure(figsize=(10, 8))
        sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
        plt.close()

def main():
    # Setup logging
    setup_logger('logs/evaluation.log')
    logging.info("Starting model evaluation...")
    
    # Define paths
    model_path = "outputs/vehicle_pedestrian_detection/weights/best.pt"
    data_yaml = "data/dataset.yaml"
    output_dir = "outputs/evaluation"
    
    # Evaluate model
    metrics, class_metrics, speed_metrics = evaluate_model(
        model_path=model_path,
        data_yaml=data_yaml,
        output_dir=output_dir,
        conf_threshold=0.25,
        iou_threshold=0.45
    )
    
    logging.info("Model evaluation completed")

if __name__ == "__main__":
    main() 