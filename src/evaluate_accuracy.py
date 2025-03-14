import logging
from pathlib import Path
from models.detection_model import DetectionModel
from utils.logger import setup_logger
import yaml
import json
from typing import Dict
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(
    model_path: str,
    data_yaml: str,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45
) -> Dict:
    """
    Evaluate model performance on validation dataset.
    
    Args:
        model_path (str): Path to model weights
        data_yaml (str): Path to data configuration file
        conf_threshold (float): Confidence threshold
        iou_threshold (float): IoU threshold
    
    Returns:
        Dict: Dictionary containing evaluation metrics
    """
    logger = logging.getLogger(__name__)
    
    # Initialize model
    logger.info("Loading model...")
    model = DetectionModel(model_path)
    
    # Load dataset configuration
    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Run evaluation
    logger.info("Starting evaluation...")
    results = model.evaluate(
        data=data_yaml,
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=True
    )
    
    # Extract metrics
    metrics = {
        'mAP50': results.box.map50,          # mAP at IoU=0.5
        'mAP50-95': results.box.map,         # mAP at IoU=0.5:0.95
        'precision': results.box.mp,          # mean precision
        'recall': results.box.mr,            # mean recall
        'f1-score': 2 * (results.box.mp * results.box.mr) / (results.box.mp + results.box.mr),
    }
    
    # Get per-class metrics
    class_metrics = {}
    for i, class_name in enumerate(data_config['names']):
        class_metrics[class_name] = {
            'precision': results.box.p[i],
            'recall': results.box.r[i],
            'mAP50': results.box.ap50[i],
            'f1-score': 2 * (results.box.p[i] * results.box.r[i]) / (results.box.p[i] + results.box.r[i])
        }
    
    # Log results
    logger.info("\nOverall Metrics:")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.4f}")
    
    logger.info("\nPer-class Metrics:")
    for class_name, class_metric in class_metrics.items():
        logger.info(f"\n{class_name}:")
        for metric, value in class_metric.items():
            logger.info(f"{metric}: {value:.4f}")
    
    # Save results
    output_dir = Path('outputs/evaluation')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics to JSON
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump({
            'overall': metrics,
            'per_class': class_metrics
        }, f, indent=4)
    
    # Create visualizations
    plot_metrics(metrics, class_metrics, output_dir)
    
    return metrics, class_metrics

def plot_metrics(metrics: Dict, class_metrics: Dict, output_dir: Path):
    """Create visualization plots for metrics."""
    plt.style.use('seaborn')
    
    # Plot overall metrics
    plt.figure(figsize=(10, 6))
    plt.bar(metrics.keys(), [float(v) for v in metrics.values()])
    plt.title('Overall Model Metrics')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / 'overall_metrics.png')
    plt.close()
    
    # Plot per-class metrics
    metrics_to_plot = ['precision', 'recall', 'mAP50', 'f1-score']
    for metric in metrics_to_plot:
        plt.figure(figsize=(12, 6))
        class_names = list(class_metrics.keys())
        values = [class_metrics[name][metric] for name in class_names]
        
        plt.bar(class_names, values)
        plt.title(f'Per-class {metric}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / f'per_class_{metric}.png')
        plt.close()
    
    # Create confusion matrix heatmap if available
    if hasattr(metrics, 'confusion_matrix'):
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            metrics['confusion_matrix'],
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=list(class_metrics.keys()),
            yticklabels=list(class_metrics.keys())
        )
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(output_dir / 'confusion_matrix.png')
        plt.close()

def main():
    # Setup logging
    setup_logger('logs/evaluation.log')
    logging.info("Starting model evaluation...")
    
    # Evaluate model
    metrics, class_metrics = evaluate_model(
        model_path='models/weights/yolo11n.pt',
        data_yaml='data/dataset.yaml',
        conf_threshold=0.25,
        iou_threshold=0.45
    )
    
    logging.info("Evaluation completed!")

if __name__ == "__main__":
    main() 