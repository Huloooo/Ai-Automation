import numpy as np
from typing import Dict, List, Tuple
import logging

def calculate_metrics(results: Dict) -> Dict:
    """
    Calculate various metrics from training results.
    
    Args:
        results (Dict): Training results from YOLO model
        
    Returns:
        Dict: Calculated metrics
    """
    metrics = {}
    
    try:
        # Extract metrics from results
        if hasattr(results, 'metrics'):
            metrics.update({
                'mAP50': results.metrics.get('metrics/mAP50(B)', 0),
                'mAP50-95': results.metrics.get('metrics/mAP50-95(B)', 0),
                'precision': results.metrics.get('metrics/precision(B)', 0),
                'recall': results.metrics.get('metrics/recall(B)', 0),
                'f1-score': results.metrics.get('metrics/F1(B)', 0),
                'confusion_matrix': results.metrics.get('metrics/confusion_matrix', None)
            })
        
        # Calculate additional metrics if available
        if hasattr(results, 'results_dict'):
            results_dict = results.results_dict
            metrics.update({
                'epochs': len(results_dict),
                'best_epoch': results_dict.index(max(results_dict['metrics/mAP50(B)'])),
                'final_loss': results_dict['train/box_loss'][-1],
                'learning_rates': results_dict['lr/pg0']
            })
        
        logging.info("Metrics calculated successfully")
        return metrics
        
    except Exception as e:
        logging.error(f"Error calculating metrics: {str(e)}")
        return {}

def calculate_class_metrics(results: Dict) -> Dict[str, Dict]:
    """
    Calculate per-class metrics.
    
    Args:
        results (Dict): Training results from YOLO model
        
    Returns:
        Dict: Per-class metrics
    """
    class_metrics = {}
    
    try:
        if hasattr(results, 'metrics'):
            for class_name in results.metrics.get('names', []):
                class_metrics[class_name] = {
                    'precision': results.metrics.get(f'metrics/precision({class_name})', 0),
                    'recall': results.metrics.get(f'metrics/recall({class_name})', 0),
                    'f1-score': results.metrics.get(f'metrics/F1({class_name})', 0),
                    'mAP50': results.metrics.get(f'metrics/mAP50({class_name})', 0)
                }
        
        return class_metrics
        
    except Exception as e:
        logging.error(f"Error calculating class metrics: {str(e)}")
        return {}

def calculate_detection_speed(results: Dict) -> Dict:
    """
    Calculate detection speed metrics.
    
    Args:
        results (Dict): Training results from YOLO model
        
    Returns:
        Dict: Speed metrics
    """
    speed_metrics = {}
    
    try:
        if hasattr(results, 'metrics'):
            speed_metrics.update({
                'inference_time': results.metrics.get('speed/inference', 0),
                'preprocessing_time': results.metrics.get('speed/preprocess', 0),
                'postprocessing_time': results.metrics.get('speed/postprocess', 0),
                'total_time': results.metrics.get('speed/total', 0)
            })
        
        return speed_metrics
        
    except Exception as e:
        logging.error(f"Error calculating speed metrics: {str(e)}")
        return {} 