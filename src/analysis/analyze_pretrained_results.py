import os
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import numpy as np
from datetime import datetime

def analyze_results():
    # Create output directory
    output_dir = "pretrained_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # Sample results from our test
    detections = {
        'person': 15,
        'car': 6,
        'motorcycle': 1,
        'truck': 1,
        'train': 2,
        'kite': 10,
        'sports ball': 2,
        'surfboard': 1,
        'suitcase': 1,
        'elephant': 1,
        'cell phone': 1,
        'remote': 2,
        'parking meter': 1,
        'cow': 1
    }
    
    # Create detection distribution plot
    plt.figure(figsize=(15, 8))
    sns.barplot(x=list(detections.keys()), y=list(detections.values()))
    plt.xticks(rotation=45, ha='right')
    plt.title('Distribution of Detections by Class')
    plt.xlabel('Class')
    plt.ylabel('Number of Detections')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'detection_distribution.png'))
    plt.close()
    
    # Calculate statistics
    total_detections = sum(detections.values())
    avg_detections_per_image = total_detections / 10  # 10 sample images
    
    # Create statistics report
    report_path = os.path.join(output_dir, 'pretrained_analysis.txt')
    with open(report_path, 'w') as f:
        f.write("Pre-trained YOLOv8 Model Analysis\n")
        f.write("=" * 30 + "\n\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("1. Detection Statistics\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total detections: {total_detections}\n")
        f.write(f"Average detections per image: {avg_detections_per_image:.1f}\n")
        f.write(f"Number of unique classes detected: {len(detections)}\n\n")
        
        f.write("2. Class Distribution\n")
        f.write("-" * 20 + "\n")
        for cls, count in sorted(detections.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_detections) * 100
            f.write(f"{cls}: {count} ({percentage:.1f}%)\n")
        
        f.write("\n3. Key Observations\n")
        f.write("-" * 20 + "\n")
        f.write("- Person detection is most frequent (33.3% of all detections)\n")
        f.write("- Vehicle-related classes (car, motorcycle, truck, train) account for 22.2% of detections\n")
        f.write("- Model shows capability to detect multiple object types in single images\n")
        f.write("- Detection confidence threshold set at 0.3 for reliable results\n")
    
    print(f"Analysis complete! Results saved in '{output_dir}' directory")

if __name__ == "__main__":
    analyze_results() 