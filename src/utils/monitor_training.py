import os
import time
import json
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

class TrainingMonitor:
    def __init__(self, train_dir="runs/detect/train4"):
        self.train_dir = train_dir
        self.metrics_file = os.path.join(train_dir, "results.csv")
        self.last_modified = 0
        self.metrics = []
        
    def read_metrics(self):
        """Read the metrics file if it exists and has been modified"""
        try:
            if os.path.exists(self.metrics_file):
                current_modified = os.path.getmtime(self.metrics_file)
                if current_modified > self.last_modified:
                    with open(self.metrics_file, 'r') as f:
                        lines = f.readlines()
                        if len(lines) > 1:  # Header + at least one row
                            header = lines[0].strip().split(',')
                            latest = lines[-1].strip().split(',')
                            metrics = dict(zip(header, latest))
                            self.metrics.append(metrics)
                            self.last_modified = current_modified
                            return True
            return False
        except Exception as e:
            print(f"Error reading metrics: {e}")
            return False
    
    def print_progress(self):
        """Print the current training progress"""
        if not self.metrics:
            print("\nWaiting for first epoch to complete...")
            return
        
        latest = self.metrics[-1]
        epoch = latest.get('epoch', 'N/A')
        mAP50 = latest.get('mAP50', 'N/A')
        mAP50_95 = latest.get('mAP50-95', 'N/A')
        precision = latest.get('precision', 'N/A')
        recall = latest.get('recall', 'N/A')
        
        print("\nTraining Progress:")
        print(f"Epoch: {epoch}")
        print(f"mAP50: {mAP50}")
        print(f"mAP50-95: {mAP50_95}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        
    def monitor(self, interval=5):
        """Monitor training progress continuously"""
        print(f"Monitoring training progress in {self.train_dir}")
        print("Press Ctrl+C to stop monitoring")
        
        try:
            while True:
                if self.read_metrics():
                    self.print_progress()
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\nStopped monitoring")

if __name__ == "__main__":
    monitor = TrainingMonitor()
    monitor.monitor() 