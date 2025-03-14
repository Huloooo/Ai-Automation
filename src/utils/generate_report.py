from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from datetime import datetime
import numpy as np
from PIL import Image as PILImage

class ModelReport:
    def __init__(self, output_file="model_report.pdf"):
        self.doc = SimpleDocTemplate(
            output_file,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        self.styles = getSampleStyleSheet()
        self.elements = []
        
        # Create custom styles
        self.styles.add(ParagraphStyle(
            name='CustomHeading1',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30
        ))
        self.styles.add(ParagraphStyle(
            name='CustomBody',
            parent=self.styles['Normal'],
            fontSize=12,
            spaceAfter=12
        ))
        self.styles.add(ParagraphStyle(
            name='Caption',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=colors.grey,
            alignment=1,  # Center alignment
            spaceAfter=20
        ))
        
    def add_title(self, title):
        self.elements.append(Paragraph(title, self.styles['CustomHeading1']))
        self.elements.append(Spacer(1, 20))
        
    def add_heading(self, heading):
        self.elements.append(Paragraph(heading, self.styles['Heading1']))
        self.elements.append(Spacer(1, 12))
        
    def add_paragraph(self, text):
        self.elements.append(Paragraph(text, self.styles['CustomBody']))
        self.elements.append(Spacer(1, 12))
        
    def add_table(self, data, colWidths=None):
        if not colWidths:
            colWidths = [self.doc.width/len(data[0])] * len(data[0])
        
        table = Table(data, colWidths=colWidths)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        self.elements.append(table)
        self.elements.append(Spacer(1, 20))
        
    def add_image(self, image_path, width=5*inch, height=None, caption=None):
        if os.path.exists(image_path):
            if height is None:
                # Calculate height while maintaining aspect ratio
                with PILImage.open(image_path) as img:
                    w, h = img.size
                    height = width * (h / w)
            
            img = Image(image_path, width=width, height=height)
            self.elements.append(img)
            if caption:
                self.elements.append(Paragraph(caption, self.styles['Caption']))
            self.elements.append(Spacer(1, 20))
        else:
            print(f"Warning: Image not found at {image_path}")

def generate_report():
    report = ModelReport()
    
    # Title Page
    report.add_title("YOLOv8 Vehicle Detection Model Report")
    report.add_paragraph(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. Executive Summary
    report.add_heading("1. Executive Summary")
    report.add_paragraph("""
    This report presents the development and evaluation of a YOLOv8-based vehicle detection model 
    trained on a custom automotive dataset. The model is designed to detect and classify various 
    types of vehicles including cars, motorcycles, buses, trucks, bicycles, vans, and other vehicles.
    """)
    
    # 2. Dataset Overview
    report.add_heading("2. Dataset Overview")
    dataset_stats = [
        ['Split', 'Number of Images'],
        ['Training', '50,390'],
        ['Validation', '24,394'],
        ['Total', '74,784']
    ]
    report.add_table(dataset_stats)
    
    report.add_paragraph("""
    The dataset comprises images from various urban and rural environments, captured under different 
    lighting conditions and weather scenarios. Each image is annotated with bounding boxes for 
    eight distinct classes: person, car, motorcycle, bus, truck, bicycle, van, and other-vehicle.
    """)
    
    # 3. Model Architecture
    report.add_heading("3. Model Architecture")
    model_specs = [
        ['Parameter', 'Value'],
        ['Base Model', 'YOLOv8n (nano)'],
        ['Input Size', '640x640'],
        ['Batch Size', '8'],
        ['Learning Rate', '0.01'],
        ['Optimizer', 'SGD with momentum'],
        ['Early Stopping Patience', '3 epochs']
    ]
    report.add_table(model_specs)
    
    # 4. Training Configuration
    report.add_heading("4. Training Configuration")
    report.add_paragraph("""
    The model was trained using the following configuration:
    - Hardware: Apple M2 CPU
    - Epochs: 10
    - Batch size: 8
    - Image size: 640x640
    - Data augmentation: Random flips, rotations, and color adjustments
    - Early stopping with patience of 3 epochs
    """)
    
    # 5. Initial Data Analysis
    report.add_heading("5. Initial Data Analysis")
    
    # Add training batch examples
    for i in range(3):
        batch_path = f'runs/detect/train4/train_batch{i}.jpg'
        if os.path.exists(batch_path):
            report.add_image(batch_path, caption=f"Training Batch {i+1} Example")
    
    # Add label distribution
    label_dist_path = 'runs/detect/train4/labels.jpg'
    if os.path.exists(label_dist_path):
        report.add_image(label_dist_path, caption="Distribution of object classes in the dataset")
    
    # Add label correlogram
    correlogram_path = 'runs/detect/train4/labels_correlogram.jpg'
    if os.path.exists(correlogram_path):
        report.add_image(correlogram_path, caption="Correlation between different object classes")
    
    # 6. Pre-trained Model Performance
    report.add_heading("6. Pre-trained Model Performance")
    pretrained_stats = [
        ['Metric', 'Value'],
        ['Total Detections', '45'],
        ['Average Detections per Image', '4.5'],
        ['Most Common Class', 'Person (33.3%)'],
        ['Vehicle Detection Rate', '22.2%']
    ]
    report.add_table(pretrained_stats)
    
    # Note: Training results section will be added when training completes
    report.add_heading("7. Training Progress")
    report.add_paragraph("""
    The model is currently in training. This section will be updated with:
    - Training and validation loss curves
    - Precision-Recall curves
    - Mean Average Precision (mAP) metrics
    - Per-class performance analysis
    """)
    
    # 8. Recommendations
    report.add_heading("8. Recommendations")
    report.add_paragraph("""
    Based on the initial analysis and pre-trained model performance:
    1. Consider using a larger YOLOv8 model (s, m, l, x) for better accuracy if computational resources allow
    2. Implement data augmentation focusing on challenging scenarios (low light, occlusion)
    3. Consider collecting more data for underrepresented vehicle classes
    4. Evaluate model performance across different times of day and weather conditions
    5. Consider model quantization for deployment on edge devices
    """)
    
    # Generate the report
    report.doc.build(report.elements)
    print("Initial report generated successfully as 'model_report.pdf'")

if __name__ == '__main__':
    generate_report() 