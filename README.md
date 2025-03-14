# Vehicle Detection System using YOLOv8

This project implements a vehicle detection system using the YOLOv8 object detection model. It can detect various types of vehicles including cars, motorcycles, buses, trucks, and bicycles.

## Project Structure

```
.
├── config/             # Configuration files
├── data/              # Dataset files
│   ├── processed/     # Processed images and labels
│   └── raw/          # Raw dataset files
├── models/            # Trained model weights
├── notebooks/         # Jupyter notebooks for exploration
├── results/           # Output files
│   ├── analysis/     # Analysis results and visualizations
│   ├── detections/   # Detection results on images
│   └── reports/      # Generated reports
├── src/              # Source code
│   ├── detection/    # Detection scripts
│   │   ├── run_detection.py
│   │   └── run_pretrained_inference.py
│   ├── analysis/     # Analysis scripts
│   │   ├── analyze_initial_data.py
│   │   └── analyze_pretrained_results.py
│   └── utils/        # Utility scripts
│       ├── generate_report.py
│       └── monitor_training.py
├── logs/             # Training and runtime logs
├── runs/             # Training runs and checkpoints
├── dataset.yaml      # Dataset configuration
├── requirements.txt  # Python dependencies
└── README.md        # Project documentation
```

## Setup

1. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Vehicle Detection

Run detection on a single image:
```bash
python src/detection/run_detection.py path/to/image.jpg
```

The annotated image will be saved in `results/detections/`.

### Generate Reports

Generate analysis report:
```bash
python src/utils/generate_report.py
```

The report will be saved in `results/reports/`.

### Model Training

Monitor training progress:
```bash
python src/utils/monitor_training.py
```

## Model Details

- Base Model: YOLOv8n (nano)
- Input Size: 640x640
- Supported Vehicle Classes:
  - Car
  - Motorcycle
  - Bus
  - Truck
  - Bicycle

## Results

Detection results and performance metrics can be found in `results/reports/model_report.pdf`.

## License

MIT License

Copyright (c) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. 