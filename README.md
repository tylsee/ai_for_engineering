# AI Infrastructure Defect Detection & Analysis

An end-to-end computer vision and severity assessment pipeline for detecting structural defects (such as Potholes, Cracks, Spalling, and Corrosion). This project evaluates multiple state-of-the-art object detection models and features a real-time demonstrator that calculates a weighted structural health score based on defect criticality, spatial impact, and model confidence.

---

## Key Features

* **Multi-Model Evaluation:** Trains and compares five leading object detection architectures: YOLOv8s, YOLOv11s, RT-DETR, Faster R-CNN, and EfficientDet.
* **Intelligent Severity Scoring:** Goes beyond simple detection by using a weighted multi-criteria algorithm (`analyser.py`) to calculate a 0–100% structural risk gauge.
* **Interactive GUI:** Includes a demonstrator application (`app/main.py`) for processing images, video streams, and real-time webcam feeds.
* **Comprehensive Metrics:** Detailed performance profiles, confusion matrices, and precision-recall curves for every model are documented.

---

## Repository Structure

The project is organized into modular directories for data, training, inference, and evaluation:

* **`app/`**: Contains the GUI demonstrator application (`main.py`).
* **`data/`**: Directory for the dataset (`train`, `test`) and the configuration file (`data.yaml`).
* **`docs/`**: Training metrics, confusion matrices, F1 curves, and visual evaluation results for all five trained models.
* **`models/`**: Stores the exported best weights (`.pt` and `.pth` files) for inference.
* **`notebooks/`**: Jupyter notebooks used for data preparation, training individual models, and final evaluation.
* **`src/`**: Core backend logic.
  * `inference.py`: Handles model loading and prediction generation.
  * `analyser.py`: Contains the `DefectAnalyzer` class for drawing dynamic bounding boxes and calculating the weighted severity index.
* **`requirements.txt`**: Python dependencies required to run the pipeline.

---

## Installation

1. Clone the repository to your local machine:

```bash
git clone https://github.com/your-username/ai_for_engineering.git
cd ai_for_engineering
```

2. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Running the AI Demonstrator (GUI)

```bash
python app/main.py
```

*(Note: If the app is built with Streamlit, run `streamlit run app/main.py` instead).*

### 2. Running Inference via CLI

```bash
python src/inference.py --source path/to/your/image.jpg --model models/yolov11s_best.pt
```

### 3. Training and Evaluation

```bash
jupyter notebook
```

Open `notebooks/evaluation.ipynb` to view the comparative analysis of model performances (mAP, inference speed, etc.).

---

## Models Included

1. **YOLOv11s** (`yolov11s_best.pt`) - Optimized for fast, real-time edge inference.
2. **YOLOv8s** (`yolov8s_best.pt`) - Reliable baseline for single-stage detection.
3. **RT-DETR** (`rt_detr_best.pt`) - Real-Time DEtection TRansformer for high accuracy.
4. **Faster R-CNN** (`faster_rcnn_resnet18_best.pth`) - Two-stage detector prioritizing localization precision.
5. **EfficientDet** (`efficientdet_best.pth`) - Scalable architecture balancing speed and accuracy.

Metrics for each model (Confusion Matrices, PR Curves) can be found in the respective folders under `docs/`.
