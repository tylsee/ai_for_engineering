# COS40007 — AI-Based Structural Defect Detection

Detect and localise structural defects in infrastructure images using object detection models.

**5 defect classes:** cracks · spalling · corrosion · potholes · paint degradation  
**3 models compared:** YOLOv11n · YOLOv8n · SSDLite320-MobileNetV3  
**Dataset:** 4,626 labelled images across train / val / test splits

---

## Quick Start

### Option A — Local Device

**Requirements:** Python 3.10+, CUDA GPU recommended (CPU works but training is very slow)

```bash
# 1. Clone the repo
git clone https://github.com/tylsee/ai_for_engineering.git
cd ai_for_engineering

# 2. Install dependencies
#    If you have an NVIDIA GPU, install PyTorch with CUDA first:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
#    Then install everything else:
pip install -r requirements.txt

# 3. Data is already in the repo — skip straight to training
#    Train all 3 models :
python scripts/train_all.py

#    Or train one model at a time:
python scripts/train_all.py --model yolo11n
python scripts/train_all.py --model yolov8n
python scripts/train_all.py --model ssdlite

# 4. Run the Streamlit dashboard
streamlit run ui/app.py
```

---

### Option B — Google Colab 

Colab sessions can disconnect unexpectedly, so always save work to Google Drive and train one model per session.

**Step 1 — Open the setup notebook**

Upload or open `notebooks/00_colab_setup.ipynb` in Google Colab.

**Step 2 — Mount Google Drive**
```python
from google.colab import drive
drive.mount('/content/drive')
```

**Step 3 — Clone the repo into Drive (first time only)**
```bash
# First time only — creates COS40007/ai_for_engineering/ inside your Drive
!git clone https://github.com/tylsee/ai_for_engineering /content/drive/MyDrive/COS40007/ai_for_engineering
```

Every session after that, just navigate to it:
```python
%cd /content/drive/MyDrive/COS40007/ai_for_engineering
!git pull origin main   # get latest changes from teammates
```

**Step 4 — Install dependencies**
```bash
# Colab already has PyTorch with CUDA — only install the rest
!pip install -q ultralytics albumentations opencv-python Pillow pandas matplotlib \
    seaborn scikit-learn streamlit pyyaml torchmetrics tqdm \
    pycocotools faster-coco-eval pypdf
```

**Step 5 — Verify GPU**
```python
import torch
print(torch.cuda.is_available())          # should print True
print(torch.cuda.get_device_name(0))      # e.g. Tesla T4
```

**Step 6 — Train one model per session**

| Session | Command | Est. time on T4 |
|---------|---------|-----------------|
| 1 | `!python scripts/train_all.py --model yolo11n` | ~60–90 min |
| 2 | `!python scripts/train_all.py --model yolov8n` | ~60–90 min |
| 3 | `!python scripts/train_all.py --model ssdlite` | ~30–45 min |

**Step 7 — Push weights to GitHub after each session**
```bash
!git add runs/
!git commit -m "Add trained weights from Colab session"
!git push origin main
```

> The full setup notebook with all these steps pre-written is at `notebooks/00_colab_setup.ipynb`.

---

## Project Structure

```
ai_for_engineering/
├── data/
│   ├── images/{train,val,test}/   ← 4,626 labelled images
│   ├── labels/{train,val,test}/   ← YOLO .txt annotations
│   ├── data.yaml                  ← YOLO dataset config
│   └── manifest.csv               ← per-source image counts
├── models/
│   ├── ssdlite_detector.py        ← SSDLite320-MobileNetV3
│   ├── resnet_detector.py
│   └── efficientnet_detector.py
├── notebooks/
│   ├── 00_colab_setup.ipynb       ← Colab setup (start here on Colab)
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_training.ipynb
│   └── 03_evaluation.ipynb
├── scripts/
│   ├── train_all.py               ← main training script
│   ├── smoke_test.py              ← quick pipeline check (~10 sec)
│   ├── 01_reorganize_data.py      ← rebuilds data/ from raw datasets
│   └── 02_augmentation.py        ← Albumentations pipeline
├── runs/
│   ├── yolo11n/v1/                ← trained weights + metrics
│   ├── run_log.csv                ← all training runs logged here
│   └── *.png                     ← EDA charts
├── ui/
│   ├── app.py                     ← Streamlit dashboard
│   └── inference.py               ← inference helpers
├── download_datasets.py           ← download raw datasets from Google Drive
└── requirements.txt
```

---

## Training Options

```bash
# Train all 3 models sequentially
python scripts/train_all.py

# Train a specific model only
python scripts/train_all.py --model yolo11n
python scripts/train_all.py --model yolov8n
python scripts/train_all.py --model ssdlite

# Train multiple specific models
python scripts/train_all.py --model yolov8n ssdlite
```

Each run auto-increments a version: `runs/yolo11n/v1/`, `v2/`, etc.  
Results are always appended to `runs/run_log.csv` — no previous runs are overwritten.

---

## Run the Dashboard

```bash
streamlit run ui/app.py
```

Upload an image → see bounding boxes, class labels, confidence scores, and severity estimates.

---

## Run a Quick Sanity Check

Before starting a long training run, verify the pipeline works end-to-end in ~10 seconds:

```bash
python scripts/smoke_test.py
```

---

## Dataset

| Split | Images | Labels |
|-------|--------|--------|
| train | 3,237 | 3,237 |
| val | 694 | 694 |
| test | 695 | 695 |
| **Total** | **4,626** | **4,626** |

**Bounding box distribution (train set):**

| Class | BBoxes | Notes |
|-------|--------|-------|
| cracks | 3,439 | Per-object boxes from wall-crack + road-damage datasets |
| potholes | 883 | Per-object boxes from road-damage dataset |
| paint_degradation | 304 | Whole-image box (classification-only source) |
| corrosion | 267 | Whole-image box (classification-only source) |
| spalling | 236 | Whole-image box (classification-only source) |


---

## Results

| Model | mAP@0.5 | mAP@0.5:0.95 | Params | Notes |
|-------|---------|--------------|--------|-------|
| YOLOv11n | 0.7634 | 0.6651 | 2.6M | Training complete |
| YOLOv8n | — | — | ~3.2M | Training in progress |
| SSDLite-MobileNetV3 | — | — | 2.3M | Training in progress |

---

## Dependencies

```bash
# NVIDIA GPU (recommended)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# CPU only
pip install -r requirements.txt
```
