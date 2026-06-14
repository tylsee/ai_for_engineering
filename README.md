# COS40007 — AI-Based Structural Defect Detection

Detect and localise structural defects in infrastructure images using object detection models.

**5 defect classes:** corrosion · cracks · paint degradation · potholes · spalling  
**5 models compared:** YOLOv11s · YOLOv8s · EfficientDet-D0 · Faster R-CNN · RT-DETR-L  
**Dataset:** 6,422 labelled images · Dataset 10 (Roboflow v10)

---

## Team Members

| # | Name | Role |
|---|------|------|
| 1 | Norman | Training Lead — YOLOv11s, YOLOv8s |
| 2 | Emily | Data & Evaluation — EfficientDet-D0, dataset preparation |
| 3 | Adil | AI Evaluation & XAI Specialist — evaluation pipeline, IoU/FP/FN analysis, explainability |
| 4 | Tyler | Deployment — Streamlit dashboard (`app/main.py`) |

---

## Quick Start

### Option A — Google Colab (recommended for training)

1. Upload `data/` as `defect_dataset.zip` to Google Drive at `MyDrive/COS40007/defect_dataset.zip`.
2. Open **`notebooks/colab_train_evaluate.ipynb`** in Colab → **Runtime → T4 GPU → Run all**.
3. Download `runs_results.zip` from Drive and unzip into your local `runs/`.

For model-specific training, use the dedicated notebooks:
- `notebooks/train_yolo8_yolo11.ipynb` — YOLOv8s + YOLOv11s
- `notebooks/train_efficientdet.ipynb` — EfficientDet-D0
- `notebooks/train_faster_r-cnn.ipynb` — Faster R-CNN
- `notebooks/train_rt_detr.ipynb` — RT-DETR-L

### Option B — Local

```bash
git clone https://github.com/tylsee/ai_for_engineering.git
cd ai_for_engineering

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# Run the Streamlit dashboard
streamlit run app/main.py
```

---

## Project Structure

```
ai_for_engineering/
├── app/
│   └── main.py                       ← Streamlit dashboard
├── data/
│   ├── images/{train,val,test}/      ← 6,422 labelled images (4493/964/965)
│   ├── labels/{train,val,test}/      ← YOLO .txt annotations
│   ├── data.yaml                     ← YOLO dataset config
│   ├── manifest.csv                  ← per-source image counts
│   └── unseen/images/                ← 15 manually collected test images (generalisation)
├── models/                           ← trained checkpoints
│   ├── yolo11s.pt
│   ├── yolov8s.pt
│   ├── efficientdet_d0.pth
│   ├── faster_rcnn.pth
│   └── rt_detr_l.pth
├── notebooks/
│   ├── 03_evaluation.ipynb           ← Adil's evaluation: IoU, FP/FN analysis, XAI, unseen data
│   ├── evaluation.ipynb              ← Emily's evaluation: mAP table, per-class AP, confusion matrices
│   ├── colab_train_evaluate.ipynb    ← all-in-one Colab notebook
│   ├── train_yolo8_yolo11.ipynb      ← YOLOv8s + YOLOv11s training
│   ├── train_efficientdet.ipynb      ← EfficientDet-D0 training
│   ├── train_faster_r-cnn.ipynb      ← Faster R-CNN training
│   └── train_rt_detr.ipynb           ← RT-DETR-L training
├── runs/
│   ├── *.png                         ← evaluation charts (IoU histograms, FP/FN, XAI, comparison)
│   ├── run_log.csv                   ← all training runs
│   ├── yolov11s/smartcampus_640/     ← YOLOv11s results
│   ├── yolov8s/smartcampus_640/      ← YOLOv8s results
│   ├── yolov8n/v2/                   ← YOLOv8n v2 (Adil's training run)
│   ├── efficientdet_d0/v1/           ← EfficientDet-D0 results
│   ├── faster_rcnn/v1/               ← Faster R-CNN results
│   └── rtdetr_l/v1/                  ← RT-DETR-L results
├── scripts/                          ← data preprocessing and training scripts
├── src/                              ← source modules
├── reports/                          ← project report drafts
└── requirements.txt
```

---

## Dataset

| Split | Images |
|-------|--------|
| train | 4,493 |
| val | 964 |
| test | 965 |
| **Total** | **6,422** |

**5 defect classes (Dataset 10, Roboflow v10):**

| ID | Class |
|----|-------|
| 0 | corrosion |
| 1 | cracks |
| 2 | paint_degradation |
| 3 | potholes |
| 4 | spalling |

---

## Evaluation Results

All models evaluated on the 649-image test set. IoU/FP/FN analysis on a 300-image diagnostic subset.

| Model | mAP@50 | mAP@50-95 | Precision | Recall | F1 | mIoU |
|-------|--------|-----------|-----------|--------|----|------|
| **YOLOv11s** | **0.7799** | **0.5825** | 0.8250 | 0.7029 | 0.7591 | **0.850** |
| YOLOv8s | 0.7587 | 0.5472 | 0.7877 | 0.6863 | 0.7335 | 0.833 |
| EfficientDet-D0* | 0.7200 | 0.4920 | 0.7947 | **0.7692** | 0.7812 | 0.840 |
| Faster R-CNN | 0.6967 | 0.4054 | 0.4908 | 0.7554 | 0.5922 | 0.782 |
| RT-DETR-L | 0.5800 | 0.3510 | 0.7420 | — | 0.5800 | 0.770 |

*EfficientDet-D0 was retrained after correcting a class-label indexing bug. See `reports/` for details.

**Recommended model: YOLOv11s** — best mAP@50, highest mIoU, and most balanced FP/FN (127/127).

---

## Evaluation Notebooks

### `notebooks/03_evaluation.ipynb` (Adil)
- Loads all 5 model checkpoints and runs inference on the test set
- Computes IoU distributions per model and per class
- Generates FP/FN error analysis charts with worst-case example grids
- Runs XAI (Grad-CAM / attention heatmaps) on YOLOv11s
- Tests generalisation on 15 manually collected unseen images (`data/unseen/`)
- Outputs: `runs/iou_histogram*.png`, `runs/fp_fn_by_class.png`, `runs/xai_*.png`, `runs/unseen_predictions*.png`

### `notebooks/evaluation.ipynb` (Emily)
- Primary mAP evaluation using the Dataset 10 test split
- Per-class AP tables and confusion matrices for all 5 models
- Source for Table 4 values in the project report

---

## Dashboard

```bash
streamlit run app/main.py
```

Upload an image to get bounding box predictions, class labels, and confidence scores from any of the 5 trained models.

---

## Dependencies

```bash
# NVIDIA GPU (recommended)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# CPU only
pip install -r requirements.txt
```
