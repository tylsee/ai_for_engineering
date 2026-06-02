# COS40007 — AI-Based Structural Defect Detection

Detect and localise structural defects in infrastructure images using object detection models.

**5 defect classes:** cracks · spalling · corrosion · potholes · paint degradation
**3 models compared:** YOLOv11n · YOLOv8n · SSDLite320-MobileNetV3
**Dataset:** 6,422 labelled images · 10,000 boxes · perfectly balanced (2,000 boxes/class)

---

## Quick Start

### Option A — Google Colab (recommended)

Local training is slow on a weak GPU. The all-in-one notebook trains + evaluates everything
in one "Run all", and saves results back to Drive.

1. **Rebuild + zip the data** (locally):
   ```bash
   python scripts/01_reorganize_data.py     # rebuild data/ (only if sources changed)
   ```
   then create `defect_dataset.zip` from `data/` (archive root must contain `data/`).
2. **Upload** `defect_dataset.zip` to Google Drive at `MyDrive/COS40007/defect_dataset.zip`.
3. Open **`notebooks/colab_train_evaluate.ipynb`** in Colab → **Runtime → Change runtime type → T4 GPU**.
4. **Runtime → Run all.** The notebook mounts Drive, copies + unzips the data locally, trains
   YOLOv11n + YOLOv8n + SSDLite, evaluates on the test set, and writes
   `runs_results.zip` (weights + charts) back to `MyDrive/COS40007/`.
5. Download `runs_results.zip`, unzip into your local `runs/`, then run the dashboard (Option B step 4).

> The notebook is self-contained — it needs only the data zip (the SSDLite model is inlined).
> It auto-scales batch size to the GPU and also runs locally if you open it outside Colab.

### Option B — Local Device

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

# 3. (If data/ changed) rebuild it; otherwise it is already in the repo
python scripts/01_reorganize_data.py

# 4. Train all 3 models (auto-versioned into runs/)
python scripts/train_all.py
#    ...or one at a time:
python scripts/train_all.py --model yolo11n

# 5. Run the Streamlit dashboard
streamlit run ui/app.py
```

---

## Project Structure

```
ai_for_engineering/
├── data/
│   ├── images/{train,val,test}/   ← 6,422 labelled images (4493/964/965)
│   ├── labels/{train,val,test}/   ← YOLO .txt annotations
│   ├── data.yaml                  ← YOLO dataset config (rewritten at runtime)
│   └── manifest.csv               ← per-source image counts
├── dataset/                       ← raw source datasets (gitignored, on Drive)
├── models/
│   └── ssdlite_detector.py        ← SSDLite320-MobileNetV3
├── notebooks/
│   ├── colab_train_evaluate.ipynb ← ALL-IN-ONE for Colab (start here on Colab)
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_training.ipynb
│   └── 03_evaluation.ipynb
├── scripts/
│   ├── 01_reorganize_data.py      ← rebuilds data/ from raw datasets (dedup+balance+split)
│   ├── 02_augmentation.py         ← Albumentations pipeline
│   ├── train_all.py               ← train all 3 models locally
│   ├── smoke_test.py              ← quick pipeline check (~10 sec)
│   └── download_datasets.py       ← download raw datasets from Google Drive
├── runs/
│   ├── yolo11n/v1/                ← trained weights + metrics (auto-versioned)
│   ├── run_log.csv                ← all training runs logged here
│   └── *.png                      ← EDA + evaluation charts
├── ui/
│   ├── app.py                     ← Streamlit dashboard (with model-version selector)
│   └── inference.py               ← inference helpers
├── weights/                       ← COCO-pretrained YOLO base weights
└── requirements.txt
```

---

## Dataset

| Split | Images | Labels |
|-------|--------|--------|
| train | 4,493 | 4,493 |
| val | 964 | 964 |
| test | 965 | 965 |
| **Total** | **6,422** | **6,422** |

**Bounding-box distribution (perfectly balanced — 2,000 boxes/class, ratio 1.00x):**

| Class | Boxes | Source |
|-------|-------|--------|
| cracks | 2,000 | wall-crack + road-damage (real per-object boxes) |
| spalling | 2,000 | Roboflow spalling / spalling2 / spalling3 (real boxes) |
| corrosion | 2,000 | Roboflow Corrosion (real boxes) |
| potholes | 2,000 | road-damage + Roboflow pothole sets (real boxes) |
| paint_degradation | 2,000 | Roboflow paint-degradation (real boxes) |

Built by `scripts/01_reorganize_data.py`: collect → perceptual a-hash dedup (2,023 dupes
removed) → balance to 2,000 boxes/class → stratified, leak-free 70/15/15 split.

---

## Training Options

```bash
# Train all 3 models sequentially (local)
python scripts/train_all.py

# Train a specific model only
python scripts/train_all.py --model yolo11n
python scripts/train_all.py --model yolov8n
python scripts/train_all.py --model ssdlite
```

Each run auto-increments a version: `runs/yolo11n/v1/`, `v2/`, etc. Results are appended to
`runs/run_log.csv` — no previous runs are overwritten. Ultralytics prints a live progress
bar to the terminal during training.

---

## Run the Dashboard

```bash
streamlit run ui/app.py
```

Upload an image → see bounding boxes, class labels, confidence scores, and severity estimates.
The sidebar lets you pick which trained model version to test on.

---

## Quick Sanity Check

Before a long training run, verify the pipeline end-to-end in ~10 seconds:

```bash
python scripts/smoke_test.py
```

---

## Results

Final models are trained on Colab on the balanced 6,422-image dataset; run
`notebooks/colab_train_evaluate.ipynb` (or `notebooks/03_evaluation.ipynb` locally) to
regenerate `runs/model_comparison.csv` and the per-class / severity charts.

---

## Dependencies

```bash
# NVIDIA GPU (recommended)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# CPU only
pip install -r requirements.txt
```
