# CLAUDE.md

Always ask me before any push and commit to github.

Guidance for Claude Code when working in this repository.

---

## Assignment

**COS40007 — Design Project: AI-Based Structural Defect Detection**
**Submission:** Week 13 — Friday 13 June 2026, 11:59 PM | **Interview:** Week 14
**Group:** 4–5 students | **Total:** 50 marks

Goal: Train and compare object detection models (bounding boxes) to detect structural defects in infrastructure images/video. Must be **object detection**, not image classification.

---

## Marking Rubric (50 marks total — always check work against this)

| Task | Marks | Level 5 requirement |
|------|-------|---------------------|
| 1. Data Collection, Labelling & Preprocessing | 10 | Large, diverse dataset; accurate labels; clear class definitions; **strong preprocessing & augmentation with written justification** |
| 2. Training & Validation | 5 | Multiple models well-trained; **hyperparameters justified**; proper validation strategy |
| 3. Detection on Unseen Data (Generalisation) | 5 | Strong generalisation; handles variation; **clear justification of performance & limitations** |
| 4. Evaluation Metrics & Discussion | 10 | Uses mAP, IoU, Precision/Recall; **deep analysis**; strong justification of chosen methods |
| 5. User Interface / System Implementation | 5 | Functional, intuitive; well-integrated; demonstrates engineering design |
| 6. Interview (Individual) | 15 | Deep understanding across dataset, model, evaluation; explains trade-offs & limitations |

**Key interview questions from the brief:**
1. How was your dataset collected? How did you ensure variation in surface types, lighting, scale, viewpoints?
2. How did you configure training (epochs, LR, image size)? What was the impact of these parameters?
3. How reliable is your model for engineering decision-making? What are its limitations?

---

## Defect Classes (5)

| ID | Class | Sources |
|----|-------|---------|
| 0 | `cracks` | wall-crack-hole-normal (crack class) + road-damage (crack class) + concrete major_crack + concrete minor_crack |
| 1 | `spalling` | concrete-structural-defect / spalling folder |
| 2 | `corrosion` | concrete-structural-defect / stain folder (visual proxy — surface staining as precursor) |
| 3 | `potholes` | road-damage-potholes-cracks (pothole class) |
| 4 | `paint_degradation` | concrete-structural-defect / peeling folder |

Minimum 200 labelled images per class (1,000+ total). Multi-class detection in a single image is required.

---

## Dataset Statistics (current, after reorganisation)

| Split | Images | Labels |
|-------|--------|--------|
| train | 3,237 | 3,237 |
| val | 694 | 694 |
| test | 695 | 695 |
| **Total** | **4,626** | **4,626** |

**Bounding box distribution (train set):**

| Class | BBoxes | Notes |
|-------|--------|-------|
| cracks | 3,439 | Real per-object bboxes from wall-crack + road-damage datasets |
| potholes | 883 | Real per-object bboxes from road-damage dataset |
| paint_degradation | 304 | Whole-image bbox (classification-only source) |
| corrosion | 267 | Whole-image bbox (classification-only source) |
| spalling | 236 | Whole-image bbox (classification-only source) |

**Class imbalance — known limitation:**
cracks (3,439) dominates over spalling (236), corrosion (267), paint_degradation (304). YOLO uses class-weighted loss which partially mitigates this. Minority classes will have lower per-class AP — acknowledge in report. The concrete-structural-defect source contributes whole-image bboxes AND is the main source for 3 of the 4 minority classes.

**Note on whole-image bboxes:** concrete-structural-defect has no annotation files — each image was labelled with a single full-image bounding box (cx=0.5, cy=0.5, w=1.0, h=1.0). This is a valid technique for converting classification datasets to detection when no localization ground truth exists. Acknowledge this limitation in the report.

---

## Source Dataset → Class Mapping

### wall-crack-hole-normal
- Original classes: `normal(0)`, `crack(1)`, `hole(2)`
- **Keep:** images with at least one `crack(1)` bbox → remap to `cracks(0)`
- **Discard:** images with only `normal` or `hole` annotations (not relevant to our 5 classes)
- Image formats: `.jpg` and `.png` (both supported — this was a critical bug fixed 27 May 2026)

### road-damage-potholes-cracks
- Original classes: `Pothole(0)`, `Crack(1)`, `Manhole(2)`
- **Keep:** `Pothole→potholes(3)`, `Crack→cracks(0)`
- **Discard:** manhole-only images

### concrete-structural-defect
- Classification-only dataset (no annotation files)
- **Used folders:** `spalling(1)`, `stain→corrosion(2)`, `peeling→paint_degradation(4)`, `major_crack→cracks(0)`, `minor_crack→cracks(0)`
- **Deleted folders:** `algae/`, `normal/`, `Wall/` (not relevant to our 5 classes)
- Whole-image bounding boxes used as proxy annotations

---

## Run Versioning

Every time `python scripts/train_all.py` is run, each model auto-increments its version:
- YOLO models save to `runs/{model}/v{N}/` (Ultralytics folder structure)
- PyTorch models save to `runs/{model}/best_v{N}.pth`
- All runs are appended to `runs/run_log.csv` with `run_id = {model}_v{N}_{timestamp}`

**Current runs (all complete):**
| Model | Version | Epochs | Best mAP@0.5 | mAP@0.5:0.95 | Weights | Status |
|-------|---------|--------|-------------|--------------|---------|--------|
| YOLOv11n | v1 | 100 | 0.7653 (ep95) | 0.6615 | runs/yolo11n/v1/weights/best.pt | Done |
| YOLOv8n | v1 | 100 | 0.7636 (ep90) | 0.6669 | runs/yolov8n/v1/weights/best.pt | Done |
| SSDLite | v1 | 30 | 0.2387 | 0.1888 | runs/ssdlite/best_v1.pth | Done |

**SSDLite note:** Low mAP (0.24) is expected — anchor-based models need careful anchor tuning for defect scales; whole-image bboxes for 3/5 classes hurt anchor matching. Valid comparison point: demonstrates why anchor-free YOLO outperforms SSDLite on this dataset.

**Abandoned partial runs (do not use):**
- `runs/yolo11n/v2/` — 17 epochs only, aborted
- `runs/yolov8n/v2/` — 13 epochs only, aborted

**Note:** YOLOv8n AMP was auto-disabled by Ultralytics on GTX 1650 (NaN loss risk). FP32 training — results are still valid.

**Weight selection:** `ui/inference.py` and `notebooks/03_evaluation.ipynb` both use `_best_yolo_weights()` — picks the vN folder with the highest mAP@0.5 in results.csv, not the highest version number.

---

## Adding Future Datasets

1. Add a new `process_my_dataset()` function in `scripts/01_reorganize_data.py` following the existing pattern. It must return `list[tuple[Path, list[str]]]` (image path, YOLO label lines).
2. Add a `summary.append(...)` call in `main()`.
3. Re-run `python scripts/01_reorganize_data.py` — it clears and rebuilds `data/` from scratch.
4. Check class distribution: ensure all 5 classes still have ≥200 bboxes in train.

Supported image extensions: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp` (added 27 May 2026 after discovering wall-crack contained `.png` files that were missed).

---

## Models to Compare (3 required)

| Model | Architecture | Pre-trained weights | Purpose |
|-------|-------------|---------------------|---------|
| YOLOv11n | One-stage, anchor-free | COCO/ImageNet | Latest YOLO — speed + accuracy benchmark |
| YOLOv8n | One-stage, anchor-free | COCO/ImageNet | Established YOLO baseline — measure v8→v11 improvement |
| SSDLite320-MobileNetV3 | One-stage, anchor-based | ImageNet backbone | Google edge/mobile standard — real-world deployability |

All use **transfer learning** from pre-trained weights.

**Comparison narrative:**
- YOLOv11n vs YOLOv8n → same architecture family, different generations: quantifies how much YOLO improved
- Both YOLO vs SSDLite → fundamentally different design: anchor-free vs anchor-based, speed vs mobile efficiency
- Interview answer for "why these models": YOLO-family evolution + edge-deployment contrast (SSD is the industry standard for embedded/drone inspection systems)

---

## Severity Estimation (required)

```python
severity = (bbox_width * bbox_height) / (img_width * img_height) * 100
# Low: <5%  |  Medium: 5–20%  |  High: >20%
```

---

## GPU Configuration — RTX 1650 (4 GB VRAM)

| Model | batch | img_size | AMP | Notes |
|-------|-------|----------|-----|-------|
| YOLOv11n | 8 | 640 | ✓ | `amp=True` in Ultralytics; workers=0 on Windows |
| YOLOv8n | 8 | 640 | ✓ | `amp=True` in Ultralytics; workers=0 on Windows |
| SSDLite320-MobileNetV3 | 2 | 640 | ✓ | Model internally resizes to 320×320; 2.3M params — very fast |

- Always call `torch.cuda.empty_cache()` between model evaluations
- If OOM occurs: reduce batch to 1 or image size to 416
- Two-phase training for SSDLite (freeze backbone epochs 1–5, unfreeze epoch 6+)

---

## Tech Stack

- **Training:** Python, PyTorch, Ultralytics YOLO, torchvision
- **Annotation format:** YOLO `.txt`
- **Evaluation:** mAP@0.5, mAP@0.5:0.95, IoU, Precision-Recall curves (torchmetrics)
- **UI/Demonstrator:** Streamlit — image/video upload → bounding box output with severity
- **Experiment tracking:** `runs/run_log.csv`

---

## Directory Structure

```
ai_for_engineering/
├── data/                        ← unified training data (generated by scripts/01)
│   ├── images/{train,val,test}/ ← all classes mixed (correct for YOLO)
│   ├── labels/{train,val,test}/ ← YOLO .txt annotations
│   ├── augmented/               ← output of scripts/02_augmentation.py
│   ├── data.yaml                ← YOLO dataset config
│   └── manifest.csv             ← per-source image counts
├── dataset/                     ← original source datasets (reference only)
│   ├── wall-crack-hole-normal/
│   ├── road-damage-potholes-cracks/
│   └── concrete-structural-defect/
├── models/                      ← shared architecture definitions
│   ├── resnet_detector.py       ← (kept for reference, not in main comparison)
│   ├── efficientnet_detector.py ← (kept for reference, not in main comparison)
│   └── ssdlite_detector.py      ← SSDLite320-MobileNetV3 (active)
├── notebooks/
│   ├── 01_data_preprocessing.ipynb  ← EDA, class balance, augmentation demo
│   ├── 02_training.ipynb            ← train all 3 models
│   └── 03_evaluation.ipynb          ← mAP, PR curves, comparison table
├── ui/
│   ├── app.py                   ← Streamlit dashboard (run: streamlit run ui/app.py)
│   └── inference.py             ← model loading and inference helpers
├── scripts/
│   ├── 01_reorganize_data.py    ← maps dataset/ → data/ (re-runnable, idempotent)
│   └── 02_augmentation.py       ← Albumentations pipeline
├── runs/                        ← training outputs, weights, logs
│   ├── yolo11n/v1/, v2/, ...    ← YOLOv11n runs (auto-versioned)
│   ├── yolov8n/v1/, v2/, ...    ← YOLOv8n runs (auto-versioned)
│   ├── ssdlite/best_v1.pth, ... ← SSDLite checkpoints (auto-versioned)
│   └── run_log.csv              ← all runs logged here
└── report/
```

**Why flat-by-split (not by-category)?**
Object detection datasets use flat split folders because class identity is encoded in the label files (the `class_id` on each line), not in the directory structure. Per-category folders are an image-classification convention. YOLO and Faster R-CNN both expect `images/train/` → mixed classes.

---

## How to Run (Step by Step)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Data (already done — skip if data/ is populated)
python scripts/01_reorganize_data.py

# 3. (Optional) Augment training set
python scripts/02_augmentation.py --n 2

# 4. EDA and preprocessing
jupyter notebook notebooks/01_data_preprocessing.ipynb

# 5. Train all 3 models (~2-4 hours on RTX 1650)
jupyter notebook notebooks/02_training.ipynb

# 6. Evaluate and compare
jupyter notebook notebooks/03_evaluation.ipynb

# 7. Run Streamlit dashboard
streamlit run ui/app.py
```

---

## Evaluation Results (Test Set — 695 images, run 28 May 2026)

### Model Comparison Table

| Model | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | Speed (ms/img) |
|-------|---------|-------------|-----------|--------|---------------|
| YOLOv11n | **0.7557** | **0.6719** | 0.7507 | 0.7041 | 6.4 |
| YOLOv8n | 0.7477 | 0.6592 | 0.7792 | 0.6912 | 5.6 |
| SSDLite | 0.0530 | 0.0434 | — | — | 20.4 |

Saved: `runs/model_comparison.csv`, `runs/model_comparison.png`

### Per-Class AP@0.5 (YOLOv11n — best model)

| Class | Images | Instances | Precision | Recall | AP@0.5 | AP@0.5:0.95 |
|-------|--------|-----------|-----------|--------|--------|-------------|
| cracks | 492 | 753 | 0.643 | 0.404 | 0.514 | 0.324 |
| spalling | 57 | 57 | 0.865 | 0.912 | 0.950 | 0.950 |
| corrosion | 53 | 53 | 0.834 | 0.906 | 0.959 | 0.959 |
| potholes | 121 | 201 | 0.541 | 0.299 | 0.376 | 0.148 |
| paint_degradation | 58 | 58 | 0.871 | 1.000 | 0.979 | 0.979 |

Saved: `runs/per_class_ap_yolov11n.png`, `runs/per_class_ap_yolov8n.png`

### Per-Class AP@0.5 (YOLOv8n)

| Class | Precision | Recall | AP@0.5 | AP@0.5:0.95 |
|-------|-----------|--------|--------|-------------|
| cracks | 0.646 | 0.405 | 0.505 | 0.319 |
| spalling | 0.887 | 0.860 | 0.937 | 0.927 |
| corrosion | 0.962 | 0.868 | 0.929 | 0.927 |
| potholes | 0.558 | 0.323 | 0.396 | 0.151 |
| paint_degradation | 0.843 | 1.000 | 0.971 | 0.971 |

### Severity Distribution (test set ground truth)
- Low (<5% area): 587 instances
- Medium (5–20%): 160 instances
- High (>20%): 375 instances

Saved: `runs/severity_distribution.png`, `runs/prediction_samples.png`

### Key Findings for Report

**Why cracks AP is lowest (0.51) despite most training samples:**
Cracks have tight, irregular bboxes varying wildly in size/orientation. Harder IoU matching than the whole-image bboxes used for spalling/corrosion/paint_degradation.

**Why pothole AP is low (0.38):**
Road images have high viewpoint/scale variation. Only 883 train bboxes — insufficient relative to class diversity.

**Why paint_degradation, spalling, corrosion have near-perfect AP (0.95–0.98):**
These use whole-image bboxes (cx=0.5, cy=0.5, w=1.0, h=1.0). IoU is trivially high because the ground truth covers the full image — inflated metric, not true localisation accuracy. Acknowledge in report.

**SSDLite test mAP (0.053) vs training mAP (0.209):**
Training mAP was computed on val set at epoch 24. The lower test mAP may reflect: (a) anchor mismatch — SSDLite anchors designed for COCO objects, not defect scales; (b) whole-image bboxes hurt anchor matching at inference; (c) the model genuinely overfitted to val. Report as expected underperformance of anchor-based detection on this dataset.

**YOLOv11n slightly beats YOLOv8n** (+0.8% mAP@0.5, +1.3% mAP@0.5:0.95) — demonstrates one generation of YOLO improvement.

---

## Deliverables Checklist (vs Rubric)

### Task 1: Data Collection, Labelling & Preprocessing (10 marks)
- [x] Labelled dataset — YOLO `.txt` format (4,626 images, 5 classes)
- [x] Class distribution chart (`runs/class_distribution.png`)
- [x] Augmentation script with justification (`scripts/02_augmentation.py`)
- [ ] Written justification of augmentation choices in report
- [ ] Acknowledge whole-image bbox limitation for concrete-structural data

### Task 2: Training & Validation (5 marks)
- [x] 3 model architectures: YOLOv11n (mAP@0.5=0.756), YOLOv8n (0.748), SSDLite (0.053 test / 0.209 val)
- [x] Transfer learning from pre-trained weights (COCO for YOLO, ImageNet for SSDLite backbone)
- [x] Two-phase training for SSDLite (backbone freeze → unfreeze)
- [x] Auto-versioned training runs in runs/
- [x] Training logs in `runs/run_log.csv`
- [ ] Written justification of hyperparameter choices in report

### Task 3: Detection on Unseen Data (5 marks)
- [x] Test split kept separate (695 images, never seen during training)
- [x] Test results in `notebooks/03_evaluation.ipynb`
- [ ] Written analysis of generalisation performance

### Task 4: Evaluation Metrics & Discussion (10 marks)
- [x] mAP@0.5 and mAP@0.5:0.95 (all 3 models)
- [x] Per-class AP (all 5 classes for YOLO models)
- [x] Model comparison table (`runs/model_comparison.csv`)
- [x] Severity distribution chart (`runs/severity_distribution.png`)
- [x] Prediction sample visualisations (`runs/prediction_samples.png`)
- [ ] IoU histogram (not yet in evaluation notebook)
- [ ] FP/FN error analysis (not yet in evaluation notebook)

### Task 5: User Interface (5 marks)
- [x] Streamlit dashboard with image upload
- [x] Bounding box overlay with labels, confidence, severity
- [x] Model comparison tab
- [ ] Video inference support (stretch goal)

### Task 6: Interview (15 marks)
- Ensure every team member can explain: dataset mapping decisions, class imbalance handling, training configuration, metric interpretation, severity formula, model trade-offs

---

## Key Deadlines

| Milestone | Date |
|-----------|------|
| Tutor consultation (MANDATORY — 20% penalty if missed) | Before Week 11 |
| Phase 1: Dataset + pipeline ready | **29 May 2026** |
| Phase 2: Models trained + UI built | 5 June 2026 |
| Phase 3: Unseen testing + report | 6–11 June 2026 |
| Final submission | **13 June 2026, 11:59 PM** |
| Interview | Week 14 |

---

## Shared Dataset

Google Drive: `https://drive.google.com/drive/folders/11RjURZYUAUUG_TKsp1b7S0ibegOBsap`

## Member Responsibilities

| Member | Focus |
|--------|-------|
| 1 — Data Engineering | Preprocessing pipeline, augmentation (lighting/noise/scale), class balance analysis |
| 2 — Training Lead | YOLO/DL framework setup, hyperparameter tuning (lr, batch, epochs), training logs |
| 3 — Evaluation | mAP/IoU/PR curves, unseen data generalisation test, error analysis (FP/FN) |
| 4 — Deployment | Streamlit/web UI, inference pipeline, severity estimation integration |
| 5 — MLOps & Report | Centralised run comparison table, final report consolidation |
