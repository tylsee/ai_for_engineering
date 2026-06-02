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
| 0 | `cracks` | wall-crack-hole-normal (crack class) + road-damage (crack class) |
| 1 | `spalling` | Roboflow: spalling/ + spalling2/ (real localized bboxes) |
| 2 | `corrosion` | Roboflow: Corrosion/ (real localized bboxes) |
| 3 | `potholes` | road-damage-potholes-cracks + Roboflow: potholesv1/V2/V3/V4 (real localized bboxes) |
| 4 | `paint_degradation` | Roboflow: paint-degradation/ (real localized bboxes) |

Minimum 200 labelled images per class (1,000+ total). Multi-class detection in a single image is required.

**Note:** concrete-structural-defect (whole-image proxy bboxes) is RETIRED — real localized Roboflow annotations now exist for all 5 classes. The `--keep-proxies` flag in `01_reorganize_data.py` re-enables it for comparison only.

---

## Dataset Statistics (current — rebuilt 3 June 2026, spalling3 added)

| Split | Images | Labels |
|-------|--------|--------|
| train | 4,493 | 4,493 |
| val | 964 | 964 |
| test | 965 | 965 |
| **Total** | **6,422** | **6,422** |

**Bounding box distribution — PERFECTLY BALANCED (all classes capped at 2000):**

| Class | train | val | test | TOTAL | Source |
|-------|-------|-----|------|-------|--------|
| cracks | 1,381 | 307 | 312 | 2,000 | wall-crack + road-damage (real bboxes) |
| spalling | 1,426 | 282 | 292 | 2,000 | Roboflow: spalling/ + spalling2/ + **spalling3/** (real bboxes) |
| corrosion | 1,400 | 300 | 300 | 2,000 | Roboflow: Corrosion/ (real bboxes) |
| potholes | 1,404 | 319 | 277 | 2,000 | road-damage + Roboflow pothole folders (real bboxes) |
| paint_degradation | 1,383 | 301 | 316 | 2,000 | Roboflow: paint-degradation/ (real bboxes) |
| **TOTAL** | | | | **10,000** | |

**Class balance: 1.00x ratio** (was 1.82x before spalling3, originally 14x). Every class hits the TARGET_PER_CLASS=2000 bbox cap. `spalling3` (Roboflow yolo-xxm9l/spalling-wcoze, 1011 imgs) was added 3 Jun 2026 specifically to lift spalling from its earlier 782-box floor to the full 2000.

**Deduplication:** 2,023 duplicate/unreadable images removed using perceptual a-hash (8×8 exact match) before splitting (13,929 collected → 11,906 unique). The four pothole Roboflow folders had heavy re-upload duplication. Without dedup, identical images leak across train/test → inflated mAP. Split verified: zero a-hash signatures span multiple splits.

**PotholesV2 is a chess dataset** (nc=13 chess piece classes). Pothole is class index 6 — only `{6: 3}` remap kept.

---

## Source Dataset → Class Mapping

### wall-crack-hole-normal
- Original classes: `normal(0)`, `crack(1)`, `hole(2)`
- **Keep:** images with at least one `crack(1)` bbox → remap to `cracks(0)`
- **Discard:** images with only `normal` or `hole` annotations
- Image formats: `.jpg` and `.png` (both supported)

### road-damage-potholes-cracks
- Original classes: `Pothole(0)`, `Crack(1)`, `Manhole(2)`
- **Keep:** `Pothole→potholes(3)`, `Crack→cracks(0)`
- **Discard:** manhole-only images

### Roboflow: paint-degradation/
- Roboflow export format (train/valid/test splits with YOLO .txt labels)
- Original class 0 → `paint_degradation(4)`

### Roboflow: Corrosion/
- Roboflow export format
- Original class 0 → `corrosion(2)`

### Roboflow: spalling/, spalling2/ and spalling3/
- All three folders processed together via `process_spalling()`
- Original class 0 → `spalling(1)`
- spalling3 (yolo-xxm9l/spalling-wcoze, `nc:1 names:['Spalling']`, 1011 train imgs) added 3 Jun 2026 to reach the 2000-box target

### Roboflow: potholesv1/, PotholesV2/, potholesV3/, PotholesV4/
- All processed via `process_potholes_roboflow()`
- v1/V3/V4: class 0 → `potholes(3)`
- PotholesV2: **chess dataset** (nc=13) — class 6 → `potholes(3)`, all other classes discarded
- 1,988 duplicates removed by a-hash: v1/V3/V4 are the same images re-uploaded

### concrete-structural-defect (RETIRED as primary source)
- Was: classification-only, whole-image proxy bboxes (cx=0.5, cy=0.5, w=1.0, h=1.0)
- **No longer used** — real localized Roboflow annotations replace it for all affected classes
- Function kept in `01_reorganize_data.py`; re-enable with `--keep-proxies` flag for ablation

---

## Run Versioning

Every time `python scripts/train_all.py` is run, each model auto-increments its version:
- YOLO models save to `runs/{model}/v{N}/` (Ultralytics folder structure)
- PyTorch models save to `runs/{model}/best_v{N}.pth`
- All runs are appended to `runs/run_log.csv` with `run_id = {model}_v{N}_{timestamp}`

**Local `runs/` = OLD-data BASELINE only, kept intentionally for the report's before/after story.**

The local GTX 1650 is too weak; final training moved to Google Colab (see "Training on Google
Colab"). After cleanup (3 Jun 2026) only the two complete old-data runs remain — deliberately
kept as the "unbalanced / proxy-bbox" baseline to contrast against the final balanced results:

| Folder | Epochs | Trained on | Role |
|--------|--------|-----------|------|
| runs/yolo11n/v1 | 100 | OLD: 3,237 imgs, proxy bboxes, ~14x imbalance | report baseline ("before") |
| runs/yolov8n/v1 | 100 | OLD: 3,237 imgs, proxy bboxes, ~14x imbalance | report baseline ("before") |

Old-data baseline numbers are recorded under "Evaluation Results → OLD dataset results" below.

Removed in cleanup (aborted runs / cruft): `yolo11n/v2` (17ep), `yolo11n/v3` (8ep killed),
`yolov8n/v2` (17ep), a stray `runs/yolo/ssdlite/best_v1.pth` (path-bug remnant), and
`runs/train_newdata.log`.

**Canonical final runs come from Colab** (`colab_train_evaluate.ipynb`), trained on the
6,422-img / 10,000-box perfectly balanced set. On Colab `runs/` starts empty, so it produces
fresh `yolo11n/v1`, `yolov8n/v1`, `ssdlite/best_v1.pth`, zipped back as `runs_results.zip`.

**Importing Colab results without clobbering the baseline:** the Colab `v1` folders will collide
with the local old-data `v1`. Drop the Colab runs in as `v2` (rename on import) so both coexist
and are selectable in the UI version dropdown. Caveat: `_best_yolo_weights()` picks by mAP@0.5,
and the proxy-bbox baseline has inflated mAP — so "Best (auto)" may favour the OLD run; pick the
Colab version explicitly in the dropdown.

**Note:** AMP auto-disabled by Ultralytics on GTX 1650 (NaN loss). On Colab AMP is ON (stable).

**Weight selection:** `ui/inference.py` and `notebooks/03_evaluation.ipynb` both use `_best_yolo_weights()` — picks the vN folder with the highest mAP@0.5 in results.csv, not the highest version number. Safe in presence of partial/aborted runs.

**UI version selector:** Sidebar dropdown in `ui/app.py` lets you pick "Best (auto)" or any specific vN run by mAP. Old weights remain accessible.

---

## Adding Future Datasets

1. Add a new `process_my_dataset()` function in `scripts/01_reorganize_data.py`. Each item must be an `Item` dataclass with `img`, `lines`, `source`, `classes` (frozenset of int class IDs), and `sig` (perceptual a-hash int or None).
2. Add it to `collect_items()` in the correct order.
3. Re-run `python scripts/01_reorganize_data.py` — clears and rebuilds `data/` from scratch. Global dedup + balance + split run automatically.
4. Check class distribution: ensure all 5 classes still have ≥200 bboxes in train.

Supported image extensions: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`

**Pipeline flags:**
- `--keep-proxies` — re-includes concrete-structural-defect whole-image bboxes (ablation only)
- Default: real localized bboxes only

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
├── CLAUDE.md
├── README.md
├── requirements.txt
├── COS40007+Design+project+2026_Topics_v2.pdf
├── data/                        ← unified training data (generated by scripts/01)
│   ├── images/{train,val,test}/ ← all classes mixed (correct for YOLO)
│   ├── labels/{train,val,test}/ ← YOLO .txt annotations
│   ├── data.yaml                ← YOLO dataset config (regenerated at runtime)
│   └── manifest.csv             ← per-source image counts
├── dataset/                     ← original source datasets (reference only)
│   ├── wall-crack-hole-normal/
│   ├── road-damage-potholes-cracks/
│   ├── concrete-structural-defect/  ← retired as primary source
│   ├── paint-degradation/       ← Roboflow (new)
│   ├── Corrosion/               ← Roboflow (new)
│   ├── spalling/                ← Roboflow (new)
│   ├── spalling2/               ← Roboflow (new)
│   ├── spalling3/               ← Roboflow (added 3 Jun — balances spalling to 2000)
│   ├── potholesv1/              ← Roboflow (new)
│   ├── PotholesV2/              ← Roboflow chess dataset — class 6 = pothole
│   ├── potholesV3/              ← Roboflow (same images as v1, deduped)
│   └── PotholesV4/              ← Roboflow (same images as v1, deduped)
├── models/
│   └── ssdlite_detector.py      ← SSDLite320-MobileNetV3 (active)
├── notebooks/
│   ├── 01_data_preprocessing.ipynb  ← EDA, class balance, augmentation demo (local)
│   ├── 02_training.ipynb            ← train all 3 models (local)
│   ├── 03_evaluation.ipynb          ← mAP, PR curves, comparison table (local)
│   └── colab_train_evaluate.ipynb   ← ALL-IN-ONE for Google Colab (setup+EDA+train+eval+save)
├── ui/
│   ├── app.py                   ← Streamlit dashboard (run: streamlit run ui/app.py)
│   └── inference.py             ← model loading, version listing, inference helpers
├── scripts/
│   ├── 01_reorganize_data.py    ← COLLECT→DEDUP→BALANCE→SPLIT→SAVE pipeline
│   ├── 02_augmentation.py       ← Albumentations pipeline
│   ├── download_datasets.py     ← dataset downloader
│   ├── smoke_test.py            ← fast training smoke test
│   └── train_all.py             ← queue all 3 models sequentially (local)
├── runs/                        ← training outputs, weights, logs
│   ├── yolo11n/v1/             ← OLD-data baseline, kept for report before/after
│   ├── yolov8n/v1/             ← OLD-data baseline, kept for report before/after
│   ├── run_log.csv              ← all runs logged here
│   └── class_distribution.png  ← balanced dataset (2000/class, stacked by split)
│   (final weights arrive from Colab as runs_results.zip → unzip here)
└── weights/
    ├── yolo11n.pt               ← COCO pretrained YOLOv11n (transfer learning base)
    └── yolov8n.pt               ← COCO pretrained YOLOv8n (transfer learning base)
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
# Add --keep-proxies to include concrete-structural-defect proxy bboxes (ablation only)

# 3. (Optional) Augment training set
python scripts/02_augmentation.py --n 2

# 4. EDA and preprocessing
jupyter notebook notebooks/01_data_preprocessing.ipynb

# 5. Train all 3 models sequentially (~overnight on GTX 1650 — or use Colab, see below)
python scripts/train_all.py     # Ultralytics prints a live progress bar to the terminal

# 6. Evaluate and compare (run after training completes)
jupyter notebook notebooks/03_evaluation.ipynb

# 7. Run Streamlit dashboard
streamlit run ui/app.py
```

---

## Training on Google Colab (recommended — local GPU is too weak)

The all-in-one notebook `notebooks/colab_train_evaluate.ipynb` does setup → EDA →
train all 3 models → evaluate → save results, in one "Run all". It is self-contained
(SSDLite model definition is inlined), so the **only** thing it needs is the data zip.

**Workflow:**
1. Rebuild + zip the data locally (zip excludes `*.cache` and `augmented/`):
   ```bash
   python scripts/01_reorganize_data.py        # rebuild data/ (if changed)
   # defect_dataset.zip is regenerated by the same one-liner used in chat,
   # or zip data/ manually — the archive root must contain data/
   ```
   This produces `defect_dataset.zip` (~290 MB) in the project root (gitignored).
2. Upload `defect_dataset.zip` to Google Drive at `MyDrive/COS40007/defect_dataset.zip`
   (drag-and-drop in the Drive web page — reliable for large files).
3. Open `colab_train_evaluate.ipynb` in Colab → **Runtime → Change runtime type → T4 GPU**.
4. **Runtime → Run all.**

**Key Colab design points:**
- The notebook copies the zip from mounted Drive to Colab **local disk** before unzipping —
  training directly off mounted Drive is very slow (per-file latency).
- Hyperparameters auto-scale: on a ≥12 GB GPU (Colab T4/L4) it uses `YOLO batch=16`,
  `SSDLite batch=8`, `workers=2`; on a small/Windows GPU it falls back to `batch=8/2`, `workers=0`.
- AMP is enabled on Colab (stable there, unlike the GTX 1650 which forces FP32).
- `data.yaml` is rewritten at runtime with the correct absolute path (the path baked into the zip is ignored).
- **Part 4 zips `runs/` back to `MyDrive/COS40007/runs_results.zip`** — Colab is ephemeral, so
  download it and unzip into local `runs/` to use the new weights in the Streamlit UI.
- Edit `DRIVE_ZIP_PATH` in cell 0.2 if you store the zip elsewhere.
- The same notebook also runs locally (detects non-Colab, skips mount/unzip, uses existing `data/`).

---

## Evaluation Results

### OLD dataset results — BASELINE for the report's "before" (unbalanced + proxy bboxes)

> Kept intentionally as the "before" baseline. These v1 runs were trained on the OLD dataset
> (3,237 imgs, ~14x class imbalance, whole-image proxy bboxes for spalling/corrosion/paint).
> The near-perfect AP on those 3 classes is artificially inflated by full-image IoU — precisely
> the weakness the rebuild fixes. The before/after contrast is the report narrative.
> Weights kept: `runs/yolo11n/v1`, `runs/yolov8n/v1` (SSDLite old weights removed; number below kept).

| Model | mAP@0.5 | mAP@0.5:0.95 | Notes |
|-------|---------|-------------|-------|
| YOLOv11n v1 | 0.7557 | 0.6719 | old data, proxy bboxes (inflated) |
| YOLOv8n v1 | 0.7477 | 0.6592 | old data, proxy bboxes (inflated) |
| SSDLite v1 | 0.0530 | 0.0434 | old data, proxy bboxes |

### NEW dataset results (pending — re-run notebooks/03_evaluation.ipynb after v3 training)

Run `notebooks/03_evaluation.ipynb` after training completes to regenerate:
- `runs/model_comparison.csv` + `runs/model_comparison.png`
- `runs/per_class_ap_yolov11n.png`, `runs/per_class_ap_yolov8n.png`
- `runs/severity_distribution.png`, `runs/prediction_samples.png`

### Key Findings for Report (update after new eval)

**Why cracks AP was lowest (old data):**
Cracks have tight, irregular bboxes varying wildly in size/orientation. Harder IoU matching. Expect improvement with new balanced dataset (1,401 real bboxes vs 3,439 before, but all real).

**Why pothole AP was low (old data):**
Road images have high viewpoint/scale variation. Now ~1,408 bboxes from multiple Roboflow sources — expect improvement.

**Why paint_degradation/spalling/corrosion had near-perfect AP (old data):**
Whole-image proxy bboxes (IoU trivially high) — inflated metric. New dataset uses real localized bboxes, so expect more realistic (lower) AP for these classes. This is a story worth telling in the report: before vs after the proxy-bbox retirement.

**SSDLite expected to underperform YOLO:**
Anchor-based design mismatches defect scales; COCO anchors not tuned for narrow cracks or small potholes. Two-phase training (freeze 5 epochs, unfreeze) helps but anchor mismatch is fundamental.

---

## Deliverables Checklist (vs Rubric)

### Task 1: Data Collection, Labelling & Preprocessing (10 marks)
- [x] Labelled dataset — YOLO `.txt` format (6,422 images, 10,000 boxes, 5 classes)
- [x] Real localized bboxes for all 5 classes (retired whole-image proxy approach)
- [x] Global a-hash deduplication (2,023 duplicates removed — methodologically sound)
- [x] **Perfect class balance: 1.00x ratio** (2000 boxes/class; was 14x) via capped bbox balancing + spalling3
- [x] Class distribution chart (`runs/class_distribution.png`)
- [x] Augmentation script with justification (`scripts/02_augmentation.py`)
- [ ] Written justification of augmentation choices in report
- [ ] Document dedup + proxy-bbox retirement methodology in report (good story)

### Task 2: Training & Validation (5 marks)
- [x] 3 model architectures: YOLOv11n, YOLOv8n, SSDLite (v3 training in progress on new data)
- [x] Transfer learning from pre-trained weights (COCO for YOLO, ImageNet for SSDLite backbone)
- [x] Two-phase training for SSDLite (backbone freeze → unfreeze)
- [x] Auto-versioned training runs in runs/
- [x] Training logs in `runs/run_log.csv`
- [x] UI version selector — compare any trained run side-by-side
- [ ] Written justification of hyperparameter choices in report
- [ ] Update run table in CLAUDE.md with v3 results after training

### Task 3: Detection on Unseen Data (5 marks)
- [x] Test split kept separate (881 images, never seen during training, verified leak-free)
- [ ] Re-run `notebooks/03_evaluation.ipynb` after v3 training completes
- [ ] Written analysis of generalisation performance

### Task 4: Evaluation Metrics & Discussion (10 marks)
- [ ] Re-run `notebooks/03_evaluation.ipynb` to get fresh metrics on new test set
- [ ] mAP@0.5 and mAP@0.5:0.95 (all 3 models on new data)
- [ ] Per-class AP (all 5 classes for YOLO models)
- [ ] Model comparison table (`runs/model_comparison.csv`)
- [ ] Severity distribution chart (`runs/severity_distribution.png`)
- [ ] Prediction sample visualisations (`runs/prediction_samples.png`)
- [ ] IoU histogram
- [ ] FP/FN error analysis

### Task 5: User Interface (5 marks)
- [x] Streamlit dashboard with image upload
- [x] Bounding box overlay with labels, confidence, severity
- [x] Model version selector dropdown (pick any vN run or auto-best)
- [x] Model comparison tab (shows run_log + model_comparison.csv)
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
