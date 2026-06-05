# COS40007 — AI-Based Structural Defect Detection

Detect and localise structural defects in infrastructure images using **object detection** (bounding
boxes, not image classification).

**5 defect classes:** `cracks` · `spalling` · `corrosion` · `potholes` · `paint_degradation`

**Models compared (small models only):**
- **YOLOv11s** — one-stage anchor-free CNN (primary)
- **YOLOv8s** — one-stage anchor-free CNN (generation comparison)
- **SSDLite320-MobileNetV3** — anchor-based mobile/edge baseline
- **RT-DETR (rtdetr-l)** — end-to-end transformer (architecture diversity)

All use transfer learning from pre-trained weights. **Medium/large variants (`m`/`l`/`x`) are
deliberately excluded** — the comparison is scoped to small detectors.

---

## Dataset

The final dataset is **v3** (in `data_v3/`). v2 (in `data/`) is kept as a diagnostic baseline for
the report's before/after story. v3 was rebuilt from cleaner sources with corrosion aspect-ratio
filtering and train-only weak-class crop augmentation, then a targeted cleanup of mislabeled and
unlearnable images.

| Stage | Images | Boxes | Notes |
|-------|--------|-------|-------|
| v2 (`data/`, diagnostic) | 4,694 | 9,015 | real localized boxes, 1.01x balance, leak-free |
| v3 (`data_v3/`, final) | ~5,200 | ~13,000 | cleaner sources + crop-aug; train oversampled toward weak classes |

Built by `scripts/01_reorganize_data.py` (collect → perceptual-hash dedup → balance → leak-free split,
cleaning at source). Rebuild v3 with:

```bash
python scripts/01_reorganize_data.py --out data_v3
python scripts/verify_rebuild.py --data data_v3
```

See `CLAUDE.md` (Dataset v3 section) and `docs/dataset_cleaning_report.md` for the full pipeline.

---

## Training workflow (default: `baseline_640`)

The default YOLO workflow is **`baseline_640` only** (640 px, 110 epochs, AdamW, cosine LR).

A 768-px fine-tune stage was implemented and tested on both v2 and v3; it **did not improve
validation mAP** (v3: finetune_768 ≈ 0.44 vs baseline_640 ≈ 0.46), so it is **disabled by default**
(`RUN_FINETUNE_768 = False`) and kept only as an optional ablation. Higher resolution was not the
bottleneck — the remaining limitation is weak-class data quality (cracks, corrosion,
paint_degradation). See `docs/weak_class_next_steps.md`.

Order: build/verify `data_v3` → train **YOLOv11s** `baseline_640` → train **YOLOv8s** `baseline_640`
→ (optional) SSDLite + RT-DETR → score the **test set once** for the final comparison.

### Run it

The three all-in-one notebooks share the same training/eval cells (inlined from `scripts/` by
`scripts/update_training_notebooks.py`); enable one model per session via the `RUN_*` switches in
Part 2.1:

```bash
jupyter notebook notebooks/local_train_evaluate.ipynb     # local (set QUICK_TEST=True for a 3-epoch smoke test)
# notebooks/colab_train_evaluate.ipynb                     # Google Colab T4
# notebooks/kaggle_train_evaluate.ipynb                    # Kaggle T4
```

The local GTX 1650 (4 GB) is only for the data pipeline and quick checks — run real training on a
Kaggle/Colab T4. Upload `defect_dataset_v3.zip` (built by `scripts/zip_data.py --src data_v3`) as the
training data.

**Records (kept separate):** `runs/experiment_tracker.csv` = validation per stage;
`runs/model_comparison.csv` = final test only.

---

## Quick Start (local)

```bash
# 1. (NVIDIA GPU) install PyTorch with CUDA first, then the rest
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# 2. (only if sources changed) rebuild the dataset
python scripts/01_reorganize_data.py --out data_v3

# 3. quick pipeline sanity check (~10s)
python scripts/smoke_test.py

# 4. train + evaluate (all-in-one notebook)
jupyter notebook notebooks/local_train_evaluate.ipynb

# 5. run the dashboard
streamlit run ui/app.py
```

---

## Project Structure

```
ai_for_engineering/
├── data/                 ← v2 dataset (diagnostic baseline)
├── data_v3/              ← v3 dataset (FINAL — train on this)
├── dataset/              ← raw source datasets (gitignored, on Drive)
├── best/                 ← downloaded v3 Kaggle results (weights + curves)
├── models/
│   └── ssdlite_detector.py
├── notebooks/
│   ├── local_train_evaluate.ipynb    ← all-in-one (local)
│   ├── colab_train_evaluate.ipynb    ← all-in-one (Colab T4)
│   ├── kaggle_train_evaluate.ipynb   ← all-in-one (Kaggle T4)
│   └── rtdetr_addon.py               ← source of the optional RT-DETR cell
├── scripts/              ← dataset build/repair/verify + training + notebook generator
├── ui/
│   ├── app.py            ← Streamlit dashboard (model + version selector)
│   └── inference.py      ← inference helpers
├── runs/                 ← training outputs, EDA charts, logs (nano baseline + audit)
├── weights/              ← COCO-pretrained base weights
├── docs/                 ← reports, rationale, cleanup plan
├── README.md
├── CLAUDE.md
└── requirements.txt
```

---

## Dashboard

```bash
streamlit run ui/app.py
```

Upload an image → bounding boxes, class labels, confidence, and severity
(`area% = bbox_area / image_area × 100`; Low <5%, Medium 5–20%, High >20%). The sidebar selects the
model (YOLOv11s/YOLOv8s small, YOLOv11n/v8n legacy nano, SSDLite) and the specific trained run.

---

## Evaluation

mAP@0.5, mAP@0.5:0.95, per-class AP, Precision/Recall/F1, confusion matrices, FPS, and model size.
Validation goes to `runs/experiment_tracker.csv`; the test set is scored once into
`runs/model_comparison.csv`. Per-class results motivate the v3 weak-class work.
