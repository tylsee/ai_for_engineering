# Dataset Inspection & Cleaning (Report Step 1)

Scripts: `scripts/dataset_check.py` (inspection, reproducible) and
`scripts/clean_dataset.py` (in-place fix). Both are idempotent and re-runnable.

## 1a — Integrity & tiny-box audit

Inspected all 6,422 images / 10,000 boxes. Every image had ≥1 box and every label had a
matching image (0 empty, 0 orphan). Computing each box's area ratio (w·h, normalised)
surfaced two distinct problems that the "perfectly balanced" headline hid:

| Issue | Count | Classes | Decision |
|-------|-------|---------|----------|
| **Degenerate boxes** (w=0 or h=0, e.g. label `2 0 0 0 1`) | 121 | 114 corrosion, 7 spalling | **Dropped** |
| **Thin slivers** (one side 0 < dim < 0.012) | 391 | mostly corrosion/spalling | **Clamped** to MIN_DIM |
| **Small-but-valid** (area < 0.5%, both dims ≥ 0.012) | ~1,300 | mostly potholes (38%), corrosion | **Kept** |

**Why drop the degenerate boxes (not enlarge):** they are Roboflow polygon→bounding-box
export failures — a collapsed polygon produces a zero-width/height box whose true extent is
unknown. Visual review (`runs/flagged_corrosion_examples.png`) confirmed they sit on real
corrosion but carry no usable geometry. Fabricating a size would inject label noise into the
hardest class (corrosion) for an IoU-based metric, so they were removed. 118 of the 121 were
the **only** box on their image, so those 118 images were removed. This breaks the perfect
2000/class balance slightly (corrosion → 1886), which is an acceptable trade for clean labels.

**Why clamp the thin slivers (not drop):** unlike the degenerate boxes, these have a *known*
small extent (genuinely thin cracks / corrosion lines). Raising the small side to
MIN_DIM = 0.012 (~7.7 px @640) makes them trainable without inventing geometry.

**Why keep small-but-valid boxes:** the 0.5% area flag mostly catches legitimately small,
distant defects (potholes 38%, corrosion 24% of their boxes). Removing them would bias the
detector against exactly the hard, far-field cases that matter for real infrastructure
inspection. Small-object detection is a feature, not noise, so these were retained.

## 1b — Cross-split leakage

Re-ran 8×8 average-hash deduplication across all splits. Despite the pipeline's earlier
"zero leakage" claim, **2 cross-split duplicate pairs** were found and verified near-identical
by direct pixel comparison (mean abs diff ≈ 0.5/255 on 64×64, identical 416×416 size):

- `train/train_003394.jpg` ↔ `val/val_000655.jpg`
- `train/train_003431.jpg` ↔ `test/test_000727.jpg`

These leak training images into val/test and inflate mAP. The val/test copy of each pair was
deleted (train copy kept). After cleaning: **0 cross-split collisions**. (One harmless
within-train duplicate remains — redundancy, not leakage.)

## Result — cleaned dataset statistics

| Metric | Before | After |
|--------|--------|-------|
| Images | 6,422 | **6,302** |
| Boxes | 10,000 | **9,877** |
| Degenerate (zero-area) boxes | 121 | **0** |
| Max aspect ratio (corrosion) | 400,091 | 83.3 |
| Cross-split leakage pairs | 2 | **0** |

| Split | Images |
|-------|--------|
| train | 4,415 |
| val | 941 |
| test | 946 |

| Class | boxes (after) |
|-------|---------------|
| cracks | 2,000 |
| spalling | 1,993 |
| corrosion | 1,886 |
| potholes | 1,998 |
| paint_degradation | 2,000 |

Class balance ratio 1.06× (still effectively balanced). Chart: `runs/class_distribution.png`.

## Aspect-ratio findings → SSDLite anchors (feeds Step 3c)

Box aspect ratio (w/h) after cleaning, by class (median / 10th–90th pct):

| Class | median | 10th–90th | shape |
|-------|--------|-----------|-------|
| cracks | 1.02 | 0.40 – 2.25 | both orientations, wide spread |
| spalling | 0.96 | 0.40 – 2.48 | roughly square |
| corrosion | 1.16 | 0.06 – 31.7 | extremely elongated, both axes |
| potholes | 1.40 | 0.81 – 2.50 | square-to-wide, tight |
| paint_degradation | 0.75 | 0.35 – 1.66 | slightly tall |

Overall percentiles: 5th = 0.19, 50th = 1.04, 95th = 6.7. The default SSD/COCO anchor set
({0.5, 1, 2} plus {1/3, 3}) does **not** cover the tall (≤0.2) and very wide (≥5) shapes that
cracks and corrosion exhibit → motivates adding extreme aspect ratios to the SSDLite anchor
generator (Step 4b).
