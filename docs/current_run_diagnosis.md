# YOLOv11s — Current Run Diagnosis (epoch ~87 snapshot)

Source: `yolov11s_v1_run_results/` (`results.csv`, `args.yaml`, `best.pt`).
Snapshot stops at **epoch 87 of 110**. All eval below is on the **VAL** split
(test set kept untouched for the final honest number).

---

## 1. Is mAP plateauing? — Partly, but the measurement is mid-mosaic

| Epoch | P | R | mAP@0.5 | mAP@0.5:0.95 | train box_loss | val box_loss |
|------:|------|------|--------|--------|------|------|
| 57 | 0.622 | 0.382 | 0.397 | 0.217 | 1.557 | 2.277 |
| 70 | 0.682 | 0.382 | 0.412 | 0.229 | 1.477 | 2.270 |
| 81 | 0.677 | 0.394 | 0.428 | 0.240 | 1.427 | 2.221 |
| 87 | 0.682 | 0.394 | **0.426** | 0.242 | 1.390 | 2.259 |

mAP gain has slowed (+0.014 over epochs 70→87), **but the run has not reached
`close_mosaic`**. The config is `epochs=110, close_mosaic=20`, so mosaic
augmentation is disabled only for the **last 20 epochs (epoch 90→110)**. During
the mosaic phase every image is tiled 4-up, so each defect trains at ~half size
and the val distribution (full images) is harder to match. The characteristic
end-of-training jump when mosaic turns off (often **+0.02–0.05 mAP**) has **not
happened yet** in this snapshot. **Conclusion: let the Kaggle run finish to 110
before declaring the final number** (this is Experiment E's answer — do not
*extend* past 110, but do *finish* to 110).

## 2. Is the model overfitting? — Only mildly

- `train/box_loss` 1.56 → 1.39 (epochs 57→87): **still decreasing.**
- `val/box_loss` ~2.27 flat since epoch ~73; `val/cls_loss` still slowly falling.
- Val loss is **flat, not rising** → this is **saturation**, not destructive
  overfitting. The model is not capacity-bound either (train loss still moving).

## 3. The augmentation smoking gun (from `args.yaml`)

The run uses Ultralytics defaults that are tuned for COCO-style medium/large
objects, not tiny structural defects:

| Param | Value | Effect on tiny defects |
|-------|-------|------------------------|
| `erasing` | **0.4** | Random-erase wipes a rectangular patch 40% of the time. On a box <0.5% of the image it can **erase the entire defect while keeping the label** → trains the model to predict "nothing" → **suppresses recall directly.** |
| `scale` | 0.5 | ±50% scale jitter; stacked on mosaic, shrinks small objects further. |
| `mosaic` | 1.0 | 4-up tiling halves effective object size during 90/110 epochs. |
| `degrees` | 10.0 | Rotating thin cracks inflates their axis-aligned box → looser labels → hurts mAP@0.5:0.95 (matches the low 0.24). |

These are the basis for **Experiment C** (reduce `erasing`→0, `scale`→0.3,
`degrees`→0, `close_mosaic`→30).

## 4. Confidence threshold — what it can and cannot do

mAP@0.5 is computed by Ultralytics at conf≈0.001, **integrated over the whole
PR curve**, so **lowering the deployment confidence threshold does NOT raise
mAP** — the curve is fixed. What the sweep (Experiment A) gives:
- the right **operating point** for deployment (trade precision for recall,
  which is the correct bias for safety-critical inspection), and
- the per-class / FN / FP breakdown below.

See `experiments/expA/confidence_threshold_sweep.csv` + `.png`.

---

## 5. Why recall is low — the dataset geometry (Part 2)

Box-area buckets across all 9,877 boxes (`experiments/expA/small_box_distribution.csv`):

| Class | <0.1% | 0.1–0.5% | 0.5–1% | >1% | TOTAL | % small (<0.5%) |
|-------|------:|---------:|-------:|----:|------:|----------------:|
| cracks | 5 | 235 | 331 | 1429 | 2000 | 12.0% |
| spalling | 13 | 139 | 119 | 1722 | 1993 | 7.6% |
| corrosion | 34 | 146 | 188 | 1518 | 1886 | 9.5% |
| potholes | **79** | **679** | 350 | 890 | 1998 | **37.9%** |
| paint_degradation | 7 | 32 | 63 | 1898 | 2000 | 2.0% |

Aspect ratio (w/h) per class:

| Class | 5th pct | median | 95th pct | max |
|-------|--------:|-------:|---------:|----:|
| cracks | 0.29 | 1.02 | 2.90 | 13.97 |
| spalling | 0.28 | 0.96 | 3.80 | 83.30 |
| corrosion | **0.03** | 1.20 | **55.76** | 83.33 |
| potholes | 0.71 | 1.40 | 2.98 | 9.03 |
| paint_degradation | 0.28 | 0.75 | 2.16 | 45.38 |

**Interpretation:**
- **potholes** — 37.9% of boxes are <0.5% of image area (distant road potholes).
  At imgsz 640 these are a handful of pixels → the dominant recall sink.
  **→ Experiment B (imgsz 768) should help this class most.**
- **corrosion** — extreme aspect ratios (95th pct AR ≈ 56). Diagonal rust streaks
  forced into axis-aligned boxes give inherently loose IoU → lowest mAP@0.5:0.95.
  This is a **label-geometry limitation**, not a tuning bug (honest report point).
- **paint_degradation** — boxes are large (98% > 1% area); any weakness here is
  **texture/background confusion** (stains, shadows), not resolution.

---

## 6. SSDLite audit (Part 5) — already correct, no fix needed

Verified in `notebooks/colab_train_evaluate.ipynb` cells 5/6/20/21:

- `NUM_CLASSES = len(CLASSES) + 1 = 6` ✅ (class 0 = background)
- `labels.append(cls + 1)` in `DefectDataset` ✅ (YOLO 0–4 → SSD 1–5)
- `ANCHOR_AR = (2, 3, 5)` → boxes `{1/5..5}`, head rebuilt to match ✅
- Two-phase SGD: epochs 1–5 backbone frozen, epoch 6+ unfrozen at lr/5 ✅
- Checkpoint stores `anchor_ar` + `img_size` so eval rebuilds the matching head ✅

The old SSDLite mAP ≈ 0.001 was the **retired** broken version. The current
notebook code is fixed; expect ~0.10–0.25 mAP@0.5 once trained. No restart-fixing
action required beyond running it.

---

## 7. Per-class AP on VAL (from Experiment A)

<!-- FILLED FROM experiments/expA/per_class_ap_table.csv AFTER val completes -->
_Pending val run — see `experiments/expA/per_class_ap_table.csv`,
`confusion_matrix.png`, `BoxPR_curve.png`._

---

## Recommendation summary

> **Authoritative ordering now lives in `docs/improvement_plan.md`** (small models only). This
> diagnosis file is kept for its per-class / augmentation analysis; the table below is mapped onto
> the new plan‑step numbers.

| Plan step | Action | Where | Cost | Expected |
|-----------|--------|-------|------|----------|
| 2 | **Finish/restart YOLOv11s @640 to 110** (capture mosaic-off bump) | Kaggle/Colab | ~30 min left | +0.02–0.05 mAP |
| 3 | Train YOLOv8s @640 / 110 | Kaggle/Colab | ~2.5 h | v8-vs-v11 comparison |
| 4–5 | **768 fine-tune** YOLOv11s/v8s from 640 `best.pt` → `*_768_ft` folders | Kaggle/Colab | ~3–4 h ea | mAP ↑ mainly potholes/cracks |
| 7 | Confidence sweep / TTA / higher-imgsz val (eval-only) | local | none | operating point + class ranking |
| 8–9 | RT-DETR (*if time*) ; SSDLite (comparison baseline only) | Kaggle | — | architecture diversity |

**Small models only** — no `m`/`l`/`x` variant (see `docs/improvement_plan.md`). **Do not** extend
epochs past 110 on the 640 baseline, move test→train, augment val/test, or re-introduce proxy boxes.
The 768 gains come from the **separate `*_768_ft` fine-tune tail**, not from more epochs or a bigger
model; the 640/110 run stays the headline baseline.
