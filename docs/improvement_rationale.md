# Model & Hyperparameter Improvements — Rationale & Trade-offs

Covers report Steps 1, 2, 3 and 6. Baseline = the honest v1 result on real localized boxes
(YOLO-nano, 100 epochs, SGD): **mAP@0.5 ≈ 0.44**. Target after improvement: **mid-50s**.

## 1 — Data preprocessing: pothole size rebalancing (bigger-box bias)

**Problem.** After the v2 rebuild every class sat at 6–14% tiny boxes (<0.5% of image area) **except
potholes at 46.6%**. The tiny potholes came from the `road-damage` source (61% of its pothole boxes are
tiny — small, distant road potholes). Tiny boxes are the hardest to localize and are penalised heavily at
higher IoU thresholds, so they were the prime suspect for the low pothole AP.

**Why not "just add potholesv5".** The obvious fix — swap in `potholesv5` — is wrong: measured, v5 is
*smaller* than what we already used (median box 0.91% / 38% tiny) versus `potholesv1+V4` (median 2.14% /
18% tiny), so using it wholesale would have *raised* the tiny share. The usable signal in v5 is its **big**
images: ~1,163 images where *every* pothole box is ≥2% of the frame.

**What we did** (`balance()` in `scripts/01_reorganize_data.py`):
1. **`process_potholes_v5_big()`** — admit v5 only via its all-≥2% images, labels kept intact.
2. **Fill potholes biggest-first** — the single-class pothole pool is sorted by each image's *smallest*
   pothole box (descending), so clean big-pothole images are consumed before any tiny ones.
3. **Drop all-tiny multi-class images** — `road-damage` crack+pothole images whose potholes are *all* tiny
   are dropped (their cracks backfill from `wall_crack`/`Dataset3`).

**Design decisions (confirmed, with rationale):**
- **Relative per-class, not an absolute threshold** — each class keeps its 1,800 *biggest* boxes, so the
  1.01× balance is preserved and no class collapses. An absolute "drop <0.5%" rule would have gutted the
  inherently small/thin cracks and corrosion.
- **Select-only (never edit a kept image's labels)** — we choose *which images* to include but never delete
  a box from a kept label file, so no real defect is left present-but-unlabelled (which would train the
  model to treat a visible defect as background and corrupt evaluation).
- **Potholes only** — cracks and corrosion are *legitimately* small/elongated (thin cracks, rust streaks);
  biasing them toward big boxes would discard real defects, so the rule is pothole-scoped.

**Result:** pothole tiny<0.5% **46.6% → 7.6%** (≈700 tiny boxes replaced by big ones), all five classes
still ~1,800 boxes (**1.01× balance**), **0 degenerate, leak-free**; dataset 4,221 → **4,694 images /
9,015 boxes**. Cost: multi-class images **514 → 262** (the dropped all-tiny crack+pothole images), still
comfortably satisfying the rubric's multi-class requirement.

**Trade-off (Steps 3 & 6).** Because the *test* split is drawn from the same pool, it is now also biased
toward large potholes — reported pothole AP will look stronger, but the model has seen far fewer
small/distant potholes and will be **weaker on them in the field**. This is a deliberate
quality-over-coverage choice (clean, well-localised training signal over maximal scale coverage) and is
defensible as such — not a claim of uniformly strong pothole detection. Cracks and corrosion retain their
small/thin boxes, so small-defect capability is preserved for those classes.

## 2 — Why the baseline is the honest number to beat

The original v1 models scored mAP@0.5 ≈ 0.76, but that was inflated: spalling, corrosion and
paint_degradation used **whole-image proxy bounding boxes**, so any prediction overlapping the
frame counted as a true positive — the metric measured *presence*, not *localization*. With
real localized boxes the score fell to ≈ 0.44. That drop is not a regression; it is the model
being honestly graded on whether it puts the box in the right place. All improvements below are
measured against this honest baseline.

## 3a — Model capacity

| Change | Reason | Trade-off |
|--------|--------|-----------|
| YOLOv11n → **YOLOv11s** | COCO: YOLO11s ≈ 0.62 vs YOLO11n ≈ 0.54 mAP@0.5 → ~8 pt headroom. More channels/depth resolve fine crack texture. | ~9.4M vs 2.6M params, slower inference; fine on desktop GPU, heavy for embedded. |
| YOLOv8n → **YOLOv8s** | Same family, ~10–15% expected gain; keeps the v8-vs-v11 generation comparison fair (both at `s` scale). | ~11.2M vs 3.2M params. |
| **SSDLite custom anchors** | Default COCO anchors only cover aspect ratios {1/3..3}; our cracks/corrosion span 0.19–6.7 (5th–95th pct). Widening to `(2,3,5)` → {1/5..5} lets anchors match long/narrow defects → higher recall. | Manual analysis; 6→8 anchors/location (slightly bigger head, ~3.4→3.8M params); anchors now dataset-specific (less general). |

**Why SSDLite still trails YOLO (expected):** the gap is architectural — anchor-based matching,
COCO-tuned scales, and 320px native input mismatch narrow defects. Anchor tuning narrows the gap
but does not close it; this is a deliberate talking point, not a tuning failure.

## 3b — Training schedule

- **Epochs 100 → 110 with cosine LR, applied to all three models (fairness).** Larger models need
  more steps to converge; the extra epochs land in the low-LR tail where the model refines
  localization. Crucially, **YOLOv11s, YOLOv8s and SSDLite all train for the same 110 epochs** so
  the comparison isolates architecture, not training budget (SSDLite previously ran only 50). Risk:
  overfitting on a ~6k-image set — mitigated by best-epoch checkpointing (`best.pt` / `best_v*.pth`)
  and validation tracking.
- **SGD → AdamW (YOLO):** adaptive per-parameter LR + decoupled weight decay → faster, more
  stable convergence and typically +2–3% mAP on small datasets. Trade-off: more optimizer memory
  and mild sensitivity to LR; paired with a higher `lr0=1e-3` that AdamW tolerates well.
- **Cosine LR (`cos_lr=True`) from 1e-3:** smooth decay settles into a flatter minimum without
  manual step schedules. Flat schedules (baseline) are simpler but under-fit deep nets.
- **Batch 16 (YOLO) / 8 (SSD) + AMP** on ≥12 GB (T4); falls back to 8/2 on small GPUs. AMP halves
  activation memory and speeds throughput; enabled on T4 (stable), was off on the GTX 1650 (NaN
  loss in FP16). Risk: too-large a batch OOMs limited VRAM → the auto-fallback handles it.
- **SSDLite: two-phase SGD, momentum 0.9, lr 5e-4.** Phase 1 (epochs 1–5) freezes the ImageNet
  backbone and trains only the fresh detection head, so large head gradients don't wreck
  pretrained features; phase 2 unfreezes at lr/5 to fine-tune end-to-end. SGD+momentum is the
  stable, standard choice for SSD.

## 3c — Data augmentation & loss weighting (YOLO)

Augmentation was retuned toward **defect-safe** settings: surface defects are texture/edge cues, so heavy
colour distortion or patch-erasing destroys the very signal the model needs.

| Aug | Setting | Reason | Trade-off |
|-----|---------|--------|-----------|
| Mosaic | `mosaic=1.0`, `close_mosaic=20` | 4-image composites → more scale/context variety, strong small-object signal. Disabled last 20 epochs so the model finishes on real layouts. | Unrealistic composites if left on to the end → close_mosaic fixes that. |
| Random affine | `degrees=5`, `scale=0.4`, `translate=0.08` | Viewpoint/scale/framing invariance for field photos. Rotation cut 10→5 because walk-around inspection rarely rotates far and large angles distort thin cracks. | Less rotation robustness — acceptable, real inspection views are near-upright. |
| HSV jitter | `hsv_h=.015, hsv_s=.5, hsv_v=.35` | Lighting/weather/surface-colour variation. Saturation/value pulled back from .7/.4 so colour cues (rust, stains) aren't washed out. | Slightly less colour robustness; chosen because defects are texture-first. |
| Random erasing | `erasing=0.0` (OFF) | Default erasing blacks out random patches — on tiny/thin cracks and rust streaks it can delete the whole defect, training the model to ignore real signal → recall loss. | Loses one regularizer; mosaic + affine already supply variety. |
| H-flip | `fliplr=0.5` | Defects have no canonical left/right orientation. | None meaningful. |

**Loss weighting nudged toward localization:** `box=8.0, cls=0.4, dfl=1.7` (vs defaults 7.5/0.5/1.5). The
weak classes fail on *where* the box goes, not *what* it is, so up-weighting box/DFL regression and slightly
down-weighting classification targets the actual error mode. **Consequence:** raw box/cls/dfl loss
magnitudes are no longer comparable to default-weight runs — compare runs by **val mAP@0.5**, never by raw
loss.

**`multi_scale` — tried and removed.** Multi-scale training jitters the input size ±50%; on the T4 it pushed
imgsz to ~1216, which OOM'd at batch 16 and forced Ultralytics to auto-drop to batch 8 — roughly halving
throughput for a lever worth only ~0–2 pt. Removed; training stays at a fixed 640 / batch 16. (A short
**768 fine-tune tail** is the better small-object lever — applied at the end, not as full-run jitter.
It is **plan steps 4–5**: start from the 640 `best.pt` (not COCO), `imgsz=768`, `epochs=30–50`,
`lr0=2e-4`, light aug (`mosaic≤0.1`, `close_mosaic=0`, `degrees=3`, `scale=0.25`, `erasing=0`,
`multi_scale=False`), written to a **separate `*_768_ft` folder**. Full recipe: `docs/improvement_plan.md`.)

**MixUp/CutMix:** left off — at low data volume they tended to hurt small-object AP by overlaying defects
unrealistically; revisit at ~10% if val mAP supports it.
**Label smoothing (ε≈0.1):** not applied — current Ultralytics removed the `label_smoothing` training arg
(passing it errors). YOLO's classification loss already includes regularization.

## 6 — Trade-off summary (for the report)

- **Pothole size bias vs small-defect recall:** filtering potholes toward large, well-localised boxes
  (tiny 46.6%→7.6%) should lift pothole AP and tighten localization, but trades away small/distant pothole
  recall; the test split shares the bias, so quote pothole AP with that caveat. Cracks/corrosion keep their
  small boxes (pothole-scoped rule).
- **Capacity vs deployment:** `s` models buy accuracy with 3–4× params and slower inference —
  acceptable on a workstation, marginal on a drone/edge device. If deploying, quantize or export
  to TensorRT, or keep a nano model for the edge and an `s` model for offline review.
- **Epochs vs overfitting:** more epochs help the larger models converge but risk memorizing a
  6k-image set; best-checkpoint selection + watching val mAP@0.5:0.95 is the guard.
- **Optimizer:** AdamW = faster convergence + slightly higher mAP, at more memory and LR
  sensitivity; SGD = simpler/stabler but slower. We use AdamW for YOLO, SGD for SSD (stability).
- **Augmentation:** aggressive aug improves robustness but can fabricate unrealistic samples and
  hurt small objects; probabilities are kept moderate and mosaic is closed before the end.
- **Anchor tuning:** custom anchors lift recall on long/narrow defects but are dataset-specific
  and add manual analysis — justified here because the default anchors demonstrably miss the
  data's aspect-ratio range.

## Evaluation table (fill after training)

| Model | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | Speed (ms) | Params (M) |
|-------|---------|--------------|-----------|--------|-----------|-----------|
| YOLOv11s | _ | _ | _ | _ | _ | ~9.4 |
| YOLOv8s | _ | _ | _ | _ | _ | ~11.2 |
| SSDLite (custom anchors) | _ | _ | n/a | n/a | _ | ~3.8 |
| _baseline YOLOv8n (honest)_ | ~0.44 | ~0.25 | _ | _ | _ | ~3.2 |

Per-class AP, confusion matrices, severity distribution, and qualitative samples are produced by
Part 3 of `notebooks/local_train_evaluate.ipynb` (or its Colab twin) — `per_class_ap_*.png`,
`confusion_matrix_ssdlite.png`, `severity_distribution.png`, `prediction_samples.png`,
`model_comparison.csv` (now with Precision/Recall/F1/FPS/size columns).

**Small models only — `m`/`l`/`x` excluded by design (not just diminishing returns).** The
comparison is deliberately built around *small* detectors (YOLOv11s vs YOLOv8s primary). Medium/large
models would break the apples-to-apples small-model story, multiply params/latency, and risk
overfitting this dataset size — the larger lever is more *diverse real* data plus a **768 fine-tune
tail** (see §3c and `docs/improvement_plan.md`), **not** a bigger backbone. No `m`/`l`/`x` variant
appears anywhere in the notebooks, scripts, or experiment tables.
