# Improvement Plan — Ordered Experiment Roadmap

> **SUPERSEDED (6 Jun 2026 — updated 6 Jun 2026).** The 768 fine-tune was implemented and tested on
> both v2 and v3. On v2: val mAP 0.405 vs 640 baseline 0.416 (regressed). On v3: val mAP 0.441 vs
> 640 baseline 0.461 (regressed again, early-stopped at epoch 10). **The 768 fine-tune has been
> removed from the default workflow.** `RUN_FINETUNE_768 = False` in all notebooks. It is preserved
> as an optional ablation only. The default YOLO workflow is now `baseline_640` only. See
> CLAUDE.md → "Dataset v3 (FINAL)" and `docs/weak_class_next_steps.md` for next improvement direction.

**Authoritative ordering of the remaining model experiments.** This supersedes the ad‑hoc
"Experiment A/B/C/E" framing in `current_run_diagnosis.md` (that file is kept for its per‑class /
augmentation diagnosis only).

---

## Hard constraint — SMALL models only

Models are **capped at the `s` (small) scale**. **YOLOv11m, YOLOv8m, and any medium/large variant
(`m` / `l` / `x`) are explicitly excluded** — everywhere: notebooks, scripts, docs, and the
experiment/comparison tables.

**Why:** the report's comparison design is built around *small* detectors (YOLOv11s vs YOLOv8s, plus
SSDLite and RT‑DETR for architecture diversity). Medium/large models would (a) break that
apples‑to‑apples small‑model story, (b) multiply params/latency for a lever worth far less than more
diverse real data, and (c) risk overfitting the ~4.7k‑image set. Primary models: **YOLOv11s + YOLOv8s**.

---

## Plan (priority order)

| # | Step | Model | imgsz | Epochs | Start from | Run folder |
|---|------|-------|-------|--------|------------|------------|
| 1 | **Clean & verify the dataset** | — | — | — | — | — |
| 2 | Resume/restart **YOLOv11s** baseline | YOLOv11s | 640 | 110 | COCO `yolo11s.pt` | `runs/yolo11s/v1` |
| 3 | Train **YOLOv8s** baseline | YOLOv8s | 640 | 110 | COCO `yolov8s.pt` | `runs/yolov8s/v1` |
| 4 | **Fine‑tune YOLOv11s** @ 768 | YOLOv11s | 768 | 30–50 | **640 `best.pt`** | `runs/yolo11s/v1_768_ft` |
| 5 | **Fine‑tune YOLOv8s** @ 768 | YOLOv8s | 768 | 30–50 | **640 `best.pt`** | `runs/yolov8s/v1_768_ft` |
| 6 | *(optional)* YOLOv11s @ 896 | YOLOv11s | 896 | 30–50 | 640 `best.pt` | `runs/yolo11s/v1_896_ft` |
| 7 | **Evaluation‑only**: TTA / higher‑imgsz validation | — | 768/896 | — | trained weights | (eval only) |
| 8 | *(if time allows)* RT‑DETR | rtdetr‑l | 640 | 110 | COCO `rtdetr‑l.pt` | `runs/rtdetr/v1` |
| 9 | SSDLite — **comparison baseline only** | SSDLite | 640/320 | 110 | ImageNet backbone | `runs/ssdlite/best_v1.pth` |

**Notes on the order**
- Steps 1–3 establish the two **canonical small baselines** (640/110, fair budget) — these are the
  headline numbers in the report.
- Steps 4–6 are **resolution fine‑tune tails** layered on top of the 640 baselines (small‑object
  lever: distant potholes, thin cracks). They **must not overwrite** the 640 runs — hence the
  separate `*_768_ft` / `*_896_ft` folders.
- Step 6 (896) is **optional** — only if GPU memory and time allow (768 is the priority).
- Step 7 is **evaluation‑only** — TTA (`augment=True`) and/or validating at a higher image size are
  inference‑time experiments; they do **not** retrain and write no new training run.
- Step 8 (RT‑DETR) and Step 9 (SSDLite) round out the **architecture comparison** (transformer vs
  CNN; mobile/edge baseline). SSDLite is included for the comparison narrative, **never** as the
  best‑mAP candidate. RT‑DETR only if the schedule permits (it is the heaviest, ~32M params).

---

## 768 fine‑tune recipe (Steps 4 & 5 — and the 896 variant)

> **Implemented.** Steps 4–5 are wired into all three notebooks as a self‑contained **Part 2.5**
> cell (runs after Part 3). Source: `notebooks/finetune768_addon.py`, inserted via
> `scripts/add_finetune768_cells.py` (idempotent; it also hardens `best_yolo_weights` so the new
> `*_768_ft` folders don't break the version sort). The cell evaluates each fine‑tune on the test
> set at 768 and adds `YOLOv11s-768ft` / `YOLOv8s-768ft` rows to `runs/model_comparison.csv`
> alongside the 640 baselines. Toggle `RUN_896=True` in the cell for the optional Step 6 (896,
> YOLOv11s only). The standalone snippet below documents the exact recipe the cell applies.

Fine‑tune **from the 640 `best.pt`, not from COCO again** — the 640 run has already learned defect
features; the high‑res tail only needs to refine localization at the larger input size with a small
LR and light augmentation.

| Param | Value | Why |
|-------|-------|-----|
| `model` (weights) | **640 `best.pt`** (e.g. `runs/yolo11s/v1/weights/best.pt`) | warm‑start from learned defect features, not COCO |
| `imgsz` | **768** (896 for the optional step 6) | more pixels on small/thin defects (potholes, cracks) |
| `epochs` | **30–50** | short refinement tail; not a full retrain |
| `lr0` | **0.0002** | low LR so the fine‑tune refines, doesn't wreck, the 640 weights |
| `batch` | **4 or 8** | pick by GPU memory (768 is heavier than 640; drop to 4 on OOM) |
| `mosaic` | **0.0–0.1** | minimal — a short high‑res tail wants real layouts, not 4‑up composites |
| `close_mosaic` | **0** | no mosaic phase to close (mosaic already ~off) |
| `degrees` | **3.0** | gentle rotation only (large angles distort thin‑crack axis‑aligned boxes) |
| `scale` | **0.25** | modest scale jitter |
| `translate` | **0.05** | small framing jitter |
| `erasing` | **0.0** (OFF) | erasing can delete tiny/thin defects entirely → recall loss |
| `multi_scale` | **False** | fixed 768; multi‑scale OOMs the T4 and adds little |
| `optimizer` | AdamW | same family as the 640 baseline |
| `cos_lr` | True | smooth decay into the low‑LR tail |
| `project` / `name` | `runs/yolo11s` / `v1_768_ft` (resp. `yolov8s`) | **separate folder — never overwrite the 640 baseline** |

**Ultralytics snippet (YOLOv11s; mirror for YOLOv8s and the 896 variant):**

```python
from ultralytics import YOLO

# Step 4 — YOLOv11s 768 fine-tune from the 640 best.pt (NOT from COCO)
model = YOLO("runs/yolo11s/v1/weights/best.pt")
model.train(
    data="data/data.yaml",
    imgsz=768,
    epochs=40,            # 30–50
    lr0=0.0002,
    batch=8,              # 4 if OOM
    optimizer="AdamW",
    cos_lr=True,
    mosaic=0.1,           # 0.0–0.1
    close_mosaic=0,
    degrees=3.0,
    scale=0.25,
    translate=0.05,
    erasing=0.0,
    multi_scale=False,
    project="runs/yolo11s",
    name="v1_768_ft",     # separate folder — keeps runs/yolo11s/v1 (640) intact
    exist_ok=False,
)
```

For **Step 5** swap `runs/yolov8s/v1/weights/best.pt`, `project="runs/yolov8s"`, `name="v1_768_ft"`.
For **Step 6 (optional 896)** use the same recipe with `imgsz=896`, `batch=4`, `name="v1_896_ft"`.

---

## Evaluation‑only experiments (Step 7)

No retraining — these probe the trained weights:
- **TTA / test‑time augmentation:** `model.val(..., augment=True)` (or `predict(augment=True)`) —
  flips/scales at inference, trades speed for a small recall/mAP bump.
- **Higher‑imgsz validation:** `model.val(..., imgsz=768)` (or 896) on the **640‑trained** weights —
  shows how much of any gain is just resolution vs. genuine fine‑tuning.

Report both against the plain 640 `val`/`test` numbers so the resolution effect is isolated.

---

## What this plan deliberately does NOT do
- **No medium/large models** (`yolo11m`, `yolov8m`, `…l`, `…x`) — see the hard constraint above.
- **No overwriting the 640 baselines** — every fine‑tune writes a new `*_ft` folder.
- **No extending the 640 baseline past 110 epochs** — the 640 run stays the fair‑budget baseline;
  extra resolution comes from the separate fine‑tune tail, not more epochs at 640.
