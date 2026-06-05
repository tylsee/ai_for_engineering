# RT-DETR Experiment Report

> Report-ready writeup for the 4th comparison model. Fill the `_pending_` result
> cells after running the RT-DETR cell on Kaggle/Colab (it writes the numbers into
> `runs/model_comparison.csv` and `runs/rtdetr_conf_threshold_sweep.csv`).

## 1. Why RT-DETR was added
The project already compares three convolutional detectors: YOLOv11s (latest YOLO),
YOLOv8s (established YOLO baseline) and SSDLite320-MobileNetV3 (lightweight
anchor-based edge model). RT-DETR was added to introduce **architecture diversity** —
a transformer-based detector — so the report can contrast YOLO-style convolutional
detection against a modern transformer approach, and discuss trade-offs beyond a
single mAP number (recall behaviour, duplicate suppression, speed, model size).

## 2. What RT-DETR is
**RT-DETR = Real-Time Detection Transformer.** It is an end-to-end detector that
replaces the hand-designed parts of the pipeline (anchors, NMS) with a transformer
encoder-decoder operating on a set of learned **object queries**. Each query predicts
one object directly, so detection is end-to-end and NMS-free. We use the smallest
Ultralytics variant, `rtdetr-l.pt` (~32M params, COCO-pretrained).

## 3. How RT-DETR differs from YOLO / SSDLite
| Aspect | YOLOv11s / YOLOv8s | SSDLite | RT-DETR |
|--------|--------------------|---------|---------|
| Family | CNN, one-stage | CNN, one-stage | Transformer, end-to-end |
| Anchors | anchor-free | anchor-based (default boxes) | anchor-free (object queries) |
| NMS | yes | yes | **no** (set prediction) |
| Params | 9.4M / 11.2M | ~3.8M | ~32M |
| Strength | speed + accuracy | tiny/edge deployable | global reasoning, duplicate-free |

## 4. Training setup (fair comparison)
- **Same** cleaned dataset, same train/val/test split, **no** augmentation on val/test.
- **Same** image size (640) and epoch budget as YOLO/SSDLite (`EPOCHS`, 110 for the full run).
- Labels stay `0..4` (cracks, spalling, corrosion, potholes, paint_degradation). The
  `+1` background shift is **only** for torchvision SSDLite and is **not** applied here.
- Transfer learning from COCO-pretrained `rtdetr-l.pt`.
- Auto-versioned to `runs/rtdetr/v1/` (never overwrites YOLO/SSDLite runs).

## 5. Hyperparameter choices
| Param | Value | Rationale |
|-------|-------|-----------|
| weights | `rtdetr-l.pt` | smallest Ultralytics RT-DETR; COCO-pretrained |
| epochs | `EPOCHS` (110) | identical budget to the other 3 models |
| imgsz | 640 | identical to YOLO for a fair comparison |
| batch | 4 (T4) | RT-DETR is memory-heavier than YOLOv11s; start small, raise to 8 only if VRAM allows, drop to 2 on OOM (16->8->4->2) |
| optimizer | AdamW | standard for DETR-style training |
| lr0 | 1e-4 | DETR-style models prefer a lower LR than YOLO's 1e-3 |
| cos_lr | True | smooth decay, same as YOLO |
| AMP | True on CUDA | speed/memory; stable on T4 |

## 6. Fairness of the comparison
Same data, same split, same image size, same epoch count, same untouched test set,
all evaluated with the same Ultralytics metric code (mAP@0.5, mAP@0.5:0.95, P, R, F1).
The only differences are architecture-appropriate (optimizer LR, batch size, anchors),
which is the intended axis of comparison. **Caveat:** RT-DETR-l (~32M) has ~3x the
parameters of YOLOv11s (9.4M); this is noted as a size/speed trade-off rather than a
flaw in the comparison.

## 7. Results table (test set) — fill after training
| Model | Arch | Epochs | ImgSz | Params (M) | mAP@0.5 | mAP@0.5:0.95 | P | R | F1 | FPS | Size (MB) |
|-------|------|-------:|------:|-----------:|--------:|-------------:|--:|--:|---:|----:|----------:|
| YOLOv11s | CNN anchor-free | 110 | 640 | 9.4 | _pending_ | _pending_ | _ | _ | _ | _ | _ |
| YOLOv8s | CNN anchor-free | 110 | 640 | 11.2 | _pending_ | _pending_ | _ | _ | _ | _ | _ |
| SSDLite | CNN anchor-based | 110 | 640/320 | 3.8 | _pending_ | _pending_ | _ | _ | _ | _ | _ |
| **RT-DETR** | **Transformer** | 110 | 640 | ~32 | _pending_ | _pending_ | _ | _ | _ | _ | _ |

Source: `runs/model_comparison.csv` (RT-DETR row appended by the RT-DETR cell).

## 8. Per-class analysis — fill after training
Per-class AP@0.5:0.95 for the five classes (`runs/per_class_ap_rtdetr.png`). Compare
against YOLOv11s/YOLOv8s. Questions to answer:
- Does RT-DETR improve **recall** vs YOLO (the project's weak spot: YOLO R~0.36-0.41)?
- Does it improve **corrosion** (elongated, high-AR) or **small potholes**?
- Which classes get worse? Transformers can need more data for small, texture-like defects.

## 9. Confidence sweep (recall tuning) — fill after training
`runs/rtdetr_conf_threshold_sweep.csv/.png` sweeps conf {0.05..0.25}. For structural
inspection, **recall matters more than precision** (a missed defect is riskier than a
false alarm reviewed by an engineer), so a lower deployment threshold may be acceptable.

## 10. Failure cases / trade-offs — fill after training
- If RT-DETR mAP **< YOLO**: transformer detectors typically need more data/longer
  schedules; small, thin, texture-like defects are hard for query-based matching.
- If RT-DETR mAP **> YOLO**: query/set-prediction reasoning reduced duplicate boxes and
  improved object-level localisation.
- If RT-DETR improves **recall but lowers precision**: better for inspection *screening*.
- If RT-DETR is **slower / larger**: classic accuracy vs speed/size trade-off (FPS, MB).

## 11. Final recommendation — fill after training
RT-DETR does **not** automatically replace YOLO. It is included to strengthen the
academic comparison. YOLO may remain preferred if it has the better mAP/FPS balance;
RT-DETR may be preferred if it gives materially better recall or localisation. State
the chosen "best for deployment" model with the metric that justifies it.

---
*Reproduce:* run the RT-DETR cell in `notebooks/colab_train_evaluate.ipynb` (or
`local_train_evaluate.ipynb`) after Part 3. It is self-contained and writes all the
artifacts referenced above.
