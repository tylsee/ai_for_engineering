# Next Improvement Direction — Weak-Class Quality

> Written after v3 training results confirmed that cracks, corrosion and paint_degradation remain
> the weakest classes despite dataset cleanup and crop augmentation. The 768 fine-tune also
> regressed on v3 (val 0.441 vs 640 baseline 0.461), confirming that resolution is not the
> bottleneck. The remaining limitation is class-level data quality.

---

## Report-ready paragraphs

**On the 768 fine-tune result:**

"The 768-resolution fine-tuning stage was tested to determine whether a higher input resolution
could improve the detection of small and thin defects. However, the fine-tuned model did not
improve the validation mAP compared with the 640-resolution baseline (finetune_768: 0.441 vs
baseline_640: 0.461 on v3). This suggests that the remaining performance limitation was not
mainly caused by image resolution. Instead, the weaker performance of corrosion, paint
degradation and cracks indicates that label quality, bounding-box geometry, defect ambiguity and
class definition were more influential. Therefore, the 640 baseline was retained for final model
comparison, while future improvement should focus on weak-class dataset refinement rather than
further resolution-based fine-tuning."

**On v3 dataset and per-class results:**

"The v3 dataset improved the reliability of the training data by removing noisy sources,
filtering extreme corrosion aspect-ratio outliers, and applying train-only crop augmentation for
weak classes. However, the per-class validation results still show that cracks, corrosion and
paint degradation remain harder than potholes and spalling. This indicates that the next
improvement should focus on label review and weak-class dataset quality rather than increasing
epochs or input resolution."

---

## Specific checks for each weak class

### Cracks
- Check if bounding boxes are consistent around thin and branching cracks (diagonal cracks
  in axis-aligned boxes waste most of the box area, producing low IoU even for correct predictions).
- Check if dashcam / wide-angle road images are still present — these have cracks that are
  too small to be learnable (< 0.5% of image area). Run `scripts/clean_bad_images.py --data
  data_v3` to confirm 0 remaining after the cleanup.
- Check if crop augmentation produces too many zoomed-in patches that lose surface context
  (model may overfit to texture without position/scale cues). Consider reducing `MAX_CROPS_PER_CLASS`
  for cracks if contact sheets show unnatural zoom levels.
- If adding data: prioritise diverse wall/structural crack sources (different surfaces, angles,
  lighting) — not more road cracks (already well represented).

### Corrosion
- The corrosionv2 source is from a single ship/marine inspection domain (dark interiors,
  flashlight, steel panels). This is visually narrow. The model may fail on surface rust on
  concrete, bridges, or outdoor infrastructure (different background, lighting, colour).
- Check contact sheets (`runs/audit_v3/`) for images with very minimal rust (borderline labels)
  or images where rust and paint peeling co-occur (boxes should be on the rust, not the peeling).
- The AR filter (p95 10.6 → 3.0) already removed the worst outliers. If corrosion AP is still
  low after v3 training, the domain mismatch (ship-only source) is the likely cause.
- If adding data: prioritise outdoor structural corrosion (bridge beams, rebar, concrete-embedded
  steel) with diverse lighting and scale.

### Paint degradation
- Check if boxes overlap visually with stains, shadows, spalling, or general surface weathering
  (the class boundary with spalling is inherently ambiguous).
- The old `paint-degradation/` source (capped to 700 in v2, removed in v3) contained road
  markings and generic surface marks — confirm these are fully absent from v3 by checking the
  manifest and contact sheets.
- If contact sheets show paint_degradation boxes on images that look more like spalling
  (plaster delamination exposing brick), decide on a consistent rule and apply it uniformly.

---

## Dataset-level improvements (if rebuilding)

- **Image-level split balance:** v3 balanced by box count, but corrosion/paint may have many
  boxes per image (dense labelling) while cracks have few. A model that sees few unique crack
  images but many corrosion/paint images may underfit cracks despite equal box counts. Consider
  checking unique image counts per class per split.
- **False-negative review:** visually inspect a sample of corrosion/crack/paint images from
  val/test where the model predicts nothing — these false negatives reveal what the model has
  not learned. Add representative images of those failure modes to training.
- **Label consistency audit:** for each weak class, create contact sheets with boxes overlaid
  and check that the labelling rule is applied uniformly (e.g. does "corrosion" include rust
  stains on concrete, or only on metal? is a rust-streaked surface with peeling coating labelled
  as corrosion, paint, or both?).

---

## What NOT to do next

- Do not increase baseline_640 epochs beyond 110 until all four main models have been trained
  and compared on v3.
- Do not change LR, optimizer, augmentation, batch size, or loss weights at the same time as
  a data change — keep one variable at a time for clean attribution.
- Do not add medium/large YOLO models (YOLOv11m, YOLOv8m) to the main comparison — the
  project is scoped to small models.
- Do not re-run 768 fine-tune without a specific hypothesis (e.g. after significantly improving
  crack/corrosion label quality, resolution may help thin features more).
