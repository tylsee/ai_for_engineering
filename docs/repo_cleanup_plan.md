# Repository Cleanup Plan

> **Status (updated 2026-06-06):** most of this was a proposal; the items below were then
> **executed with approval**. v3 is the final dataset; `baseline_640` is the default YOLO workflow;
> the 768 fine-tune is an optional ablation only.
>
> **Executed with approval (2026-06-06):**
> - Moved to `_archive/old_outputs/`: `nb_eval.txt`, `nb_outputs.txt`, `defect_dataset-a4.zip`.
> - Moved to `_archive/old_experiments/`: `clean_dataset.py`, `02_augmentation.py`
>   (CLAUDE.md references updated to the new paths).
> - Removed empty `runs_kaggle/` (no files lost).
> - Copied v3 YOLOv11s weights `best/v3/yolov11s[_finetune]/*.pt` →
>   `runs/yolo11s/{baseline_640,finetune_768}/weights/best.pt` (+ `results.csv`).
> - Ran the v3 final TEST (YOLOv11s baseline_640) → `runs/model_comparison.csv`.
> - Re-zipped `data_v3/` → `defect_dataset_v3.zip` (after the duplicate-label fix).
>
> Everything else below remains a proposal; nothing else was deleted, and no commit/push was made.

## Legend
- **keep** — actively used or required evidence; do not touch.
- **archive** — outdated but worth keeping; move to `_archive/` (only after approval).
- **delete candidate** — generated/temp/empty; safe to remove (only after approval).
- **unsure** — needs a human decision before any action.

---

## Top-level files & folders

| File/folder | Current purpose | Status | Reason | Risk if removed | Action recommended |
|-------------|-----------------|--------|--------|-----------------|--------------------|
| `CLAUDE.md` | Project instructions for Claude | keep | Source of project rules/history | High | Keep (updated this round) |
| `README.md` | Human-facing project readme | keep | Entry point | High | Keep (updated this round) |
| `requirements.txt` | Python deps | keep | Needed to run | High | Keep |
| `COS40007+...PDF` | Assignment brief | keep | Reference; gitignored (`*.pdf`) | Low (re-downloadable) | Keep local |
| `data/` | **v2** dataset (diagnostic baseline) | keep | CLAUDE.md keeps v2 as diagnostic | High | Keep — but see git-tracking note below |
| `data_v3/` | **v3** final dataset | keep | The dataset for all final training | High | Keep; add to `.gitignore` (see Stage 8) |
| `dataset/` | Raw source datasets | keep | Inputs to `01_reorganize_data.py`; gitignored | Medium (large; on Drive) | Keep local |
| `defect_dataset_v3.zip` (338 MB) | v3 bundle for Kaggle/Colab upload | keep | Needed to upload v3 for training | Low (rebuild via `zip_data.py`) | Keep; ensure gitignored |
| `defect_dataset-a4.zip` (357 MB) | **v2** snapshot bundle | archive | Superseded by v3 zip | Low (rebuild from `data/`) | Archive or delete after approval; ensure gitignored |
| `best/` | **Actual v3 Kaggle results** (yolov11s baseline + finetune) | keep | Real v3 evidence (weights, curves, results.csv) | High | Keep; see "location mismatch" below |
| `runs/` | Training outputs, EDA charts, logs | keep | Baseline runs + audit + charts | High | Keep |
| `runs_kaggle/` | Empty leftover folder | delete candidate | Contains no files | None | Delete after approval |
| `nb_eval.txt` | Scratch capture of a **v2-era** Kaggle eval (635 test imgs) | delete candidate | Stale; not final v3; superseded by `model_comparison.csv` | None | Archive or delete after approval |
| `nb_outputs.txt` | Scratch capture of v2-era notebook cell outputs | delete candidate | Stale debugging dump | None | Archive or delete after approval |
| `weights/` | COCO-pretrained base weights | keep | Used by notebooks | Medium (re-downloadable) | Keep; `yolov8s.pt` currently missing (see note) |
| `.vscode/`, `.qodo/`, `.claude/` | Editor/tool config | keep | Local only; gitignored | None | Keep local |

---

## `scripts/`

| File | Current purpose | Status | Reason | Risk if removed | Action recommended |
|------|-----------------|--------|--------|-----------------|--------------------|
| `01_reorganize_data.py` | Build `data/` and `data_v3/` from raw sources | keep | Core dataset pipeline | High | Keep |
| `prepare_training_data.py` | Repair+verify dataset (inlined to notebooks) | keep | Source of Part 1.6 | High | Keep |
| `train_yolo_two_stage.py` | Config + baseline_640 (+ optional 768) + final test eval | keep | Source of Part 2.1–2.3, Part 3 | High | Keep (already baseline-default) |
| `update_training_notebooks.py` | Inlines the above into the 3 notebooks | keep | Notebook generator | High | Keep (fixed this round) |
| `verify_rebuild.py` | Post-rebuild verification | keep | v3 verify gate | Medium | Keep |
| `zip_data.py` | Forward-slash-safe dataset zip | keep | Builds upload bundle | Medium | Keep |
| `audit_class_samples.py` | Weak-class contact sheets → `runs/audit_v3/` | keep | v3 label audit | Low | Keep |
| `crop_augment_weak_classes.py` | Train-only weak-class crop aug | keep | v3 pipeline step | Medium | Keep |
| `clean_bad_images.py` | Remove mislabeled/unlearnable v3 images | keep | v3 cleanup (used this round) | Medium | Keep |
| `add_rtdetr_cells.py` | Insert optional RT-DETR cell | keep | RT-DETR integration | Medium | Keep |
| `train_all.py` | CLI nano-baseline trainer | keep | Produces the report before/after nano baseline | Medium | Keep |
| `smoke_test.py` | ~10s pipeline sanity check | keep | Useful pre-run check | Low | Keep |
| `download_datasets.py` | Download raw datasets | keep | Reproducibility | Low | Keep |
| `02_augmentation.py` | Albumentations pipeline | unsure | **Orphaned** — writes `data/augmented/` not referenced by `data.yaml` | Low | Keep for report justification, or archive; document as orphaned |
| `dataset_check.py` | Dataset integrity + AR inspection | unsure | Overlaps `prepare_training_data.py` verify + `verify_rebuild.py` | Low | Keep as diagnostic, or archive |
| `clean_dataset.py` | Legacy in-place label cleaner | archive | **Superseded** by source cleaning (`_clean_box`) per CLAUDE.md | Low | Archive after approval |
| `_build_platform_notebooks.py` | Clone colab→kaggle env cells | unsure | Title says "all four detectors"; clones a cell range; overlaps the generator's job | Medium | Keep + document role, or archive if no longer the build path |
| `__pycache__/` | Python bytecode cache | delete candidate | Generated; gitignored | None | Delete after approval (regenerates) |

---

## `notebooks/`

| File | Current purpose | Status | Reason | Risk if removed | Action recommended |
|------|-----------------|--------|--------|-----------------|--------------------|
| `local_train_evaluate.ipynb` | All-in-one (local) | keep | Canonical local notebook (repaired this round) | High | Keep |
| `colab_train_evaluate.ipynb` | All-in-one (Colab T4) | keep | Cloud training (repaired this round) | High | Keep |
| `kaggle_train_evaluate.ipynb` | All-in-one (Kaggle T4) | keep | Cloud training (repaired this round) | High | Keep |
| `rtdetr_addon.py` | Source of the optional RT-DETR cell | keep | Used by `add_rtdetr_cells.py` | Medium | Keep |
| `kagglestructuraldefectdetectionv1.ipynb` (2.1 MB, with outputs) | The **hand-run Kaggle notebook** that actually trained v3 | keep (evidence) | Carries the real v3 training/eval outputs and logs | Medium | Keep as evidence; optionally archive a strip-outputs copy to shrink it |

---

## `docs/`

| File | Current purpose | Status | Reason | Action recommended |
|------|-----------------|--------|--------|--------------------|
| `weak_class_next_steps.md` | Next-direction (weak-class quality) | keep | Current, report-ready | Keep |
| `repo_cleanup_plan.md` | This plan | keep | This document | Keep |
| `improvement_plan.md` | Experiment roadmap | keep | Has SUPERSEDED banner pointing to current state | Keep (history) |
| `improvement_rationale.md` | Hyperparameter/trade-off rationale | keep | Report Task 2/4 evidence | Keep |
| `dataset_cleaning_report.md` | v2/old dataset cleaning write-up | keep | Report Task 1 evidence | Keep |
| `rtdetr_experiment_report.md` | RT-DETR write-up (results pending) | keep | Report evidence | Keep |
| `current_run_diagnosis.md` | Older YOLO plateau diagnosis (v2-era) | unsure | Useful history but partly superseded | Keep or archive to `_archive/old_docs/` |

---

## `runs/`, `models/`, `ui/`, `weights/`

| File/folder | Status | Reason / Action |
|-------------|--------|-----------------|
| `runs/yolo11n/v1`, `runs/yolov8n/v1` | keep | Old-data nano baseline — report before/after evidence |
| `runs/yolo11n/v3` | unsure | Nano v3 run; keep if referenced in report, else archive |
| `runs/audit_v3/` | keep | Weak-class contact sheets (v3 evidence) |
| `runs/*.png`, `runs/*.csv` (`class_distribution`, `flagged_*`, `sample_images`, `run_log`) | keep | EDA/report charts and logs |
| `runs/**/weights/last.pt`, `train_batch*.jpg` | delete candidate | Already gitignored; large; only `best.pt` needed |
| `models/ssdlite_detector.py`, `models/__init__.py` | keep | SSDLite definition |
| `models/__pycache__/` | delete candidate | Generated cache |
| `ui/app.py`, `ui/inference.py`, `ui/__init__.py` | keep | Streamlit demonstrator (updated this round) |
| `weights/yolo11s.pt`, `yolo11n.pt`, `yolov8n.pt` | keep | Pretrained bases (gitignored / re-downloadable) |
| `weights/yolov8s.pt` | **missing** | Add before training YOLOv8s (auto-downloads on first run) |

---

## Issues found that are NOT just file disposition

1. **Generated notebooks were broken** (now fixed): all three had duplicate "Part 2.1" markdown
   cells and were **missing the YOLOv11s/YOLOv8s run cells**, so they defined the training
   functions but never called them. Fixed by rewriting the region rebuild in
   `update_training_notebooks.py` (idempotent now).
2. **UI weights location mismatch:** `ui/inference.py._best_yolo_weights()` looks for
   `runs/yolo11s/baseline_640/weights/best.pt`, but the actual v3 weights are in
   `best/v3/yolov11s/yolo11s_baseline.pt` (different folder name `yolov11s` and file name).
   **Manual decision needed:** either copy the v3 weights into the `runs/<model>/baseline_640/weights/`
   layout the code expects, or point the UI at `best/`. Not auto-changed (touches evidence).
3. **`data/` (v2) is committed to git (12,848 files) with ~12,800 uncommitted changes.** This bloats
   the repo. CLAUDE.md intentionally keeps labels tracked for restore — see Stage 8 for the .gitignore
   discussion. **Decision needed**, not auto-changed.
4. **`nb_eval.txt` reports v2 numbers** (`cos40007-defect-dataset-v2`, 635 test imgs) — do not quote
   these as v3 results in the report.

---

## Suggested archive layout (only if you approve moving anything)

```
_archive/
├── old_notebooks/    # (none required yet)
├── old_docs/         # current_run_diagnosis.md (optional)
├── old_experiments/  # clean_dataset.py (legacy), 02_augmentation.py (orphaned) — if not kept
└── old_outputs/      # nb_eval.txt, nb_outputs.txt, defect_dataset-a4.zip
```
