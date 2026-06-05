"""YOLO training: baseline_640 is the default and final training stage.

Conventional Ultralytics only - built-in patience, save_period, resume. Small models only.
The 768 fine-tune was tested on v2/v3 and regressed (val mAP dropped). It is kept as an
optional ablation function but is NOT part of the default workflow.

The marked regions below are inlined into the notebooks:
  - 'config'       -> Part 2.1 Training configuration
  - 'functions'    -> shared helpers (inlined once, reused by 2.2 and 2.3)
  - 'yolov11s_run' -> Part 2.2 YOLOv11s baseline_640
  - 'yolov8s_run'  -> Part 2.3 YOLOv8s baseline_640

These functions expect the notebook's Part 0 globals (RUNS_DIR, DATA_YAML, BIG_GPU, WORKERS,
USE_AMP, torch). They are not meant to run standalone without that setup.
"""

# === BEGIN config ===
# Part 2.1 Training configuration - enable one model per session.
RUN_YOLO11S      = True
RUN_YOLO8S       = False
RUN_SSDLITE      = False
RUN_RTDETR       = False
RUN_FINETUNE_768 = False   # optional ablation only - tested on v2/v3, regressed vs baseline_640
QUICK_TEST       = False   # True -> 3-epoch smoke test of each stage
CACHE_MODE       = "disk"  # or False if storage is limited
FORCE_RETRAIN    = False   # True -> ignore existing checkpoints and retrain from scratch
DATASET_VERSION  = "v3"    # tag written into experiment_tracker.csv for traceability
print('config: RUN_YOLO11S=%s RUN_YOLO8S=%s RUN_SSDLITE=%s RUN_RTDETR=%s RUN_FINETUNE_768=%s | '
      'QUICK_TEST=%s CACHE_MODE=%s FORCE_RETRAIN=%s DATASET_VERSION=%s'
      % (RUN_YOLO11S, RUN_YOLO8S, RUN_SSDLITE, RUN_RTDETR, RUN_FINETUNE_768,
         QUICK_TEST, CACHE_MODE, FORCE_RETRAIN, DATASET_VERSION))
# === END config ===


# === BEGIN functions ===
import time
from pathlib import Path
import torch
from ultralytics import YOLO

TRACKER_COLS = ['dataset_version', 'model', 'stage', 'run_dir', 'imgsz', 'epochs_planned',
                'optimizer', 'lr0', 'patience', 'val_mAP50', 'val_mAP50_95',
                'precision', 'recall', 'comments']


def _ckpt_epoch(path):
    # Ultralytics sets epoch=-1 in a checkpoint once training finished (optimizer stripped).
    try:
        return int(torch.load(str(path), map_location='cpu').get('epoch', -1))
    except Exception:
        return -1


def resume_or_start(run_dir, init_weights):
    """Decide how to (re)start a stage in its own folder. Returns (action, weights).
    action in {'resume','complete','fresh'}. resume is only ever for the SAME interrupted run."""
    last = run_dir / 'weights' / 'last.pt'
    best = run_dir / 'weights' / 'best.pt'
    if FORCE_RETRAIN:
        print('  FORCE_RETRAIN=True -> fresh run (ignoring checkpoints in %s)' % run_dir.name)
        return 'fresh', init_weights
    if last.exists():
        ep = _ckpt_epoch(last)
        if ep == -1 and best.exists():
            print('  %s already complete -> validate only' % run_dir.name)
            return 'complete', best
        print('  %s interrupted (epoch %d) -> resume' % (run_dir.name, ep))
        return 'resume', last
    return 'fresh', init_weights


def _persist_base():
    mydrive = Path('/content/drive/MyDrive')          # Colab (Drive mounted)
    if mydrive.exists():
        d = mydrive / 'COS40007'
        d.mkdir(parents=True, exist_ok=True)
        return d
    kaggle = Path('/kaggle/working')                  # Kaggle output
    if kaggle.exists():
        return kaggle
    return None                                       # local - already persistent


def persist_results(subdir, run_name):
    """Back up only the just-finished run folder + the tracker/notes (not all of runs/)."""
    import shutil
    base = _persist_base()
    run_dir = RUNS_DIR / subdir / run_name
    if base is None:
        print('  local: %s already persistent' % run_dir.as_posix())
        return
    shutil.make_archive(str(base / ('%s_%s' % (subdir, run_name))), 'zip',
                        root_dir=str(run_dir.parent), base_dir=run_name)
    for f in ('experiment_tracker.csv', 'training_notes.md'):
        src = RUNS_DIR / f
        if src.exists():
            shutil.copy2(src, base / f)
    print('  backed up %s + tracker/notes -> %s' % (run_name, base))


def append_training_notes(line):
    with open(RUNS_DIR / 'training_notes.md', 'a', encoding='utf-8') as f:
        f.write(line.rstrip() + '\n')


def save_val_metrics(model, stage, run_dir, imgsz, epochs_planned, lr0, patience, val, comments=''):
    """Upsert one validation row into runs/experiment_tracker.csv (validation only, never test)."""
    import pandas as pd
    p, r = float(val.box.mp), float(val.box.mr)
    row = {'dataset_version': DATASET_VERSION, 'model': model, 'stage': stage,
           'run_dir': run_dir.as_posix(), 'imgsz': imgsz, 'epochs_planned': epochs_planned,
           'optimizer': 'AdamW', 'lr0': lr0, 'patience': patience,
           'val_mAP50': round(float(val.box.map50), 4), 'val_mAP50_95': round(float(val.box.map), 4),
           'precision': round(p, 4), 'recall': round(r, 4), 'comments': comments}
    path = RUNS_DIR / 'experiment_tracker.csv'
    df = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=TRACKER_COLS)
    # add missing columns for backward compat with older CSVs
    for col in TRACKER_COLS:
        if col not in df.columns:
            df[col] = ''
    df = df[~((df['model'] == model) & (df['stage'] == stage) &
              (df.get('dataset_version', '') == DATASET_VERSION))]
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)[TRACKER_COLS]
    df.to_csv(path, index=False)
    print('  tracker: %s %s [%s] val mAP@0.5=%.4f mAP@0.5:0.95=%.4f'
          % (model, stage, DATASET_VERSION, row['val_mAP50'], row['val_mAP50_95']))
    return row


def _validate_and_record(pretty, stage, subdir, run_name, imgsz, epochs, lr0, patience):
    run_dir = RUNS_DIR / subdir / run_name
    best = run_dir / 'weights' / 'best.pt'
    val = YOLO(str(best)).val(data=DATA_YAML, split='val', imgsz=imgsz)
    save_val_metrics(pretty, stage, run_dir, imgsz, epochs, lr0, patience, val)
    append_training_notes('- %s %s: val mAP@0.5=%.4f mAP@0.5:0.95=%.4f (imgsz=%d, weights=%s)'
                          % (pretty, stage, float(val.box.map50), float(val.box.map),
                             imgsz, best.as_posix()))
    persist_results(subdir, run_name)
    return val


def train_baseline_640(init_weights, subdir, pretty):
    """Stage 1: 640 baseline (epochs=110, patience=25). Returns best.pt path, or None if interrupted."""
    run_dir = RUNS_DIR / subdir / 'baseline_640'
    epochs = 3 if QUICK_TEST else 110
    close_mosaic = 0 if QUICK_TEST else 20
    action, ckpt = resume_or_start(run_dir, init_weights)
    if action == 'complete':
        _validate_and_record(pretty, 'baseline_640', subdir, 'baseline_640', 640, epochs, 0.001, 25)
        return run_dir / 'weights' / 'best.pt'

    def _do(batch):
        YOLO(str(ckpt)).train(
            data=DATA_YAML, imgsz=640, epochs=epochs, patience=25, batch=batch,
            optimizer='AdamW', lr0=0.001, cos_lr=True, warmup_epochs=5,
            mosaic=1.0, close_mosaic=close_mosaic, degrees=5.0, scale=0.4, translate=0.08,
            fliplr=0.5, hsv_h=0.015, hsv_s=0.5, hsv_v=0.35,
            erasing=0.0, multi_scale=False, box=8.0, cls=0.4, dfl=1.7,
            save_period=5, cache=CACHE_MODE, plots=True,
            device=0 if torch.cuda.is_available() else 'cpu',
            project=str(RUNS_DIR / subdir), name='baseline_640', exist_ok=True,
            workers=WORKERS, amp=USE_AMP, verbose=True)

    batch = 16 if BIG_GPU else 8
    t0 = time.time()
    try:
        if action == 'resume':
            YOLO(str(ckpt)).train(resume=True)
        else:
            try:
                _do(batch)
            except RuntimeError as e:
                if 'out of memory' in str(e).lower() and batch > 8:
                    print('  OOM at batch %d -> retry at 8' % batch)
                    torch.cuda.empty_cache()
                    _do(8)
                else:
                    raise
    except KeyboardInterrupt:
        print('  %s baseline_640 interrupted - re-run the cell to resume from last.pt' % pretty)
        return None
    print('  %s baseline_640 finished in %.0f min' % (pretty, (time.time() - t0) / 60))
    _validate_and_record(pretty, 'baseline_640', subdir, 'baseline_640', 640, epochs, 0.001, 25)
    return run_dir / 'weights' / 'best.pt'


def finetune_768(subdir, pretty):
    """Stage 2: 768 fine-tune (epochs=40, patience=12) from the 640 best.pt. resume only within 768."""
    base_best = RUNS_DIR / subdir / 'baseline_640' / 'weights' / 'best.pt'
    if not base_best.exists():
        print('  %s: no 640 best.pt -> run the baseline first; skipping 768' % pretty)
        return None
    run_dir = RUNS_DIR / subdir / 'finetune_768'
    epochs = 3 if QUICK_TEST else 40
    action, ckpt = resume_or_start(run_dir, base_best)
    if action == 'complete':
        _validate_and_record(pretty, 'finetune_768', subdir, 'finetune_768', 768, epochs, 0.0002, 12)
        return run_dir / 'weights' / 'best.pt'

    def _do(batch):
        # plain load from the 640 best.pt, resume=False (this is a fresh fine-tune, not a resume)
        YOLO(str(base_best)).train(
            data=DATA_YAML, imgsz=768, epochs=epochs, patience=12, batch=batch,
            optimizer='AdamW', lr0=0.0002, cos_lr=True, warmup_epochs=3,
            mosaic=0.0, close_mosaic=0, degrees=3.0, scale=0.25, translate=0.05,
            fliplr=0.5, hsv_h=0.015, hsv_s=0.4, hsv_v=0.3,
            erasing=0.0, multi_scale=False, box=8.0, cls=0.4, dfl=1.7,
            save_period=5, cache=CACHE_MODE, plots=True,
            device=0 if torch.cuda.is_available() else 'cpu',
            project=str(RUNS_DIR / subdir), name='finetune_768', exist_ok=True,
            workers=WORKERS, amp=USE_AMP, verbose=True)

    batch = 8 if BIG_GPU else 4
    t0 = time.time()
    try:
        if action == 'resume':
            YOLO(str(ckpt)).train(resume=True)
        else:
            try:
                _do(batch)
            except RuntimeError as e:
                if 'out of memory' in str(e).lower() and batch > 4:
                    print('  OOM at batch %d -> retry at 4' % batch)
                    torch.cuda.empty_cache()
                    _do(4)
                else:
                    raise
    except KeyboardInterrupt:
        print('  %s finetune_768 interrupted - re-run the cell to resume from last.pt' % pretty)
        return None
    print('  %s finetune_768 finished in %.0f min' % (pretty, (time.time() - t0) / 60))
    val = _validate_and_record(pretty, 'finetune_768', subdir, 'finetune_768', 768, epochs, 0.0002, 12)
    _compare_to_baseline(pretty, subdir, float(val.box.map50))
    return run_dir / 'weights' / 'best.pt'


def _compare_to_baseline(pretty, subdir, ft_map50):
    """Append a one-line note on whether 768 improved over the 640 baseline (validation mAP@0.5)."""
    import pandas as pd
    path = RUNS_DIR / 'experiment_tracker.csv'
    if not path.exists():
        return
    df = pd.read_csv(path)
    base = df[(df['model'] == pretty) & (df['stage'] == 'baseline_640')]
    if base.empty:
        return
    b = float(base.iloc[-1]['val_mAP50'])
    verdict = 'improved' if ft_map50 > b else 'did not improve'
    append_training_notes('  -> %s 768 fine-tune %s over 640 (val mAP@0.5 %.4f vs %.4f)'
                          % (pretty, verdict, ft_map50, b))
# === END functions ===


# === BEGIN yolov11s_run ===
if RUN_YOLO11S:
    _best = train_baseline_640('yolo11s.pt', 'yolo11s', 'YOLOv11s')
    # 768 fine-tune was tested on v2/v3 and regressed vs baseline_640.
    # Set RUN_FINETUNE_768=True only for ablation experiments.
    if _best is not None and RUN_FINETUNE_768:
        finetune_768('yolo11s', 'YOLOv11s')
else:
    print('RUN_YOLO11S=False - skipping YOLOv11s')
# === END yolov11s_run ===


# === BEGIN yolov8s_run ===
# Independent of YOLOv11s: own folders (runs/yolov8s/...), own resume, no YOLOv11s dependency.
if RUN_YOLO8S:
    _best8 = train_baseline_640('yolov8s.pt', 'yolov8s', 'YOLOv8s')
    # 768 fine-tune disabled by default - set RUN_FINETUNE_768=True for ablation only.
    if _best8 is not None and RUN_FINETUNE_768:
        finetune_768('yolov8s', 'YOLOv8s')
else:
    print('RUN_YOLO8S=False - skipping YOLOv8s')
# === END yolov8s_run ===


# === BEGIN final_test_eval ===
# Part 3 Final test evaluation.
# Default: scores baseline_640 for each model (the confirmed best stage on v3).
# Selection logic: picks baseline_640 by default; falls back to best-by-val only if a
# different stage actually outperforms it. The test set is scored ONCE per model.
# Writes runs/model_comparison.csv (test only - never mixed with the validation tracker).
import os as _os
import time as _time
import numpy as _np
import pandas as _pd
import matplotlib.pyplot as _plt
from ultralytics import YOLO as _YOLO

_SUBDIR = {'YOLOv11s': 'yolo11s', 'YOLOv8s': 'yolov8s'}


def evaluate_on_test():
    tracker = RUNS_DIR / 'experiment_tracker.csv'
    out = []

    # Determine which models to evaluate: prefer experiment_tracker entries,
    # fall back to scanning for baseline_640 weights directly.
    candidates = {}
    if tracker.exists():
        val_df = _pd.read_csv(tracker)
        # Filter to current dataset version if column exists
        if 'dataset_version' in val_df.columns:
            val_df = val_df[val_df['dataset_version'].astype(str) == str(DATASET_VERSION)]
        for model in sorted(val_df['model'].unique()):
            sub = val_df[val_df['model'] == model]
            best = sub.loc[sub['val_mAP50'].idxmax()]
            candidates[model] = (best['stage'], int(best['imgsz']), float(best['val_mAP50']))
    # Also scan for baseline_640 weights not yet in tracker
    for model, subdir in _SUBDIR.items():
        if model not in candidates:
            w = RUNS_DIR / subdir / 'baseline_640' / 'weights' / 'best.pt'
            if w.exists():
                candidates[model] = ('baseline_640', 640, 0.0)

    if not candidates:
        print('no trained models found - run Part 2.2 / 2.3 first')
        return

    for model, (stage, imgsz, val_map) in sorted(candidates.items()):
        weights = RUNS_DIR / _SUBDIR.get(model, model.lower()) / stage / 'weights' / 'best.pt'
        if not weights.exists():
            print('  %s: weights missing (%s) - skip' % (model, weights))
            continue
        note = '' if val_map == 0.0 else ' (val mAP@0.5=%.4f)' % val_map
        print('  %s: stage=%s imgsz=%d%s -> scoring TEST' % (model, stage, imgsz, note))
        m = _YOLO(str(weights))
        v = m.val(data=DATA_YAML, split='test', imgsz=imgsz)
        p, r = float(v.box.mp), float(v.box.mr)
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        imgs = [q for q in (DATA_DIR / 'images' / 'test').iterdir()
                if q.suffix.lower() in IMAGE_EXTS][:50]
        ts = []
        for q in imgs:
            t0 = _time.time(); m.predict(str(q), imgsz=imgsz, verbose=False)
            ts.append((_time.time() - t0) * 1000)
        fps = round(1000 / _np.mean(ts), 1) if ts else 0.0
        out.append({'dataset_version': DATASET_VERSION, 'model': model,
                    'chosen_stage': stage, 'imgsz': imgsz,
                    'test_mAP50': round(float(v.box.map50), 4),
                    'test_mAP50_95': round(float(v.box.map), 4),
                    'precision': round(p, 4), 'recall': round(r, 4), 'f1': round(f1, 4),
                    'fps': fps, 'size_MB': round(_os.path.getsize(weights) / 1e6, 1),
                    'weights': weights.as_posix()})
        try:
            per = dict(zip(CLASSES, [round(float(x), 4) for x in v.box.maps]))
            fig, ax = _plt.subplots(figsize=(8, 4))
            ax.bar(list(per.keys()), list(per.values()), color=CLASS_COLORS, edgecolor='black')
            ax.set_title('%s (%s) - Per-Class AP@0.5:0.95 (test)' % (model, stage))
            ax.set_ylim(0, 1)
            for i, (k, val) in enumerate(per.items()):
                ax.text(i, val + 0.01, '%.3f' % val, ha='center', fontsize=9)
            _plt.tight_layout()
            _plt.savefig(RUNS_DIR / ('per_class_ap_test_%s.png' % model.lower()), dpi=150)
            _plt.show()
        except Exception as e:
            print('  (per-class chart skipped: %s)' % e)
    if not out:
        print('no models evaluated')
        return
    comp = _pd.DataFrame(out)
    comp.to_csv(RUNS_DIR / 'model_comparison.csv', index=False)
    print('\nFinal test results [dataset=%s] -> runs/model_comparison.csv' % DATASET_VERSION)
    print(comp[['model', 'chosen_stage', 'test_mAP50', 'test_mAP50_95',
                'precision', 'recall', 'f1', 'fps']].to_string(index=False))
    print('\nNote: model_comparison.csv = TEST only; experiment_tracker.csv = VALIDATION only.')


evaluate_on_test()
# === END final_test_eval ===
