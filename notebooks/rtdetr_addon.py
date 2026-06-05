# ============ RT-DETR (transformer detector) - train + evaluate (self-contained) ============
# Runs AFTER Part 3 so runs/model_comparison.csv already exists for the other 3 models.
# Uses the SAME data.yaml, train/val/test split, image size and epoch budget for a FAIR
# comparison. Labels stay 0..4 (NO +1 shift - that is only for torchvision SSDLite).
# Trains -> evaluates on the UNSEEN test set -> per-class AP -> confidence sweep ->
# prediction samples -> MERGES its row into runs/model_comparison.csv.
from ultralytics import RTDETR
import os, time
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, matplotlib.patches as patches
from PIL import Image

# ---- RT-DETR config (self-contained; reuses EPOCHS/IMG_SIZE/BIG_GPU from Part 0) ----
RTDETR_WEIGHTS = 'rtdetr-l.pt'              # smallest Ultralytics RT-DETR (~32M params)
RTDETR_EPOCHS  = EPOCHS                      # SAME budget as YOLO/SSD (fair comparison)
RTDETR_IMG     = IMG_SIZE                    # 640, same as YOLO
RTDETR_BATCH   = 4 if BIG_GPU else 2         # heavier than YOLO; raise to 8 only if VRAM allows,
                                             # drop to 2 on CUDA OOM (16->8->4->2)
RTDETR_LR      = 1e-4                        # DETR-style models prefer a lower LR than YOLO
print('RT-DETR config: %s epochs=%d imgsz=%d batch=%d lr=%.0e'
      % (RTDETR_WEIGHTS, RTDETR_EPOCHS, RTDETR_IMG, RTDETR_BATCH, RTDETR_LR))

# ---- 2.4  TRAIN (auto-versioned, never overwrites existing runs) ----
_sub = 'rtdetr'
_ver = next_version(_sub)
print('RT-DETR -> runs/%s/v%d  (epochs=%d, imgsz=%d, batch=%d)'
      % (_sub, _ver, RTDETR_EPOCHS, RTDETR_IMG, RTDETR_BATCH))
rtdetr = RTDETR(RTDETR_WEIGHTS)                      # downloads rtdetr-l.pt (~63 MB) if absent
_t0 = time.time()
rtdetr.train(data=DATA_YAML, epochs=RTDETR_EPOCHS, imgsz=RTDETR_IMG, batch=RTDETR_BATCH,
             optimizer='AdamW', lr0=RTDETR_LR, cos_lr=True,
             device=0 if torch.cuda.is_available() else 'cpu',
             project=str(RUNS_DIR / _sub), name='v%d' % _ver, exist_ok=False,
             amp=USE_AMP, workers=WORKERS, plots=True, verbose=True)
_mins = (time.time() - _t0) / 60
_wts = RUNS_DIR / _sub / ('v%d' % _ver) / 'weights' / 'best.pt'
print('RT-DETR trained in %.0f min -> %s' % (_mins, _wts))

# ---- 3.x  EVALUATE on the UNSEEN test set ----
rtdetr_best = RTDETR(str(_wts))
val = rtdetr_best.val(data=DATA_YAML, split='test', imgsz=RTDETR_IMG)
p, r = float(val.box.mp), float(val.box.mr)
f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
params = sum(pp.numel() for pp in rtdetr_best.model.parameters()) / 1e6
_imgs = [q for q in (DATA_DIR / 'images' / 'test').iterdir()
         if q.suffix.lower() in IMAGE_EXTS][:50]
_ts = []
for q in _imgs:
    _a = time.time(); rtdetr_best.predict(str(q), verbose=False); _ts.append((time.time() - _a) * 1000)
_fps = round(1000 / np.mean(_ts), 1) if _ts else 0.0
row = {'Epochs': RTDETR_EPOCHS, 'Image Size': RTDETR_IMG, 'Optimizer': 'AdamW',
       'Params (M)': round(params, 2),
       'mAP@0.5': round(float(val.box.map50), 4), 'mAP@0.5:0.95': round(float(val.box.map), 4),
       'Precision': round(p, 4), 'Recall': round(r, 4), 'F1': round(f1, 4),
       'FPS': _fps, 'Size (MB)': round(os.path.getsize(_wts) / 1e6, 1),
       'Notes': 'transformer (DETR), AdamW, cosine LR'}
print('RT-DETR test: mAP@0.5=%.4f mAP@0.5:0.95=%.4f P=%.3f R=%.3f F1=%.3f FPS=%.1f'
      % (row['mAP@0.5'], row['mAP@0.5:0.95'], p, r, f1, _fps))
log_run('RT-DETR', RTDETR_EPOCHS, RTDETR_BATCH, RTDETR_IMG, 'AdamW',
        row['mAP@0.5'], row['mAP@0.5:0.95'], p, r, _mins, params, str(_wts),
        notes='RT-DETR-l transformer, lr=%.0e' % RTDETR_LR)

# ---- MERGE into runs/model_comparison.csv (other 3 models already written by Part 3) ----
_cmp = RUNS_DIR / 'model_comparison.csv'
comp = pd.read_csv(_cmp, index_col=0) if _cmp.exists() else pd.DataFrame()
comp.loc['RT-DETR'] = pd.Series(row)
comp.to_csv(_cmp)
print('\nUpdated', _cmp)
print(comp.to_string())

# ---- PER-CLASS AP ----
per = dict(zip(CLASSES, [round(float(x), 4) for x in val.box.maps]))
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(list(per.keys()), list(per.values()), color=CLASS_COLORS, edgecolor='black')
ax.set_title('RT-DETR - Per-Class AP@0.5:0.95'); ax.set_ylim(0, 1)
for i, (k, v) in enumerate(per.items()):
    ax.text(i, v + 0.01, '%.3f' % v, ha='center', fontsize=9)
plt.tight_layout(); plt.savefig(RUNS_DIR / 'per_class_ap_rtdetr.png', dpi=150); plt.show()

# ---- CONFIDENCE SWEEP (recall is the project's weak spot -> find the operating point) ----
rows = []
for t in [0.05, 0.10, 0.15, 0.20, 0.25]:
    v = rtdetr_best.val(data=DATA_YAML, split='test', imgsz=RTDETR_IMG, conf=t, verbose=False)
    pp, rr = float(v.box.mp), float(v.box.mr)
    rows.append({'conf': t, 'precision': round(pp, 4), 'recall': round(rr, 4),
                 'f1': round(2 * pp * rr / (pp + rr) if (pp + rr) > 0 else 0.0, 4),
                 'mAP@0.5': round(float(v.box.map50), 4),
                 'mAP@0.5:0.95': round(float(v.box.map), 4)})
sweep = pd.DataFrame(rows); sweep.to_csv(RUNS_DIR / 'rtdetr_conf_threshold_sweep.csv', index=False)
print('\nRT-DETR confidence sweep (test):'); print(sweep.to_string(index=False))
fig, ax = plt.subplots(figsize=(7, 5))
for col in ('precision', 'recall', 'f1'):
    ax.plot(sweep['conf'], sweep[col], 'o-', label=col.capitalize())
ax.set_xlabel('Confidence threshold'); ax.set_ylabel('Score')
ax.set_title('RT-DETR confidence sweep (test)'); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(RUNS_DIR / 'rtdetr_conf_threshold_sweep.png', dpi=120); plt.show()

# ---- QUALITATIVE PREDICTIONS (same first 6 unseen test images as the YOLO samples) ----
_test6 = sorted(q for q in (DATA_DIR / 'images' / 'test').iterdir()
                if q.suffix.lower() in IMAGE_EXTS)[:6]
fig, axes = plt.subplots(2, 3, figsize=(15, 8)); axes = axes.flatten()
for i, q in enumerate(_test6):
    res = rtdetr_best.predict(str(q), conf=0.25, verbose=False)[0]
    im = Image.open(q).convert('RGB'); axes[i].imshow(im); w, h = im.size
    for box in res.boxes:
        c = int(box.cls.item()); cf = float(box.conf.item())
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        s = severity_label((x2 - x1) / w, (y2 - y1) / h)
        axes[i].add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, lw=2,
                          edgecolor=CLASS_COLORS[c], facecolor='none'))
        axes[i].text(x1, y1 - 5, '%s %.2f [%s]' % (CLASSES[c], cf, s),
                     color=CLASS_COLORS[c], fontsize=7, fontweight='bold')
    axes[i].axis('off'); axes[i].set_title(q.name, fontsize=8)
plt.suptitle('RT-DETR Predictions with Severity (unseen test)')
plt.tight_layout(); plt.savefig(RUNS_DIR / 'prediction_samples_rtdetr.png', dpi=120); plt.show()
print('\nRT-DETR done. Compare in runs/model_comparison.csv (now 4 models).')
