"""
Full training pipeline: YOLOv11n → YOLOv8n → SSDLite320-MobileNetV3.
Each run auto-increments a version: v1, v2, v3, ...
Run: python scripts/train_all.py
"""

import csv, sys, time
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / 'data'
RUNS_DIR     = PROJECT_ROOT / 'runs'
sys.path.insert(0, str(PROJECT_ROOT))

DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASSES     = ['cracks', 'spalling', 'corrosion', 'potholes', 'paint_degradation']
NUM_CLASSES = len(CLASSES) + 1   # +1 for background
DATA_YAML   = str(DATA_DIR / 'data.yaml')

# Rewrite data.yaml with the correct absolute path for this machine.
# The committed data.yaml may contain a hardcoded path from a different OS/machine.
def _write_data_yaml():
    content = (
        f"path: {DATA_DIR.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"nc: {len(CLASSES)}\n"
        f"names: {CLASSES}\n"
    )
    (DATA_DIR / 'data.yaml').write_text(content)

_write_data_yaml()

YOLO_EPOCHS  = 100
TORCH_EPOCHS = 30
YOLO_BATCH   = 8
TORCH_BATCH  = 2
LR           = 1e-4
IMG_SIZE     = 640
USE_AMP      = torch.cuda.is_available()

# ── Versioning ────────────────────────────────────────────────────────────────

def next_yolo_version(model_name: str) -> int:
    """Auto-increment: returns next unused vN folder under runs/{model_name}/."""
    d = RUNS_DIR / model_name
    d.mkdir(parents=True, exist_ok=True)
    v = 1
    while (d / f'v{v}').exists():
        v += 1
    return v


def next_torch_version(model_name: str) -> int:
    """Auto-increment: returns next unused best_vN.pth under runs/{model_name}/."""
    d = RUNS_DIR / model_name
    d.mkdir(parents=True, exist_ok=True)
    v = 1
    while (d / f'best_v{v}.pth').exists():
        v += 1
    return v


# ── Image helpers ─────────────────────────────────────────────────────────────
IMAGE_EXTS = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']


def _iter_images(directory):
    seen = set()
    for ext in IMAGE_EXTS:
        for img in Path(directory).glob(ext):
            key = img.name.lower()
            if key not in seen:
                seen.add(key)
                yield img


# ── Dataset ───────────────────────────────────────────────────────────────────
class DefectDataset(Dataset):
    def __init__(self, img_dir, lbl_dir, img_size=IMG_SIZE):
        self.img_dir  = Path(img_dir)
        self.lbl_dir  = Path(lbl_dir)
        self.img_size = img_size
        self.images   = sorted(_iter_images(self.img_dir))

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        lbl_path = self.lbl_dir / (img_path.stem + '.txt')
        try:
            img = Image.open(img_path).convert('RGB').resize((self.img_size, self.img_size))
            tensor = TF.to_tensor(img)
        except Exception:
            # Return blank image with no boxes if file is corrupted
            tensor = torch.zeros(3, self.img_size, self.img_size)
            return tensor, {'boxes': torch.zeros((0,4), dtype=torch.float32),
                            'labels': torch.zeros(0, dtype=torch.int64),
                            'image_id': torch.tensor([idx]),
                            'area': torch.zeros(0, dtype=torch.float32),
                            'iscrowd': torch.zeros(0, dtype=torch.int64)}
        boxes, labels = [], []
        if lbl_path.exists():
            for line in lbl_path.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) != 5: continue
                cls, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:])
                x1 = max(0.0, (cx - bw/2) * self.img_size)
                y1 = max(0.0, (cy - bh/2) * self.img_size)
                x2 = min(float(self.img_size), (cx + bw/2) * self.img_size)
                y2 = min(float(self.img_size), (cy + bh/2) * self.img_size)
                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2]); labels.append(cls + 1)
        if boxes:
            boxes_t  = torch.tensor(boxes,  dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.int64)
            area     = (boxes_t[:,3]-boxes_t[:,1]) * (boxes_t[:,2]-boxes_t[:,0])
        else:
            boxes_t  = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros(0,      dtype=torch.int64)
            area     = torch.zeros(0,      dtype=torch.float32)
        return tensor, {'boxes': boxes_t, 'labels': labels_t,
                        'image_id': torch.tensor([idx]), 'area': area,
                        'iscrowd': torch.zeros(len(labels_t), dtype=torch.int64)}


def collate_fn(batch): return tuple(zip(*batch))


def make_loader(split, batch_size=TORCH_BATCH, shuffle=False):
    ds = DefectDataset(DATA_DIR/'images'/split, DATA_DIR/'labels'/split)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate_fn, num_workers=0, drop_last=shuffle)


# ── Training helpers ──────────────────────────────────────────────────────────
from torchmetrics.detection.mean_ap import MeanAveragePrecision


def train_epoch(model, optimizer, loader):
    model.train()
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)
    total  = 0.0
    for images, targets in tqdm(loader, desc='  train', leave=False):
        images  = [i.to(DEVICE) for i in images]
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=USE_AMP):
            losses = sum(model(images, targets).values())
        scaler.scale(losses).backward(); scaler.step(optimizer); scaler.update()
        total += losses.item()
    return total / len(loader)


@torch.no_grad()
def evaluate_model(model, loader):
    model.eval()
    metric = MeanAveragePrecision(iou_type='bbox', class_metrics=True)
    for images, targets in tqdm(loader, desc='  eval ', leave=False):
        images  = [i.to(DEVICE) for i in images]
        outputs = model(images)
        preds = [{'boxes': o['boxes'].cpu(), 'scores': o['scores'].cpu(),
                  'labels': o['labels'].cpu()} for o in outputs]
        gts   = [{'boxes': t['boxes'].cpu(), 'labels': t['labels'].cpu()} for t in targets]
        metric.update(preds, gts)
    torch.cuda.empty_cache()
    return metric.compute()


def save_ckpt(model, optimizer, epoch, metrics, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 'metrics': metrics}, path)


def log_run(name, version, epochs, map50, map50_95, prec, rec, mins, params_m, weights):
    row = [f"{name}_v{version}_{int(time.time())}", name, version, epochs,
           TORCH_BATCH, IMG_SIZE, LR,
           round(map50, 4), round(map50_95, 4), round(prec, 4), round(rec, 4),
           round(mins, 1), round(params_m, 2), weights, '']
    RUNS_DIR.mkdir(exist_ok=True)
    log_path = RUNS_DIR / 'run_log.csv'
    write_header = not log_path.exists()
    with open(log_path, 'a', newline='') as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(['run_id','model','version','epochs','batch_size','img_size','lr',
                        'map50','map50_95','precision','recall','train_time_min','params_M',
                        'weights_path','notes'])
        w.writerow(row)
    print(f"  Logged: {name} v{version}")


def train_torch_model(model, model_name, train_loader, val_loader):
    """Two-phase training: freeze backbone epochs 1-5, unfreeze at epoch 6."""
    version  = next_torch_version(model_name)
    ckpt_dir = RUNS_DIR / model_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = ckpt_dir / f'best_v{version}.pth'

    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"{model_name} params: {params_m:.1f}M -> saving to {best_ckpt.name}")

    # Phase 1: freeze backbone
    for p in model.backbone.parameters(): p.requires_grad = False
    opt   = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                               lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=5)

    best50 = 0.0
    t0     = time.time()
    for epoch in range(1, TORCH_EPOCHS + 1):
        if epoch == 6:
            # Phase 2: unfreeze backbone, lower LR
            for p in model.backbone.parameters(): p.requires_grad = True
            opt   = torch.optim.AdamW(model.parameters(), lr=LR/10, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=TORCH_EPOCHS - 5)

        loss = train_epoch(model, opt, train_loader)
        sched.step()
        m   = evaluate_model(model, val_loader)
        m50 = float(m['map_50'])
        print(f"  Epoch {epoch:3d}/{TORCH_EPOCHS} | loss={loss:.4f} | mAP@0.5={m50:.4f}")
        if m50 > best50:
            best50 = m50
            save_ckpt(model, opt, epoch, m, best_ckpt)

    elapsed = (time.time() - t0) / 60
    final_m = evaluate_model(model, val_loader)
    log_run(model_name, version, TORCH_EPOCHS,
            float(final_m['map_50']), float(final_m['map']),
            0.0, 0.0, elapsed, params_m, str(best_ckpt))
    print(f"{model_name} done: best mAP@0.5={best50:.4f}  ({elapsed:.0f} min)\n")
    return best50


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train defect detection models.')
    parser.add_argument(
        '--model', nargs='+',
        choices=['yolo11n', 'yolov8n', 'ssdlite', 'all'],
        default=['all'],
        help='Which model(s) to train. Examples:\n'
             '  python train_all.py                       # train all 3\n'
             '  python train_all.py --model yolo11n       # only YOLOv11n\n'
             '  python train_all.py --model yolov8n ssdlite  # skip YOLOv11n'
    )
    args  = parser.parse_args()
    train = set(args.model)
    if 'all' in train:
        train = {'yolo11n', 'yolov8n', 'ssdlite'}

    print(f"Device : {DEVICE}")
    print(f"AMP    : {USE_AMP}")
    print(f"Data   : {DATA_YAML}")
    print(f"Models : {', '.join(sorted(train))}\n")

    # ── 1. YOLOv11n ───────────────────────────────────────────────────────────
    if 'yolo11n' in train:
        print("=" * 60)
        print("STEP: YOLOv11n Training")
        print("=" * 60)

        from ultralytics import YOLO

        v11_ver = next_yolo_version('yolo11n')
        print(f"YOLOv11n → saving to runs/yolo11n/v{v11_ver}")
        yolo11 = YOLO('yolo11n.pt')
        t0     = time.time()
        yolo11.train(data=DATA_YAML, epochs=YOLO_EPOCHS, imgsz=IMG_SIZE,
                     batch=YOLO_BATCH, lr0=LR,
                     device=0 if torch.cuda.is_available() else 'cpu',
                     project=str(RUNS_DIR/'yolo11n'), name=f'v{v11_ver}',
                     exist_ok=False, save=True, amp=True, workers=0, verbose=False)
        yolo11_min = (time.time() - t0) / 60

        yolo11_val = yolo11.val(data=DATA_YAML, verbose=False)
        yolo11_w   = str(RUNS_DIR/'yolo11n'/f'v{v11_ver}'/'weights'/'best.pt')
        yolo11_p   = sum(p.numel() for p in yolo11.model.parameters()) / 1e6
        log_run('YOLOv11n', v11_ver, YOLO_EPOCHS,
                float(yolo11_val.box.map50), float(yolo11_val.box.map),
                float(yolo11_val.box.mp),   float(yolo11_val.box.mr),
                yolo11_min, yolo11_p, yolo11_w)
        print(f"YOLOv11n done: mAP@0.5={yolo11_val.box.map50:.4f}  ({yolo11_min:.0f} min)\n")


    # ── 2. YOLOv8n ────────────────────────────────────────────────────────────
    if 'yolov8n' in train:
        print("=" * 60)
        print("STEP: YOLOv8n Training")
        print("=" * 60)

        from ultralytics import YOLO

        v8_ver = next_yolo_version('yolov8n')
        print(f"YOLOv8n → saving to runs/yolov8n/v{v8_ver}")
        yolo8  = YOLO('yolov8n.pt')
        t0     = time.time()
        yolo8.train(data=DATA_YAML, epochs=YOLO_EPOCHS, imgsz=IMG_SIZE,
                    batch=YOLO_BATCH, lr0=LR,
                    device=0 if torch.cuda.is_available() else 'cpu',
                    project=str(RUNS_DIR/'yolov8n'), name=f'v{v8_ver}',
                    exist_ok=False, save=True, amp=True, workers=0, verbose=False)
        yolo8_min = (time.time() - t0) / 60

        yolo8_val = yolo8.val(data=DATA_YAML, verbose=False)
        yolo8_w   = str(RUNS_DIR/'yolov8n'/f'v{v8_ver}'/'weights'/'best.pt')
        yolo8_p   = sum(p.numel() for p in yolo8.model.parameters()) / 1e6
        log_run('YOLOv8n', v8_ver, YOLO_EPOCHS,
                float(yolo8_val.box.map50), float(yolo8_val.box.map),
                float(yolo8_val.box.mp),   float(yolo8_val.box.mr),
                yolo8_min, yolo8_p, yolo8_w)
        print(f"YOLOv8n done: mAP@0.5={yolo8_val.box.map50:.4f}  ({yolo8_min:.0f} min)\n")


    # ── 3. SSDLite320-MobileNetV3 ─────────────────────────────────────────────
    if 'ssdlite' in train:
        print("=" * 60)
        print("STEP: SSDLite320-MobileNetV3 Training")
        print("=" * 60)

        from models.ssdlite_detector import build_model as build_ssd

        train_loader = make_loader('train', shuffle=True)
        val_loader   = make_loader('val')

        ssd = build_ssd(num_classes=NUM_CLASSES, pretrained=True).to(DEVICE)
        train_torch_model(ssd, 'ssdlite', train_loader, val_loader)


    # ── Summary ───────────────────────────────────────────────────────────────
    import pandas as pd
    print("=" * 60)
    print("TRAINING COMPLETE — Run Log:")
    print("=" * 60)
    if (RUNS_DIR / 'run_log.csv').exists():
        df   = pd.read_csv(RUNS_DIR/'run_log.csv')
        available = ['model', 'backbone', 'epochs', 'map50', 'map50_95', 'train_time_min', 'params_M']
        cols = [c for c in available if c in df.columns]
        print(df[cols].to_string(index=False))
    else:
        print("No completed runs in run_log.csv yet.")
