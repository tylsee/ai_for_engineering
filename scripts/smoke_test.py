"""Quick smoke test — validates the full pipeline before the long training run."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from models.ssdlite_detector import build_model as build_ssd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / 'data'
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES  = 6
USE_AMP      = torch.cuda.is_available()
IMG_SIZE     = 640
IMAGE_EXTS   = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']

print(f"Device: {DEVICE}  AMP: {USE_AMP}")


def _iter_images(directory):
    seen = set()
    for ext in IMAGE_EXTS:
        for img in Path(directory).glob(ext):
            key = img.name.lower()
            if key not in seen:
                seen.add(key)
                yield img


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


# ── Dataset ───────────────────────────────────────────────────────────────────
train_ds = DefectDataset(DATA_DIR/'images'/'train', DATA_DIR/'labels'/'train')
val_ds   = DefectDataset(DATA_DIR/'images'/'val',   DATA_DIR/'labels'/'val')
tl = DataLoader(train_ds, batch_size=2, shuffle=False, collate_fn=collate_fn, num_workers=0)
vl = DataLoader(val_ds,   batch_size=2, shuffle=False, collate_fn=collate_fn, num_workers=0)
imgs, targets = next(iter(tl))
print(f"Train batches: {len(tl)}, Val batches: {len(vl)}")
print(f"Sample target boxes shape: {targets[0]['boxes'].shape}")

# ── Model build ───────────────────────────────────────────────────────────────
ssd = build_ssd(num_classes=NUM_CLASSES, pretrained=False).to(DEVICE)
sp  = sum(p.numel() for p in ssd.parameters()) / 1e6
print(f"SSDLite320-MobileNetV3: {sp:.1f}M params")

# ── Forward pass ──────────────────────────────────────────────────────────────
test_imgs = [i.to(DEVICE) for i in imgs]
test_tgts = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

ssd.train()
with torch.amp.autocast('cuda', enabled=USE_AMP):
    loss_ssd = sum(ssd(test_imgs, test_tgts).values())
print(f"SSDLite loss: {loss_ssd.item():.4f}")

print("\nAll smoke tests passed!")
