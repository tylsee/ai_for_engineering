"""Train-only crop augmentation for the weak classes (cracks, corrosion, paint_degradation).

For each TRAIN image with a weak-class box, crop a padded window around the box so the small defect
fills more of the frame, recompute the YOLO labels for boxes inside the window, and save it as an
extra training image. Originals are kept; val/test are never touched. Default is a dry run.

Usage:
    python scripts/crop_augment_weak_classes.py --data data_v3            # dry run (counts only)
    python scripts/crop_augment_weak_classes.py --data data_v3 --apply    # write crops
Idempotent: clears previous *_crop* train files before applying.
"""
import argparse
import random
from collections import Counter
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
CLASSES = ['cracks', 'spalling', 'corrosion', 'potholes', 'paint_degradation']
WEAK_CLASSES = {0, 2, 4}          # cracks, corrosion, paint_degradation
IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

MAX_CROPS_PER_IMAGE = 2
MAX_CROPS_PER_CLASS = 800
MIN_CROP_FRAC = 0.35              # crop window is at least this fraction of the image side
PAD_FRAC = 0.5                    # pad around the target box by this fraction of its size
MIN_VIS = 0.3                     # keep a box only if >= this fraction of its area is inside the crop


def read_yolo_label(lbl):
    rows = []
    if lbl.exists():
        for ln in lbl.read_text().splitlines():
            p = ln.split()
            if len(p) == 5:
                rows.append([int(p[0])] + [float(v) for v in p[1:]])
    return rows


def write_yolo_label(lbl, rows):
    lbl.write_text(''.join(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n" for c, cx, cy, w, h in rows))


def _crop_window(cx, cy, w, h, W, H):
    """Pixel crop box around a YOLO box: padded, floored to MIN_CROP_FRAC of the image side."""
    cw = max(w * W * (1 + 2 * PAD_FRAC), MIN_CROP_FRAC * W)
    ch = max(h * H * (1 + 2 * PAD_FRAC), MIN_CROP_FRAC * H)
    x1 = int(max(0, min(cx * W - cw / 2, W - cw)))
    y1 = int(max(0, min(cy * H - ch / 2, H - ch)))
    x2 = int(min(W, x1 + cw))
    y2 = int(min(H, y1 + ch))
    return x1, y1, x2, y2


def _reproject(rows, x1, y1, x2, y2, W, H):
    """Recompute YOLO boxes relative to a pixel crop; keep boxes with >= MIN_VIS area inside."""
    cw, ch = x2 - x1, y2 - y1
    out = []
    for c, cx, cy, w, h in rows:
        bx1, by1, bx2, by2 = (cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H
        ix1, iy1, ix2, iy2 = max(bx1, x1), max(by1, y1), min(bx2, x2), min(by2, y2)
        iw, ih = ix2 - ix1, iy2 - iy1
        orig = (bx2 - bx1) * (by2 - by1)
        if iw <= 0 or ih <= 0 or orig <= 0 or (iw * ih) / orig < MIN_VIS:
            continue
        out.append([c, ((ix1 + ix2) / 2 - x1) / cw, ((iy1 + iy2) / 2 - y1) / ch, iw / cw, ih / ch])
    return out


def crop_weak_classes(data_dir, apply=False, seed=42):
    img_dir = data_dir / 'images' / 'train'
    lbl_dir = data_dir / 'labels' / 'train'

    # idempotent: remove previous crops first
    removed = 0
    for p in list(img_dir.glob('*_crop*')) + list(lbl_dir.glob('*_crop*')):
        removed += 1
        if apply:
            p.unlink()

    rng = random.Random(seed)
    lbls = [l for l in lbl_dir.glob('*.txt') if '_crop' not in l.stem]
    rng.shuffle(lbls)
    per_class, made = Counter(), 0
    for lbl in lbls:
        rows = read_yolo_label(lbl)
        weak = [r for r in rows if r[0] in WEAK_CLASSES]
        if not weak:
            continue
        img = next((img_dir / (lbl.stem + e) for e in IMAGE_EXTS if (img_dir / (lbl.stem + e)).exists()), None)
        if img is None:
            continue
        rng.shuffle(weak)
        here = 0
        for (c, cx, cy, w, h) in weak:
            if here >= MAX_CROPS_PER_IMAGE or per_class[c] >= MAX_CROPS_PER_CLASS:
                continue
            try:
                im = Image.open(img).convert('RGB')
            except Exception:
                break
            W, H = im.size
            x1, y1, x2, y2 = _crop_window(cx, cy, w, h, W, H)
            new_rows = _reproject(rows, x1, y1, x2, y2, W, H)
            if not new_rows:
                continue
            made += 1
            here += 1
            per_class[c] += 1
            if apply:
                stem = f"{lbl.stem}_crop{here}"
                im.crop((x1, y1, x2, y2)).save(img_dir / (stem + '.jpg'), 'JPEG', quality=95)
                write_yolo_label(lbl_dir / (stem + '.txt'), new_rows)
    return removed, made, per_class


def count_boxes(data_dir):
    out = {}
    for s in ('train', 'val', 'test'):
        c, n = Counter(), 0
        d = data_dir / 'labels' / s
        if d.exists():
            for lbl in d.glob('*.txt'):
                n += 1
                for ln in lbl.read_text().splitlines():
                    p = ln.split()
                    if len(p) == 5:
                        c[int(p[0])] += 1
        out[s] = (n, c)
    return out


def print_counts(title, counts):
    print(f"\n{title}")
    print(f"  {'split':<6}{'images':>8}  " + "".join(f"{n[:5]:>9}" for n in CLASSES))
    for s in ('train', 'val', 'test'):
        n, c = counts[s]
        print(f"  {s:<6}{n:>8}  " + "".join(f"{c[i]:>9}" for i in range(len(CLASSES))))


def main():
    ap = argparse.ArgumentParser(description='Train-only crop augmentation for weak classes.')
    ap.add_argument('--data', default='data', help='dataset dir under the project root (default: data)')
    ap.add_argument('--apply', action='store_true', help='write crop files (default: dry run)')
    args = ap.parse_args()
    data_dir = ROOT / args.data

    before = count_boxes(data_dir)
    print_counts('Per-class box counts BEFORE crop-aug:', before)

    removed, made, per_class = crop_weak_classes(data_dir, apply=args.apply)
    mode = 'APPLIED' if args.apply else 'DRY RUN (no files written)'
    cls_summary = ', '.join(f"{CLASSES[c]}={n}" for c, n in sorted(per_class.items())) or 'none'
    print(f"\n{mode}: {'cleared' if args.apply else 'would clear'} {removed} old crop files; "
          f"{'wrote' if args.apply else 'would write'} {made} crops ({cls_summary})")

    after = count_boxes(data_dir)
    print_counts('Per-class box counts AFTER crop-aug:', after)
    for s in ('val', 'test'):
        assert before[s] == after[s], f"{s} split changed! {before[s]} -> {after[s]}"
    print("\nval/test unchanged (train-only). OK")


if __name__ == '__main__':
    main()
