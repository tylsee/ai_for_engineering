"""Contact sheets for weak-class label review (cracks, corrosion, paint_degradation).

Samples labelled TRAIN boxes per class and renders a grid of padded crops, each tagged with the
source image, box area %, and aspect ratio, so labels can be eyeballed before crop augmentation.

Usage:
    python scripts/audit_class_samples.py --data data_v3 --tag before
    python scripts/audit_class_samples.py --data data_v3 --tag after
Sheets are saved to runs/audit_v3/contact_<class>_<tag>.png
"""
import argparse
import random
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
CLASSES = ['cracks', 'spalling', 'corrosion', 'potholes', 'paint_degradation']
WEAK = {'cracks': 0, 'corrosion': 2, 'paint_degradation': 4}
IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')


def read_yolo_label(lbl):
    rows = []
    if lbl.exists():
        for ln in lbl.read_text().splitlines():
            p = ln.split()
            if len(p) == 5:
                rows.append((int(p[0]), *map(float, p[1:])))
    return rows


def _image_for(lbl, img_dir):
    for ext in IMAGE_EXTS:
        p = img_dir / (lbl.stem + ext)
        if p.exists():
            return p
    return None


def create_contact_sheet(data_dir, cls_name, cid, tag, n=60, cols=10, pad_frac=0.5, seed=42):
    img_dir = data_dir / 'images' / 'train'
    lbl_dir = data_dir / 'labels' / 'train'
    samples = []
    for lbl in sorted(lbl_dir.glob('*.txt')):
        img = _image_for(lbl, img_dir)
        if img is None:
            continue
        for (c, cx, cy, w, h) in read_yolo_label(lbl):
            if c == cid:
                samples.append((img, cx, cy, w, h))
    if not samples:
        print(f"  {cls_name}: no boxes found in {lbl_dir}")
        return None
    random.Random(seed).shuffle(samples)
    samples = samples[:n]

    rows = (len(samples) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.7))
    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    for ax in axes:
        ax.axis('off')
    for i, (img, cx, cy, w, h) in enumerate(samples):
        im = Image.open(img).convert('RGB')
        W, H = im.size
        bw, bh = w * W, h * H
        x1 = max(0, int((cx - w / 2) * W - bw * pad_frac))
        y1 = max(0, int((cy - h / 2) * H - bh * pad_frac))
        x2 = min(W, int((cx + w / 2) * W + bw * pad_frac))
        y2 = min(H, int((cy + h / 2) * H + bh * pad_frac))
        ar = bw / bh if bh > 0 else 0.0
        axes[i].imshow(im.crop((x1, y1, x2, y2)))
        axes[i].set_title(f"{img.stem}\nA={w * h * 100:.2f}% AR={ar:.1f}", fontsize=5)

    out_dir = ROOT / 'runs' / 'audit_v3'
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"contact_{cls_name}_{tag}.png"
    plt.suptitle(f"{cls_name} - {len(samples)} sampled train boxes ({tag})", fontsize=10)
    plt.tight_layout()
    plt.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  {cls_name}: {len(samples)} boxes -> {out}")
    return out


def main():
    ap = argparse.ArgumentParser(description='Contact sheets for weak-class label review.')
    ap.add_argument('--data', default='data', help='dataset dir under the project root (default: data)')
    ap.add_argument('--tag', default='before', help='filename tag, e.g. before / after')
    ap.add_argument('--n', type=int, default=60, help='boxes sampled per class (default: 60)')
    args = ap.parse_args()
    data_dir = ROOT / args.data
    print(f"Contact sheets from {data_dir} (tag={args.tag}) -> runs/audit_v3/")
    for name, cid in WEAK.items():
        create_contact_sheet(data_dir, name, cid, args.tag, n=args.n)


if __name__ == '__main__':
    main()
