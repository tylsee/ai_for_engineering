"""
Reorganizes the three source datasets into a unified 5-class YOLO detection dataset.

Target class mapping:
    0: cracks
    1: spalling
    2: corrosion
    3: potholes
    4: paint_degradation

Source datasets:
  - wall-crack-hole-normal: crack(1) -> cracks(0)
  - road-damage-potholes-cracks: Pothole(0)->potholes(3), Crack(1)->cracks(0)
  - concrete-structural-defect (classification): whole-image bboxes for
    spalling->spalling(1), stain->corrosion(2), peeling->paint_degradation(4),
    major_crack->cracks(0), minor_crack->cracks(0)

Future datasets: add a new process_*() function following the same pattern.
All functions return a list of (Path, list[str]) tuples — (image_path, yolo_label_lines).
"""

import shutil
import random
import csv
from pathlib import Path

import yaml

CLASSES = ['cracks', 'spalling', 'corrosion', 'potholes', 'paint_degradation']
# Supported image extensions (case-insensitive on Windows NTFS)
IMAGE_EXTS = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# TEST_RATIO = 0.15 (remainder)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / 'data'
DATASET_DIR  = PROJECT_ROOT / 'dataset'


def _iter_images(directory: Path):
    """Yields all image files under directory, de-duplicated by lowercase stem+ext."""
    seen = set()
    for ext in IMAGE_EXTS:
        for img in directory.glob(ext):
            key = img.name.lower()
            if key not in seen:
                seen.add(key)
                yield img


def _split(items: list, seed: int = 42) -> dict:
    rng = random.Random(seed)
    rng.shuffle(items)
    n = len(items)
    t = int(n * TRAIN_RATIO)
    v = int(n * (TRAIN_RATIO + VAL_RATIO))
    return {'train': items[:t], 'val': items[t:v], 'test': items[v:]}


def _save_split(items: list[tuple], prefix: str) -> dict[str, int]:
    """Copies images and writes remapped label files. Returns per-split counts."""
    splits = _split(items)
    counts = {}
    for split, split_items in splits.items():
        counts[split] = len(split_items)
        img_out = DATA_DIR / 'images' / split
        lbl_out = DATA_DIR / 'labels' / split
        for idx, (img_path, label_lines) in enumerate(split_items):
            stem = f"{prefix}_{idx:05d}"
            shutil.copy2(img_path, img_out / (stem + img_path.suffix.lower()))
            (lbl_out / (stem + '.txt')).write_text(''.join(label_lines))
    return counts


def process_wall_crack() -> list[tuple]:
    """
    wall-crack-hole-normal  (original classes: normal=0, crack=1, hole=2)
    Keep only images that have at least one crack(1) bbox → remap to cracks(0).
    Supports .jpg and .png images.
    """
    base = DATASET_DIR / 'wall-crack-hole-normal' / 'Crack_Hole_Normal_Dataset'
    items = []
    for split in ('train', 'test'):
        img_dir = base / 'images' / split
        lbl_dir = base / 'labels' / split
        for img in _iter_images(img_dir):
            lbl = lbl_dir / (img.stem + '.txt')
            if not lbl.exists():
                continue
            lines = []
            for raw in lbl.read_text().splitlines():
                parts = raw.strip().split()
                if parts and int(parts[0]) == 1:        # crack → cracks(0)
                    lines.append(f"0 {' '.join(parts[1:])}\n")
            if lines:
                items.append((img, lines))
    return items


def process_road_damage() -> list[tuple]:
    """
    road-damage-potholes-cracks  (original classes: Pothole=0, Crack=1, Manhole=2)
    Remap: Pothole→potholes(3), Crack→cracks(0).  Discard Manhole-only images.
    """
    base    = DATASET_DIR / 'road-damage-potholes-cracks' / 'data'
    img_dir = base / 'images'
    lbl_dir = base / 'labels-YOLO'
    remap   = {0: 3, 1: 0}     # Pothole→3, Crack→0
    items   = []
    for img in _iter_images(img_dir):
        lbl = lbl_dir / (img.stem + '.txt')
        if not lbl.exists():
            continue
        lines = []
        for raw in lbl.read_text().splitlines():
            parts = raw.strip().split()
            if not parts:
                continue
            cls = int(parts[0])
            if cls in remap:
                lines.append(f"{remap[cls]} {' '.join(parts[1:])}\n")
        if lines:
            items.append((img, lines))
    return items


def process_concrete_structural() -> list[tuple]:
    """
    concrete-structural-defect (image classification format → object detection).
    Whole-image bounding box (cx=0.5, cy=0.5, w=1.0, h=1.0) per image.

    Folder → class mapping:
        spalling    → spalling(1)
        stain       → corrosion(2)
        peeling     → paint_degradation(4)
        major_crack → cracks(0)
        minor_crack → cracks(0)
    """
    base = DATASET_DIR / 'concrete-structural-defect' / 'Building_Dataset'
    folder_class = {
        'spalling':    1,
        'stain':       2,
        'peeling':     4,
        'major_crack': 0,
        'minor_crack': 0,
    }
    items = []
    for folder, cls_id in folder_class.items():
        folder_path = base / folder
        if not folder_path.exists():
            print(f"  WARNING: folder not found: {folder_path}")
            continue
        for img in _iter_images(folder_path):
            items.append((img, [f"{cls_id} 0.5 0.5 1.0 1.0\n"]))
    return items


# ── Future dataset hook ─────────────────────────────────────────────────────
# def process_my_new_dataset() -> list[tuple]:
#     """
#     Add new dataset processing here.
#     Return list of (Path, [yolo_label_lines]).
#     """
#     ...


def write_data_yaml():
    config = {
        'path': str(DATA_DIR).replace('\\', '/'),
        'train': 'images/train',
        'val':   'images/val',
        'test':  'images/test',
        'nc':    len(CLASSES),
        'names': CLASSES,
    }
    out = DATA_DIR / 'data.yaml'
    with open(out, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"  Written: {out}")


def write_manifest(summary: list[dict]):
    out = DATA_DIR / 'manifest.csv'
    with open(out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['source', 'total', 'train', 'val', 'test'])
        writer.writeheader()
        writer.writerows(summary)
    print(f"  Written: {out}")


def class_bbox_counts(label_dir: Path) -> dict[str, int]:
    counts = {c: 0 for c in CLASSES}
    for lbl in label_dir.glob('*.txt'):
        for line in lbl.read_text().splitlines():
            parts = line.strip().split()
            if parts:
                cls = int(parts[0])
                if 0 <= cls < len(CLASSES):
                    counts[CLASSES[cls]] += 1
    return counts


def clear_data_split_dirs():
    """Removes all existing images and labels before a fresh reorganisation."""
    for split in ('train', 'val', 'test'):
        for folder in ('images', 'labels'):
            d = DATA_DIR / folder / split
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)
    print("  Cleared and recreated data/images/* and data/labels/*")


def main():
    print("Clearing existing data/…")
    clear_data_split_dirs()

    summary = []

    print("\n=== Wall-Crack dataset ===")
    wc = process_wall_crack()
    counts = _save_split(wc, 'wc')
    summary.append({'source': 'wall_crack', 'total': len(wc), **counts})
    print(f"  Images used: {len(wc)}  |  {counts}")

    print("\n=== Road-Damage dataset ===")
    rd = process_road_damage()
    counts = _save_split(rd, 'rd')
    summary.append({'source': 'road_damage', 'total': len(rd), **counts})
    print(f"  Images used: {len(rd)}  |  {counts}")

    print("\n=== Concrete-Structural dataset ===")
    cs = process_concrete_structural()
    counts = _save_split(cs, 'cs')
    summary.append({'source': 'concrete_structural', 'total': len(cs), **counts})
    print(f"  Images used: {len(cs)}  |  {counts}")

    # ── Add future datasets here ─────────────────────────────────────────────
    # print("\n=== My New Dataset ===")
    # nd = process_my_new_dataset()
    # counts = _save_split(nd, 'nd')
    # summary.append({'source': 'new_dataset', 'total': len(nd), **counts})

    total = sum(s['total'] for s in summary)
    print(f"\n=== Total images reorganised: {total} ===")

    write_data_yaml()
    write_manifest(summary)

    print("\n=== Class distribution (train labels) ===")
    for cls, n in class_bbox_counts(DATA_DIR / 'labels' / 'train').items():
        print(f"  {cls:<20}: {n:>5} bounding boxes")

    print("\n=== Split totals ===")
    for split in ('train', 'val', 'test'):
        ni = len(list((DATA_DIR / 'images' / split).glob('*')))
        nl = len(list((DATA_DIR / 'labels' / split).glob('*.txt')))
        print(f"  {split}: {ni} images, {nl} label files")


if __name__ == '__main__':
    main()
