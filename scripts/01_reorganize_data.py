"""
Reorganises all source datasets into a unified, **deduplicated, class-balanced**
5-class YOLO detection dataset.

Target class mapping:
    0: cracks
    1: spalling
    2: corrosion
    3: potholes
    4: paint_degradation

Pipeline (run order inside main()):
    1. COLLECT  — every source processor returns real-bbox (image, label_lines)
                  items. Roboflow folders are read across ALL their splits
                  (train/valid/test) because we re-split globally.
    2. DEDUP    — a 64-bit perceptual average-hash (a-hash) signature removes
                  duplicate images WITHIN and ACROSS sources. Critical: the
                  pothole Roboflow folders re-upload the same photos many times;
                  without dedup an identical image would leak across the
                  train/test split boundary and inflate mAP.
    3. BALANCE  — cap each class at TARGET_PER_CLASS *bounding boxes* (instances,
                  the metric that drives detector learning). Multi-class images
                  are kept first (rubric requires multi-class detection and they
                  are scarce), then single-class images top each class up to the
                  cap. Classes below the cap (spalling) use all available data.
    4. SPLIT    — stratified 70/15/15 by each image's rarest class, so every
                  class is represented in train/val/test. Because dedup runs
                  first, no image can appear in two splits (no leakage).
    5. SAVE     — copy images (extension preserved) + write remapped labels with
                  a global index; emit data.yaml, manifest.csv, class report.

DESIGN NOTES (for the report / interview):
  - Whole-image proxy boxes (cx=.5 cy=.5 w=1 h=1) from concrete-structural-defect
    are RETIRED. We now have real localized boxes for spalling/corrosion/paint,
    so the old proxies (which produced inflated ~0.95 AP via trivial IoU) are
    dropped. process_concrete_structural() is kept for reference but not called
    unless --keep-proxies is passed.
  - Balancing is by instance (bbox) count, not image count, because a single
    crack image carries many boxes; image-capping would leave cracks dominant.

Add a future dataset: write a process_*() returning list[Item], append it to
collect_items(), then re-run. Idempotent: clears and rebuilds data/ each run.
"""

import argparse
import csv
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from PIL import Image

CLASSES = ['cracks', 'spalling', 'corrosion', 'potholes', 'paint_degradation']
IMAGE_EXTS = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
IMG_EXT_SET = {e.lstrip('*') for e in IMAGE_EXTS}

TARGET_PER_CLASS = 2000          # cap per class, measured in bounding boxes
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15                  # test = remainder (0.15)
SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
DATASET_DIR = PROJECT_ROOT / 'dataset'


@dataclass
class Item:
    """One image plus its remapped YOLO label lines (target class id space)."""
    img: Path
    lines: list[str]
    source: str
    classes: frozenset = field(default_factory=frozenset)
    sig: int | None = None       # 64-bit perceptual a-hash, filled during dedup

    @property
    def bbox_class_counts(self) -> Counter:
        c = Counter()
        for ln in self.lines:
            c[int(ln.split()[0])] += 1
        return c


# ── Source readers ──────────────────────────────────────────────────────────

def _iter_images(directory: Path):
    """Yields image files under directory, de-duplicated by lowercase name."""
    if not directory.exists():
        return
    seen = set()
    for p in sorted(directory.iterdir()):
        if p.suffix.lower() in IMG_EXT_SET and p.name.lower() not in seen:
            seen.add(p.name.lower())
            yield p


def _read_label(lbl: Path, keep_remap: dict[int, int]) -> list[str]:
    """Reads a YOLO label file, keeping only classes in keep_remap and
    remapping their ids to the target space. Returns [] if nothing kept."""
    if not lbl.exists():
        return []
    out = []
    for raw in lbl.read_text().splitlines():
        parts = raw.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        if cls in keep_remap:
            out.append(f"{keep_remap[cls]} {' '.join(parts[1:5])}\n")
    return out


def _roboflow_items(folder: Path, keep_remap: dict[int, int], source: str) -> list[Item]:
    """Reads a Roboflow YOLO export across train/valid/test splits."""
    items = []
    for split in ('train', 'valid', 'test'):
        img_dir = folder / split / 'images'
        lbl_dir = folder / split / 'labels'
        for img in _iter_images(img_dir):
            lines = _read_label(lbl_dir / (img.stem + '.txt'), keep_remap)
            if lines:
                items.append(Item(img, lines, source))
    return items


def process_wall_crack() -> list[Item]:
    """wall-crack-hole-normal: crack(1) -> cracks(0). Real boxes."""
    base = DATASET_DIR / 'wall-crack-hole-normal' / 'Crack_Hole_Normal_Dataset'
    items = []
    for split in ('train', 'test'):
        img_dir = base / 'images' / split
        lbl_dir = base / 'labels' / split
        for img in _iter_images(img_dir):
            lines = _read_label(lbl_dir / (img.stem + '.txt'), {1: 0})
            if lines:
                items.append(Item(img, lines, 'wall_crack'))
    return items


def process_road_damage() -> list[Item]:
    """road-damage-potholes-cracks: Pothole(0)->3, Crack(1)->0. Multi-class."""
    base = DATASET_DIR / 'road-damage-potholes-cracks' / 'data'
    items = []
    for img in _iter_images(base / 'images'):
        lines = _read_label(base / 'labels-YOLO' / (img.stem + '.txt'), {0: 3, 1: 0})
        if lines:
            items.append(Item(img, lines, 'road_damage'))
    return items


def process_paint_degradation() -> list[Item]:
    """paint-degradation: degradation(0) -> paint_degradation(4). Real boxes."""
    return _roboflow_items(DATASET_DIR / 'paint-degradation', {0: 4}, 'paint_degradation')


def process_corrosion() -> list[Item]:
    """Corrosion: Corrosion(0) -> corrosion(2). Real boxes."""
    return _roboflow_items(DATASET_DIR / 'Corrosion', {0: 2}, 'corrosion')


def process_spalling() -> list[Item]:
    """spalling + spalling2 + spalling3: Spalling(0) -> spalling(1). Real boxes.
    spalling3 (Roboflow yolo-xxm9l/spalling-wcoze) added 3 Jun 2026 to lift the
    spalling class toward the 2000-box target and shrink the imbalance ratio."""
    items = []
    for name in ('spalling', 'spalling2', 'spalling3'):
        items += _roboflow_items(DATASET_DIR / name, {0: 1}, 'spalling')
    return items


def process_potholes_roboflow() -> list[Item]:
    """
    potholesv1 / potholesV3 / PotholesV4: pothole(0) -> potholes(3).
    PotholesV2 is a CHESS dataset where 'pothole' is class 6 — keep only class 6.
    Heavy cross-folder duplication; dedup() removes the copies afterwards.
    """
    items = []
    for name in ('potholesv1', 'potholesV3', 'PotholesV4'):
        items += _roboflow_items(DATASET_DIR / name, {0: 3}, 'pothole_rf')
    items += _roboflow_items(DATASET_DIR / 'PotholesV2', {6: 3}, 'pothole_rf')
    return items


def process_concrete_structural() -> list[Item]:
    """
    RETIRED whole-image proxy source. Only used when --keep-proxies is passed.
    concrete-structural-defect (classification): one full-image box per image.
        spalling->1, stain->corrosion(2), peeling->paint(4), major/minor_crack->0
    """
    base = DATASET_DIR / 'concrete-structural-defect' / 'Building_Dataset'
    folder_class = {'spalling': 1, 'stain': 2, 'peeling': 4,
                    'major_crack': 0, 'minor_crack': 0}
    items = []
    for folder, cls_id in folder_class.items():
        for img in _iter_images(base / folder):
            items.append(Item(img, [f"{cls_id} 0.5 0.5 1.0 1.0\n"], 'concrete_proxy'))
    return items


# ── Perceptual dedup ────────────────────────────────────────────────────────

def _signature(path: Path) -> int | None:
    """64-bit perceptual average-hash. Robust to re-encoding/resizing. Two
    images with an identical 8x8->1-bit average pattern are treated as
    duplicates. Exact-match dedup is intentionally aggressive: across the four
    pothole folders the same photo was re-uploaded many times, and a near-dup
    leaking across the train/test boundary would inflate mAP. Random 64-bit
    collisions between genuinely different photos are statistically impossible
    (~n^2 / 2^65), so an exact match means the same image."""
    try:
        with Image.open(path) as im:
            g = im.convert('L').resize((8, 8), Image.BILINEAR)
            px = list(g.getdata())
        avg = sum(px) / 64.0
        h = 0
        for i, p in enumerate(px):
            if p >= avg:
                h |= (1 << i)
        return h
    except Exception as e:
        print(f"  WARN unreadable image skipped: {path}  ({e})")
        return None


def dedup(items: list[Item]) -> tuple[list[Item], int]:
    """Keeps the first occurrence of each unique signature. First occurrence
    is decided by collect order (multi-class sources listed first win ties)."""
    seen: dict[int, Item] = {}
    kept, removed = [], 0
    for it in items:
        sig = _signature(it.img)
        if sig is None:
            removed += 1
            continue
        it.sig = sig
        if sig in seen:
            removed += 1
            continue
        seen[sig] = it
        kept.append(it)
    return kept, removed


# ── Balancing ───────────────────────────────────────────────────────────────

def balance(items: list[Item], target: int) -> tuple[list[Item], Counter]:
    """Cap each class at `target` bounding boxes. Multi-class images are kept
    first (scarce + required by rubric), then single-class images top up each
    class. A class with fewer than `target` boxes uses everything available."""
    for it in items:
        it.classes = frozenset(it.bbox_class_counts)

    rng = random.Random(SEED)
    counts = Counter()
    selected: list[Item] = []

    multi = [it for it in items if len(it.classes) > 1]
    single_by_cls: dict[int, list[Item]] = defaultdict(list)
    for it in items:
        if len(it.classes) == 1:
            single_by_cls[next(iter(it.classes))].append(it)

    # Phase A — all multi-class images
    for it in multi:
        selected.append(it)
        counts.update(it.bbox_class_counts)

    # Phase B — top up each class with single-class images (rarest class first
    # so scarce classes are filled before abundant ones compete for the budget)
    for cls in sorted(range(len(CLASSES)), key=lambda c: counts[c]):
        pool = single_by_cls.get(cls, [])
        rng.shuffle(pool)
        for it in pool:
            if counts[cls] >= target:
                break
            selected.append(it)
            counts[cls] += len(it.lines)
    return selected, counts


# ── Stratified, leak-free split ─────────────────────────────────────────────

def split_items(items: list[Item]) -> dict[str, list[Item]]:
    """70/15/15 stratified by each image's rarest present class so every class
    appears in all splits. Each image is in exactly one split (no leakage)."""
    rng = random.Random(SEED)
    freq = Counter()
    for it in items:
        freq.update(it.classes)

    groups: dict[int, list[Item]] = defaultdict(list)
    for it in items:
        rarest = min(it.classes, key=lambda c: freq[c])
        groups[rarest].append(it)

    out = {'train': [], 'val': [], 'test': []}
    for cls in sorted(groups):
        g = groups[cls]
        rng.shuffle(g)
        n = len(g)
        t = int(n * TRAIN_RATIO)
        v = int(n * (TRAIN_RATIO + VAL_RATIO))
        out['train'] += g[:t]
        out['val'] += g[t:v]
        out['test'] += g[v:]
    return out


# ── Output ──────────────────────────────────────────────────────────────────

def clear_data_split_dirs():
    for split in ('train', 'val', 'test'):
        for folder in ('images', 'labels'):
            d = DATA_DIR / folder / split
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)
    print("  Cleared and recreated data/images/* and data/labels/*")


def save_splits(splits: dict[str, list[Item]]):
    for split, items in splits.items():
        img_out = DATA_DIR / 'images' / split
        lbl_out = DATA_DIR / 'labels' / split
        for idx, it in enumerate(items):
            stem = f"{split}_{idx:06d}"
            shutil.copy2(it.img, img_out / (stem + it.img.suffix.lower()))
            (lbl_out / (stem + '.txt')).write_text(''.join(it.lines))


def write_data_yaml():
    config = {
        'path': str(DATA_DIR).replace('\\', '/'),
        'train': 'images/train', 'val': 'images/val', 'test': 'images/test',
        'nc': len(CLASSES), 'names': CLASSES,
    }
    with open(DATA_DIR / 'data.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"  Written: {DATA_DIR / 'data.yaml'}")


def write_manifest(splits: dict[str, list[Item]]):
    per_source = defaultdict(lambda: {'train': 0, 'val': 0, 'test': 0})
    for split, items in splits.items():
        for it in items:
            per_source[it.source][split] += 1
    rows = []
    for src, c in sorted(per_source.items()):
        total = c['train'] + c['val'] + c['test']
        rows.append({'source': src, 'total': total, **c})
    with open(DATA_DIR / 'manifest.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['source', 'total', 'train', 'val', 'test'])
        w.writeheader()
        w.writerows(rows)
    print(f"  Written: {DATA_DIR / 'manifest.csv'}")


def report(splits: dict[str, list[Item]]):
    print("\n=== Per-split image counts ===")
    for split in ('train', 'val', 'test'):
        print(f"  {split:<5}: {len(splits[split])} images")

    print("\n=== Bounding-box counts per class per split ===")
    header = f"  {'class':<18}" + "".join(f"{s:>9}" for s in ('train', 'val', 'test', 'TOTAL'))
    print(header)
    grand = Counter()
    per_split_cls = {s: Counter() for s in ('train', 'val', 'test')}
    for split, items in splits.items():
        for it in items:
            per_split_cls[split].update(it.bbox_class_counts)
    for cid, name in enumerate(CLASSES):
        tr, va, te = (per_split_cls[s][cid] for s in ('train', 'val', 'test'))
        tot = tr + va + te
        grand[name] = tot
        print(f"  {name:<18}{tr:>9}{va:>9}{te:>9}{tot:>9}")
    print(f"\n  TOTAL boxes: {sum(grand.values())}")

    # Imbalance ratio (max/min) — the headline balance metric
    vals = [v for v in grand.values() if v]
    if vals:
        print(f"  Balance ratio (max/min class boxes): {max(vals) / min(vals):.2f}x")


def verify_no_leakage(splits: dict[str, list[Item]]):
    """Hard assert: no perceptual signature appears in more than one split."""
    where = {}
    leaks = 0
    for split, items in splits.items():
        for it in items:
            if it.sig in where and where[it.sig] != split:
                leaks += 1
            where[it.sig] = split
    if leaks:
        print(f"  !! LEAKAGE: {leaks} signatures span multiple splits")
    else:
        print("  Leakage check PASSED - no image signature spans splits")
    return leaks == 0


def collect_items(keep_proxies: bool) -> list[Item]:
    # Order matters: multi-class / real-box sources first so they win dedup ties.
    sources = [
        ("Road-Damage (cracks+potholes, multi-class)", process_road_damage),
        ("Wall-Crack (cracks)", process_wall_crack),
        ("Corrosion (real boxes)", process_corrosion),
        ("Paint-Degradation (real boxes)", process_paint_degradation),
        ("Spalling + Spalling2 (real boxes)", process_spalling),
        ("Potholes v1/V2/V3/V4 (real boxes, dup-heavy)", process_potholes_roboflow),
    ]
    if keep_proxies:
        sources.append(("Concrete-Structural (whole-image PROXY)", process_concrete_structural))

    items = []
    for label, fn in sources:
        got = fn()
        n_box = sum(len(it.lines) for it in got)
        print(f"  {label:<48}: {len(got):>5} imgs, {n_box:>6} boxes")
        items += got
    return items


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--target', type=int, default=TARGET_PER_CLASS,
                    help='Max bounding boxes kept per class (default 2000)')
    ap.add_argument('--keep-proxies', action='store_true',
                    help='Also include concrete-structural whole-image proxy boxes')
    args = ap.parse_args()

    print("Clearing existing data/ ...")
    clear_data_split_dirs()

    print("\n=== 1. COLLECT (real-bbox sources) ===")
    items = collect_items(args.keep_proxies)
    print(f"  collected: {len(items)} images, "
          f"{sum(len(it.lines) for it in items)} boxes (before dedup)")

    print("\n=== 2. DEDUP (perceptual a-hash, exact match) ===")
    items, removed = dedup(items)
    print(f"  removed {removed} duplicate/unreadable images -> {len(items)} unique")

    print(f"\n=== 3. BALANCE (cap {args.target} boxes/class) ===")
    selected, counts = balance(items, args.target)
    print(f"  selected {len(selected)} images")
    for cid, name in enumerate(CLASSES):
        print(f"    {name:<18}: {counts[cid]:>6} boxes")

    print("\n=== 4. SPLIT (stratified 70/15/15, leak-free) ===")
    splits = split_items(selected)

    print("\n=== 5. SAVE ===")
    save_splits(splits)
    write_data_yaml()
    write_manifest(splits)

    report(splits)
    print("\n=== Leakage verification ===")
    verify_no_leakage(splits)
    print("\nDone.")


if __name__ == '__main__':
    main()
