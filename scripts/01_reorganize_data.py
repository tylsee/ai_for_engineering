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

# Per-class bbox targets. Quality-over-quantity: strong classes are capped below
# the old flat 2000 to avoid padding with lower-grade boxes; paint is limited by
# how much real-box data exists, so it lands lower and a MILD imbalance is accepted
# (rubric needs only >=200/class; mild imbalance is fine if not extreme).
DEFAULT_TARGET = 1800
TARGET_PER_CLASS = {
    'cracks': 1800, 'spalling': 1800, 'corrosion': 1800,
    'potholes': 1800, 'paint_degradation': 1800,   # paint capped by source supply -> lands lower
}

# "Bigger box = better" rebuild (4 Jun 2026, user-directed, POTHOLES ONLY):
# potholes were 46.6% tiny (vs 6-14% for other classes) because road-damage adds many
# small distant road potholes. We bias the pothole class toward large, clearly-visible
# boxes by (a) sourcing v5 only from its all-big images, (b) filling the pothole target
# biggest-box-first, and (c) DROPPING crack+pothole multi-class images whose potholes are
# ALL tiny. Cracks/corrosion are inherently thin -> left untouched (their small boxes are
# real defects). "Select-only": we never delete a box from a kept image's label file.
POTHOLE_CID = 3
TINY_POTHOLE_AREA = 0.005   # <0.5% of image area — distant/junk potholes to bias away from
BIG_POTHOLE_AREA = 0.02     # >=2% of image area — "big, clear" potholes (v5-big image filter)

# Cap noisy / ambiguous LEGACY sources (in boxes) so cleaner new sources dominate.
# The old Roboflow paint-degradation mixed in faded road-markings -> reduce its share.
SOURCE_BOX_CAP = {'paint_degradation': 700}

# Phase-B fill priority (lower = consumed first). New clean real-box sources win;
# legacy/noisier sources only top up to the target. corrosion_v2 and the old
# Corrosion share a priority so they MIX (keeps image diversity).
SOURCE_PRIORITY = {
    'road_damage': 0, 'wall_crack': 0, 'dataset3': 1,
    'corrosion_v2': 1, 'corrosion': 1,
    'spalling': 1,
    'pothole_big': 1, 'pothole_v5_big': 1, 'pothole_v5': 2,  # prefer BIG v1/V4 + v5-big; plain v5 unused
    'paint_v2': 1, 'paint_v3': 1,        # real human boxes first
    'paint_peeling': 4,                  # optional model-assisted paint (added later)
    'paint_degradation': 6,              # OLD paint -> reduced top-up only
}
DEFAULT_PRIORITY = 3

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


MIN_BOX_DIM = 0.003   # 0.3% of an image side; thinner boxes are Roboflow polygon->bbox
                      # export artifacts (w=0/h=0 or AR in the thousands). Clamp/drop them.
MAX_AR = 50.0         # drop boxes more extreme than 50:1 (or 1:50) — full-width 2px
                      # slivers from polygon exports, not real defects. Keeps genuinely
                      # elongated cracks/rust (AR up to 50) which are legitimate.
CORROSION_MAX_AR = 12.0  # v3: drop corrosion (class 2) boxes more elongated than 12:1 — extreme
                         # axis-aligned rust-streak outliers. Removes AR>12 outliers (does NOT
                         # guarantee a lower p95). Cracks stay thin (their AR is legitimate).


def _clean_box(cx: float, cy: float, w: float, h: float):
    """Drop degenerate boxes (w/h<=0) and extreme-AR export slivers (>MAX_AR);
    clamp thin slivers to MIN_BOX_DIM and keep the box inside [0,1].
    Returns cleaned (cx,cy,w,h) or None to drop."""
    if w <= 0 or h <= 0:
        return None
    w = min(max(w, MIN_BOX_DIM), 1.0)
    h = min(max(h, MIN_BOX_DIM), 1.0)
    ar = w / h
    if ar > MAX_AR or ar < 1.0 / MAX_AR:
        return None
    cx = min(max(cx, w / 2), 1 - w / 2)
    cy = min(max(cy, h / 2), 1 - h / 2)
    return cx, cy, w, h


def _filter_class_ar(items: list[Item], cid: int, max_ar: float):
    """Drop boxes of class `cid` whose aspect ratio is more extreme than max_ar:1 (or 1:max_ar).
    An image with no boxes left is dropped. Returns (items, n_boxes_dropped, n_images_dropped)."""
    out, boxes_dropped, imgs_dropped = [], 0, 0
    for it in items:
        kept = []
        for ln in it.lines:
            p = ln.split()
            if int(p[0]) == cid:
                w, h = float(p[3]), float(p[4])
                ar = (w / h) if h > 0 else max_ar + 1
                if ar > max_ar or ar < 1.0 / max_ar:
                    boxes_dropped += 1
                    continue
            kept.append(ln)
        if kept:
            it.lines = kept
            out.append(it)
        else:
            imgs_dropped += 1
    return out, boxes_dropped, imgs_dropped


def _read_label(lbl: Path, keep_remap: dict[int, int]) -> list[str]:
    """Reads a YOLO label file, keeping only classes in keep_remap, remapping their
    ids to the target space, and cleaning degenerate/sliver boxes. [] if nothing kept."""
    if not lbl.exists():
        return []
    out = []
    for raw in lbl.read_text().splitlines():
        parts = raw.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        if cls not in keep_remap:
            continue
        try:
            cx, cy, w, h = map(float, parts[1:5])
        except ValueError:
            continue
        cleaned = _clean_box(cx, cy, w, h)
        if cleaned is None:
            continue
        cx, cy, w, h = cleaned
        out.append(f"{keep_remap[cls]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
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


def _flat_items(img_dir: Path, lbl_dir: Path, keep_remap: dict[int, int], source: str) -> list[Item]:
    """Reads a flat (images/ + labels/) YOLO dataset with no split subfolders."""
    items = []
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


def process_dataset3() -> list[Item]:
    """Dataset3 (Roboflow 'building-damage-insurance', nc=10). Keep only our classes:
    crack(2) + stairstep_crack(8) -> cracks(0); peeling_paint(6) -> paint(4). Other
    classes (mold/damp/dampness/stain/water_seepage) are dropped. Added 4 Jun to
    replace the deleted road-damage source: supplies cracks (3,450 boxes) and restores
    MULTI-CLASS images (crack+peeling) required by the rubric."""
    return _roboflow_items(DATASET_DIR / 'Dataset3', {2: 0, 8: 0, 6: 4}, 'dataset3')


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


def process_potholes_big() -> list[Item]:
    """potholesv1 + PotholesV4 (Roboflow YOLO): pothole(0) -> potholes(3).
    Manually verified 4 Jun: these carry LARGER, clearly-visible potholes
    (median box ~2.1% of image, only ~18% tiny) vs potholesv5's distant-pothole
    skew (~38% tiny). v1 and V4 are near-identical uploads — dedup collapses them.
    Preferred over v5 to lift the pothole size distribution. (PotholesV2 chess +
    potholesV3 were deleted by the user.)"""
    items = []
    for name in ('potholesv1', 'PotholesV4'):
        items += _roboflow_items(DATASET_DIR / name, {0: 3}, 'pothole_big')
    return items


def process_potholes_v5() -> list[Item]:
    """potholesv5 (Roboflow YOLO, train+valid): pothole(0) -> potholes(3). Real
    boxes; replaces the dup-heavy tiny v1/V2/V3/V4 sources. NOTE: superseded by
    process_potholes_v5_big() in collect_items() for the big-pothole-bias rebuild."""
    return _roboflow_items(DATASET_DIR / 'potholesv5', {0: 3}, 'pothole_v5')


def process_potholes_v5_big() -> list[Item]:
    """potholesv5 restricted to images where EVERY pothole box is BIG (>=2% of image
    area). Real boxes, labels kept INTACT (no box dropping — 'select-only'). The median
    v5 box is small, but ~1,163 v5 images are all-big and clean; using only those biases
    the pothole class toward large, clearly-visible potholes (user-directed 4 Jun 2026)
    without ever pulling in a tiny/distant v5 box. Replaces the plain v5 top-up."""
    items = []
    for it in _roboflow_items(DATASET_DIR / 'potholesv5', {0: 3}, 'pothole_v5_big'):
        areas = [float(ln.split()[3]) * float(ln.split()[4]) for ln in it.lines]
        if areas and min(areas) >= BIG_POTHOLE_AREA:   # all boxes big -> keep image whole
            items.append(it)
    return items


def process_corrosion_v2() -> list[Item]:
    """corrosionv2 'corrosion detect' (flat YOLO): corrosion(1) -> corrosion(2).
    Real boxes, several tight boxes per image. Mixed with the older Corrosion set
    (same priority) to combine clean labels with broader image diversity."""
    base = DATASET_DIR / 'corrosionv2' / 'corrosion detect'
    return _flat_items(base / 'images', base / 'labels', {0: 2, 1: 2}, 'corrosion_v2')


def process_paint_v3() -> list[Item]:
    """paintdegradationv3 (Roboflow YOLO): 'peel paint'(0) -> paint_degradation(4).
    Real boxes, on-topic peeling-paint imagery."""
    return _roboflow_items(DATASET_DIR / 'paintdegradationv3', {0: 4}, 'paint_v3')


def process_paint_v2() -> list[Item]:
    """paintdegradationv2 (Roboflow COCO export, flat train/): real boxes.
    category 3 peeling -> paint(4); 1 crack -> cracks(0); 4 spalling -> spalling(1);
    2 mold + 0 supercategory dropped. Multi-class images kept (rubric)."""
    import json
    base = DATASET_DIR / 'paintdegradationv2' / 'train'
    jf = base / '_annotations.coco.json'
    if not jf.exists():
        return []
    d = json.loads(jf.read_text())
    cat_remap = {3: 4, 1: 0, 4: 1}                 # peeling->paint, crack->cracks, spalling->spalling
    imgs = {im['id']: im for im in d['images']}
    by_img: dict[int, list[str]] = defaultdict(list)
    for a in d['annotations']:
        cid = a['category_id']
        if cid not in cat_remap:
            continue
        im = imgs[a['image_id']]
        W, H = im['width'], im['height']
        x, y, w, h = a['bbox']                      # COCO: x,y,w,h absolute pixels
        cleaned = _clean_box((x + w / 2) / W, (y + h / 2) / H, w / W, h / H)
        if cleaned is None:
            continue
        ccx, ccy, cw, ch = cleaned
        by_img[a['image_id']].append(
            f"{cat_remap[cid]} {ccx:.6f} {ccy:.6f} {cw:.6f} {ch:.6f}\n")
    items = []
    for img_id, lines in by_img.items():
        p = base / imgs[img_id]['file_name']
        if p.exists() and lines:
            items.append(Item(p, lines, 'paint_v2'))
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

def balance(items: list[Item], targets: dict[str, int]) -> tuple[list[Item], Counter]:
    """Cap each class at its own bbox target (`targets[class_name]`). Multi-class
    images are kept first (scarce + required by rubric); single-class images then
    top up each class, consuming higher-priority (cleaner/newer) sources before
    legacy ones. A class with fewer boxes than its target uses everything."""
    for it in items:
        it.classes = frozenset(it.bbox_class_counts)

    rng = random.Random(SEED)
    counts = Counter()
    selected: list[Item] = []

    def _pothole_areas(it):
        return [float(ln.split()[3]) * float(ln.split()[4])
                for ln in it.lines if int(ln.split()[0]) == POTHOLE_CID]

    multi = [it for it in items if len(it.classes) > 1]
    single_by_cls: dict[int, list[Item]] = defaultdict(list)
    for it in items:
        if len(it.classes) == 1:
            single_by_cls[next(iter(it.classes))].append(it)

    # Phase A — multi-class images, EXCEPT crack+pothole images whose potholes are ALL
    # tiny (<0.5%). Big-pothole bias: these all-tiny multi-class images are the bulk of
    # the tiny-pothole share, so we drop them (their cracks backfill in Phase B). Multi-
    # class images with a non-tiny pothole, and non-pothole multi-class (crack+peeling),
    # are kept. Select-only: we drop whole images, never edit a kept image's labels.
    for it in multi:
        pa = _pothole_areas(it)
        if pa and max(pa) < TINY_POTHOLE_AREA:
            continue   # all-tiny-pothole multi-class image -> drop
        selected.append(it)
        counts.update(it.bbox_class_counts)

    # Phase B — top up each class with single-class images (rarest class first so
    # scarce classes are filled before abundant ones compete for the budget). POTHOLES
    # fill biggest-box-first (bias toward large, clear potholes), source priority as the
    # tie-break; all other classes keep the source-priority fill (shuffled within tier).
    for cls in sorted(range(len(CLASSES)), key=lambda c: counts[c]):
        tgt = targets.get(CLASSES[cls], DEFAULT_TARGET)
        pool = single_by_cls.get(cls, [])
        rng.shuffle(pool)
        if cls == POTHOLE_CID:
            # rank by the image's SMALLEST pothole box (desc): prefer images with no
            # tiny boxes at all, so tiny single-class potholes are only used as a last
            # resort once the big supply (v5-big + v1/V4) is exhausted.
            pool.sort(key=lambda it: (-min(_pothole_areas(it)),
                                      SOURCE_PRIORITY.get(it.source, DEFAULT_PRIORITY)))
        else:
            pool.sort(key=lambda it: SOURCE_PRIORITY.get(it.source, DEFAULT_PRIORITY))
        for it in pool:
            if counts[cls] >= tgt:
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
    print(f"  Cleared and recreated {DATA_DIR.name}/images/* and {DATA_DIR.name}/labels/*")


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


def _cap_source(items: list[Item]) -> list[Item]:
    """Seeded truncation of any capped legacy source to its box budget, preserving
    the original collect order (so dedup tie-breaks are unaffected)."""
    if not ({it.source for it in items} & set(SOURCE_BOX_CAP)):
        return items
    rng = random.Random(SEED)
    pool = list(items); rng.shuffle(pool)
    budgets = defaultdict(int)
    keep = set()
    for it in pool:
        cap = SOURCE_BOX_CAP.get(it.source)
        if cap is not None:
            if budgets[it.source] >= cap:
                continue
            budgets[it.source] += len(it.lines)
        keep.add(id(it))
    return [it for it in items if id(it) in keep]


def collect_items(keep_proxies: bool) -> list[Item]:
    # Order matters: multi-class / real-box sources first so they win dedup ties.
    sources = [
        ("Road-Damage (cracks+potholes, multi-class)", process_road_damage),
        ("Wall-Crack (cracks)", process_wall_crack),
        ("Dataset3 (crack+stairstep->cracks, peeling->paint, multi)", process_dataset3),
        ("Corrosion-v2 (clean real boxes — v3: only corrosion source)", process_corrosion_v2),
        ("Spalling + Spalling2 + Spalling3 (real boxes)", process_spalling),
        ("Potholes-v1+V4 (big, clear potholes - preferred)", process_potholes_big),
        ("Potholes-v5-BIG (>=2% boxes only; big-pothole bias)", process_potholes_v5_big),
        ("Paint-v2 (COCO: peeling+crack+spalling real boxes)", process_paint_v2),
        ("Paint-v3 (YOLO real peeling-paint boxes)", process_paint_v3),
    ]
    # v3: old Corrosion/ and old paint-degradation/ are dropped (noisy/ambiguous). Their
    # process_* functions are kept above for reference but no longer collected.
    if keep_proxies:
        sources.append(("Concrete-Structural (whole-image PROXY)", process_concrete_structural))

    items = []
    for label, fn in sources:
        got = _cap_source(fn())
        n_box = sum(len(it.lines) for it in got)
        print(f"  {label:<52}: {len(got):>5} imgs, {n_box:>6} boxes")
        items += got
    # v3: tighten corrosion geometry — drop extreme elongated rust-streak outliers (AR>12)
    items, bd, idrop = _filter_class_ar(items, cid=2, max_ar=CORROSION_MAX_AR)
    print(f"  corrosion AR>{CORROSION_MAX_AR:.0f} outlier filter{'':<26}: dropped {bd} boxes, {idrop} images")
    return items


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--target', type=int, default=None,
                    help='Override: use this single bbox cap for ALL classes '
                         '(default uses per-class TARGET_PER_CLASS)')
    ap.add_argument('--keep-proxies', action='store_true',
                    help='Also include concrete-structural whole-image proxy boxes')
    ap.add_argument('--out', default='data',
                    help='Output dataset dir under the project root (default: data). '
                         'Use --out data_v3 to build v3 without overwriting v2 in data/.')
    args = ap.parse_args()

    global DATA_DIR
    DATA_DIR = PROJECT_ROOT / args.out
    print(f"Building dataset into: {DATA_DIR}")

    targets = ({c: args.target for c in CLASSES} if args.target is not None
               else dict(TARGET_PER_CLASS))

    print(f"Clearing existing {DATA_DIR.name}/ ...")
    clear_data_split_dirs()

    print("\n=== 1. COLLECT (real-bbox sources) ===")
    items = collect_items(args.keep_proxies)
    print(f"  collected: {len(items)} images, "
          f"{sum(len(it.lines) for it in items)} boxes (before dedup)")

    print("\n=== 2. DEDUP (perceptual a-hash, exact match) ===")
    items, removed = dedup(items)
    print(f"  removed {removed} duplicate/unreadable images -> {len(items)} unique")

    print(f"\n=== 3. BALANCE (per-class targets: {targets}) ===")
    selected, counts = balance(items, targets)
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
