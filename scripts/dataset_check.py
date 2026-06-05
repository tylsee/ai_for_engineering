"""Dataset integrity + quality inspection for the structural-defect detector.

Covers report Step 1:
  (1a) every image has >=1 bbox; flag images whose SMALLEST box < 0.5% of area
  (1b) confirm no perceptual-hash duplicates leak across train/val/test
Plus the bbox aspect-ratio distribution needed for SSDLite anchor tuning (Step 3c).

Pure stdlib + PIL/numpy/pandas (no imagehash dependency). Run:
    python scripts/dataset_check.py
"""
from __future__ import annotations
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SPLITS = ["train", "val", "test"]
CLASSES = ["cracks", "spalling", "corrosion", "potholes", "paint_degradation"]
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
AREA_THRESH = 0.005  # 0.5% of image area

def parse_label(path: Path):
    out = []
    txt = path.read_text().strip()
    if not txt:
        return out
    for line in txt.splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cid = int(float(parts[0]))
        cx, cy, w, h = map(float, parts[1:])
        out.append((cid, cx, cy, w, h))
    return out

def find_image(split: str, stem: str) -> Path | None:
    base = DATA / "images" / split
    for ext in IMG_EXTS:
        p = base / f"{stem}{ext}"
        if p.exists():
            return p
    return None

def ahash(path: Path, size: int = 8) -> int | None:
    """8x8 average hash -> 64-bit int (matches the dedup step in 01_reorganize_data.py)."""
    try:
        im = Image.open(path).convert("L").resize((size, size), Image.BILINEAR)
    except Exception:
        return None
    a = np.asarray(im, dtype=np.float32)
    bits = (a > a.mean()).flatten()
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return val

def main():
    rows = []            # per-box records
    img_rows = []        # per-image records
    empty_images = []    # images with zero boxes
    missing_images = []  # label exists but no matching image

    for split in SPLITS:
        ldir = DATA / "labels" / split
        for lp in sorted(ldir.glob("*.txt")):
            recs = parse_label(lp)
            ip = find_image(split, lp.stem)
            if ip is None:
                missing_images.append(f"{split}/{lp.stem}")
            if not recs:
                empty_images.append(f"{split}/{lp.stem}")
                img_rows.append({"split": split, "stem": lp.stem, "n_boxes": 0,
                                 "min_area": np.nan})
                continue
            areas = [w * h for (_, _, _, w, h) in recs]
            img_rows.append({"split": split, "stem": lp.stem, "n_boxes": len(recs),
                             "min_area": min(areas)})
            for (cid, cx, cy, w, h) in recs:
                rows.append({"split": split, "stem": lp.stem, "cid": cid,
                             "cls": CLASSES[cid] if cid < len(CLASSES) else f"?{cid}",
                             "w": w, "h": h, "area": w * h,
                             "ar": (w / h) if h > 0 else np.nan})

    box = pd.DataFrame(rows)
    img = pd.DataFrame(img_rows)

    print("=" * 70)
    print("STEP 1a  DATASET INTEGRITY")
    print("=" * 70)
    print(f"Total label files : {len(img)}")
    print(f"Total boxes       : {len(box)}")
    print(f"Empty (0-box) imgs: {len(empty_images)}")
    print(f"Labels w/o image  : {len(missing_images)}")

    print("\n-- boxes per class x split --")
    piv = box.pivot_table(index="cls", columns="split", values="area",
                          aggfunc="count", fill_value=0)
    piv = piv.reindex(CLASSES)[SPLITS]
    piv["TOTAL"] = piv.sum(axis=1)
    print(piv.to_string())

    print("\n-- box AREA RATIO percentiles (fraction of image) --")
    print(box["area"].describe(percentiles=[.01, .05, .10, .25, .5, .75, .95]).to_string())

    # Flag images whose SMALLEST box < threshold
    flagged = img[(img["n_boxes"] > 0) & (img["min_area"] < AREA_THRESH)]
    print(f"\n-- images with smallest bbox < {AREA_THRESH*100:.1f}% area: {len(flagged)} "
          f"({len(flagged)/len(img)*100:.2f}% of dataset) --")

    # Tiny boxes by class (count of individual boxes under threshold)
    tiny = box[box["area"] < AREA_THRESH]
    print(f"\n-- individual boxes < {AREA_THRESH*100:.1f}% area: {len(tiny)} "
          f"({len(tiny)/len(box)*100:.2f}% of all boxes) --")
    if len(tiny):
        tc = tiny.groupby("cls").size().reindex(CLASSES, fill_value=0)
        tot = box.groupby("cls").size().reindex(CLASSES, fill_value=0)
        tbl = pd.DataFrame({"tiny_boxes": tc, "total_boxes": tot})
        tbl["pct"] = (tbl["tiny_boxes"] / tbl["total_boxes"] * 100).round(2)
        print(tbl.to_string())

    # Save flagged list for review
    out = ROOT / "runs" / "flagged_tiny_boxes.csv"
    out.parent.mkdir(exist_ok=True)
    flagged.sort_values("min_area").to_csv(out, index=False)
    print(f"\nFlagged list -> {out.relative_to(ROOT)}")

    print("\n" + "=" * 70)
    print("STEP 3c  ASPECT-RATIO DISTRIBUTION (for SSDLite anchors)")
    print("=" * 70)
    print("aspect ratio = box_w / box_h  (normalised dims)")
    arr = box.groupby("cls")["ar"].describe(percentiles=[.1, .25, .5, .75, .9])
    print(arr.reindex(CLASSES).to_string())
    print("\noverall AR percentiles:")
    print(box["ar"].describe(percentiles=[.05, .1, .25, .5, .75, .9, .95]).to_string())

    print("\n" + "=" * 70)
    print("STEP 1b  CROSS-SPLIT DUPLICATE CHECK (8x8 a-hash)")
    print("=" * 70)
    sig_to_locs: dict[int, list[str]] = defaultdict(list)
    n_hashed = 0
    for split in SPLITS:
        for ip in sorted((DATA / "images" / split).iterdir()):
            if ip.suffix.lower() not in IMG_EXTS:
                continue
            h = ahash(ip)
            if h is None:
                continue
            sig_to_locs[h].append(f"{split}/{ip.name}")
            n_hashed += 1
    cross = {s: locs for s, locs in sig_to_locs.items()
             if len({l.split('/')[0] for l in locs}) > 1}
    within = {s: locs for s, locs in sig_to_locs.items()
              if len(locs) > 1 and len({l.split('/')[0] for l in locs}) == 1}
    print(f"Images hashed         : {n_hashed}")
    print(f"Unique signatures     : {len(sig_to_locs)}")
    print(f"Cross-split collisions: {len(cross)}  <-- must be 0 (leakage)")
    print(f"Within-split dup sigs : {len(within)}")
    if cross:
        print("\nFIRST 20 CROSS-SPLIT COLLISIONS:")
        for i, (s, locs) in enumerate(list(cross.items())[:20]):
            print(f"  {locs}")
    else:
        print("\nOK: no a-hash signature spans multiple splits -> no leakage detected.")

    return 0

if __name__ == "__main__":
    sys.exit(main())
