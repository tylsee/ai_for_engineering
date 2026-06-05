"""Dataset repair and verification for the structural-defect YOLO dataset.

Default is a dry run (reports what would change). Use --apply to actually modify data/.
Labels are git-tracked (restorable with `git checkout HEAD -- data/labels/`); images are NOT,
so back up data/ (or the zip) before running with --apply.

    python scripts/prepare_training_data.py            # dry run
    python scripts/prepare_training_data.py --apply     # apply repairs

The repair functions between the BEGIN/END markers are also inlined into the notebooks'
"Part 1.6 Dataset repair and verification" cell, so the same logic runs on Colab/Kaggle.
"""
import argparse
import subprocess
import sys
from pathlib import Path

# === BEGIN repair ===
from collections import Counter
from PIL import Image, ImageFile

CLASSES = ['cracks', 'spalling', 'corrosion', 'potholes', 'paint_degradation']
IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
SPLITS = ('train', 'val', 'test')


def _inspect_image(path):
    """Return one of: 'ok', 'convert' (wrong format named .jpg), 'reencode' (truncated but
    recoverable), 'remove' (cannot open). Second value is the detected format string."""
    try:
        with Image.open(path) as im:
            fmt = (im.format or '').upper()
            im.load()  # strict decode
        if path.suffix.lower() in ('.jpg', '.jpeg') and fmt not in ('JPEG', 'MPO'):
            return 'convert', fmt          # e.g. a GIF/PNG saved with a .jpg name
        return 'ok', fmt
    except Exception:
        pass
    # lenient pass: recover truncated/corrupt images PIL can still decode
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    try:
        with Image.open(path) as im:
            fmt = (im.format or '').upper()
            im.convert('RGB').load()
        return 'reencode', fmt
    except Exception:
        return 'remove', '?'
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = False


def _save_rgb(path):
    """Re-encode an image to a clean RGB file at the same path (JPEG for .jpg/.jpeg, else PNG)."""
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    try:
        with Image.open(path) as im:
            rgb = im.convert('RGB')
        if path.suffix.lower() in ('.jpg', '.jpeg'):
            rgb.save(path, 'JPEG', quality=95)
        elif path.suffix.lower() == '.png':
            rgb.save(path, 'PNG')
        else:
            rgb.save(path)
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = False


def _clean_label_lines(text):
    """Return (kept_lines, n_duplicates, n_invalid). A line is valid when it has 5 fields,
    class_id in 0..4, x/y in [0,1], and w,h in (0,1]. Exact duplicate boxes are dropped."""
    seen, kept = set(), []
    dupes = invalid = 0
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            continue
        parts = s.split()
        if len(parts) != 5:
            invalid += 1
            continue
        try:
            cid = int(parts[0])
            cx, cy, w, h = (float(v) for v in parts[1:])
        except ValueError:
            invalid += 1
            continue
        if not (0 <= cid < len(CLASSES)):
            invalid += 1
            continue
        if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
            invalid += 1
            continue
        if not (0.0 < w <= 1.0 and 0.0 < h <= 1.0):
            invalid += 1
            continue
        key = (cid, round(cx, 6), round(cy, 6), round(w, 6), round(h, 6))
        if key in seen:
            dupes += 1
            continue
        seen.add(key)
        kept.append(s)
    return kept, dupes, invalid


def prepare_dataset(data_dir, apply=False):
    """Repair images and labels under data_dir. Counts everything; only writes when apply=True."""
    data_dir = Path(data_dir)
    n = Counter()
    for split in SPLITS:
        img_dir = data_dir / 'images' / split
        lbl_dir = data_dir / 'labels' / split
        if not img_dir.exists():
            continue
        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() not in IMAGE_EXTS:
                continue
            n['files_checked'] += 1
            lbl = lbl_dir / (img.stem + '.txt')
            status, _ = _inspect_image(img)
            if status == 'remove':
                n['images_removed'] += 1
                if apply:
                    img.unlink(missing_ok=True)
                    lbl.unlink(missing_ok=True)
                continue
            if status in ('convert', 'reencode'):
                n['images_fixed'] += 1
                if apply:
                    _save_rgb(img)
            if lbl.exists():
                kept, dupes, invalid = _clean_label_lines(lbl.read_text())
                n['duplicate_labels_removed'] += dupes
                n['invalid_labels_removed'] += invalid
                if not kept:                       # nothing valid left -> drop the image too
                    n['images_removed'] += 1
                    if apply:
                        img.unlink(missing_ok=True)
                        lbl.unlink(missing_ok=True)
                elif (dupes or invalid) and apply:
                    lbl.write_text('\n'.join(kept) + '\n')
    return n


def verify_dataset(data_dir):
    """Print a short per-class / per-split summary (used when verify_rebuild.py is unavailable)."""
    data_dir = Path(data_dir)
    per = {c: 0 for c in range(len(CLASSES))}
    imgs = {s: 0 for s in SPLITS}
    total = 0
    for s in SPLITS:
        ld = data_dir / 'labels' / s
        if not ld.exists():
            continue
        for lp in ld.glob('*.txt'):
            imgs[s] += 1
            for ln in lp.read_text().splitlines():
                p = ln.split()
                if len(p) != 5:
                    continue
                try:
                    c = int(p[0])
                except ValueError:
                    continue
                if 0 <= c < len(CLASSES):
                    per[c] += 1
                    total += 1
    print('Verification summary')
    print('  images: ' + ', '.join('%s=%d' % (s, imgs[s]) for s in SPLITS)
          + ' (total %d)' % sum(imgs.values()))
    for c, name in enumerate(CLASSES):
        print('  %-18s %d' % (name, per[c]))
    nz = [v for v in per.values() if v]
    bal = (max(nz) / min(nz)) if nz else 0.0
    print('  total boxes: %d | balance ratio: %.2fx' % (total, bal))
# === END repair ===


def _find_data():
    for base in [Path.cwd(), *Path.cwd().parents]:
        if (base / 'data' / 'images' / 'train').exists():
            return base / 'data'
    raise FileNotFoundError('could not find data/images/train - run from inside the repo')


def main():
    ap = argparse.ArgumentParser(description='Repair and verify the YOLO defect dataset.')
    ap.add_argument('--data', default=None, help='path to data/ (default: auto-detect)')
    ap.add_argument('--apply', action='store_true', help='actually modify files (default: dry run)')
    args = ap.parse_args()

    data_dir = Path(args.data) if args.data else _find_data()
    if args.apply:
        print('APPLYING changes to %s (labels are git-tracked; images are not)\n' % data_dir)
    else:
        print('DRY_RUN: no files were changed. Use --apply to apply repairs.\n')

    summary = prepare_dataset(data_dir, apply=args.apply)
    print('Repair summary')
    for k in ('files_checked', 'images_fixed', 'images_removed',
              'duplicate_labels_removed', 'invalid_labels_removed'):
        print('  %-26s %d' % (k, summary[k]))
    print()

    sys.stdout.flush()
    verifier = Path(__file__).resolve().parent / 'verify_rebuild.py'
    if verifier.exists():
        cmd = [sys.executable, str(verifier)]
        if args.data:
            cmd += ['--data', str(args.data)]
        subprocess.run(cmd, check=False)
    else:
        verify_dataset(data_dir)


if __name__ == '__main__':
    main()
