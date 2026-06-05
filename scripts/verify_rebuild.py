"""Concise verification of a rebuilt dataset dir (prints a short summary).
    python scripts/verify_rebuild.py [--data data_v3]"""
import argparse
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser(description='Verify a rebuilt dataset dir.')
ap.add_argument('--data', default='data', help='dataset dir under the project root (default: data)')
DATA = ROOT / ap.parse_args().data
CLASSES = ['cracks', 'spalling', 'corrosion', 'potholes', 'paint_degradation']
SPLITS = ['train', 'val', 'test']
MIN_PER_CLASS = 200
print(f"Verifying: {DATA}\n")

per_split = {s: Counter() for s in SPLITS}
areas = defaultdict(list)
ars = defaultdict(list)
min_w = min_h = 1.0
multi_imgs = {s: 0 for s in SPLITS}
n_imgs = {s: 0 for s in SPLITS}
degenerate = 0
invalid = 0

for s in SPLITS:
    for lp in (DATA / "labels" / s).glob("*.txt"):
        n_imgs[s] += 1
        classes_here = set()
        for ln in lp.read_text().splitlines():
            if not ln.strip():
                continue
            p = ln.split()
            if len(p) != 5:
                invalid += 1
                continue
            try:
                c = int(p[0]); cx, cy, w, h = map(float, p[1:])
            except ValueError:
                invalid += 1
                continue
            if not (0 <= c < len(CLASSES)) or not (0 <= cx <= 1 and 0 <= cy <= 1) \
                    or not (0 < w <= 1 and 0 < h <= 1):
                invalid += 1
                continue
            per_split[s][c] += 1
            classes_here.add(c)
            areas[c].append(w * h)
            if h > 0:
                ars[c].append(w / h)
            min_w = min(min_w, w); min_h = min(min_h, h)
            if w <= 0 or h <= 0:
                degenerate += 1
        if len(classes_here) >= 2:
            multi_imgs[s] += 1

print("=== Per-class boxes per split ===")
print(f"{'class':<18}{'train':>7}{'val':>6}{'test':>6}{'TOTAL':>7}{'%tiny<0.5%':>11}")
grand = {}
for cid, name in enumerate(CLASSES):
    tr, va, te = (per_split[s][cid] for s in SPLITS)
    tot = tr + va + te; grand[name] = tot
    a = np.array(areas[cid]); tiny = (a < 0.005).mean() * 100 if len(a) else 0
    print(f"{name:<18}{tr:>7}{va:>6}{te:>6}{tot:>7}{tiny:>10.1f}%")
vals = [v for v in grand.values() if v]
print(f"\nTOTAL boxes: {sum(grand.values())} | balance ratio: {max(vals)/min(vals):.2f}x")
print(f"Images: train={n_imgs['train']} val={n_imgs['val']} test={n_imgs['test']} "
      f"(total {sum(n_imgs.values())})")
print(f"Multi-class images (>=2 of our classes): "
      f"train={multi_imgs['train']} val={multi_imgs['val']} test={multi_imgs['test']} "
      f"(total {sum(multi_imgs.values())})")

print("\n=== Box geometry sanity ===")
print(f"degenerate boxes (w<=0 or h<=0): {degenerate}")
print(f"invalid label lines (bad fields/class/coords): {invalid}   (0 = clean)")
print(f"min width seen : {min_w:.4f}   min height seen: {min_h:.4f}")
print(f"{'class':<18}{'AR p50':>8}{'AR p95':>8}{'AR max':>8}")
for cid, name in enumerate(CLASSES):
    a = np.array(ars[cid])
    if len(a):
        print(f"{name:<18}{np.percentile(a,50):>8.2f}{np.percentile(a,95):>8.2f}{a.max():>8.2f}")

print(f"\n=== Class minimum check (>= {MIN_PER_CLASS} boxes) ===")
classes_ok = True
for cid, name in enumerate(CLASSES):
    tot = sum(per_split[s][cid] for s in SPLITS)
    if tot < MIN_PER_CLASS:
        classes_ok = False
    print(f"  {'OK ' if tot >= MIN_PER_CLASS else 'XX '}{name:<18} {tot} boxes")

passed = classes_ok and invalid == 0 and degenerate == 0
print(f"\nRESULT: {'PASS' if passed else 'CHECK FAILED'} "
      f"(invalid={invalid}, degenerate={degenerate}, all-classes>={MIN_PER_CLASS}: {classes_ok})")
