"""Create a forward-slash zip of a dataset dir (Linux/Colab/Kaggle-safe).
PowerShell Compress-Archive uses Windows backslashes which Python zipfile on Linux treats as
literal filenames instead of path separators. Entries here are always rooted at 'data/' so the
zip unzips into data/ on Kaggle/Colab regardless of the source dir.

Usage (from project root):
    python scripts/zip_data.py                                   # data/  -> defect_dataset.zip
    python scripts/zip_data.py --src data_v3 --out defect_dataset_v3.zip
"""
import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser(description='Zip a dataset dir with forward-slash paths rooted at data/.')
ap.add_argument('--src', default='data', help='source dataset dir under the project root (default: data)')
ap.add_argument('--out', default='defect_dataset.zip', help='output zip filename (default: defect_dataset.zip)')
args = ap.parse_args()

SRC = ROOT / args.src
OUT = ROOT / args.out
assert SRC.exists(), f"source dir not found: {SRC}"

written = 0
with zipfile.ZipFile(OUT, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    for fpath in sorted(SRC.rglob('*')):
        if fpath.is_file():
            arcname = 'data/' + fpath.relative_to(SRC).as_posix()  # root at data/ for Colab/Kaggle
            zf.write(fpath, arcname)
            written += 1

size_mb = OUT.stat().st_size / 1e6
print(f"Zipped {SRC.name}/ -> {OUT.name} ({written} files, {size_mb:.1f} MB; entries rooted at data/)")
