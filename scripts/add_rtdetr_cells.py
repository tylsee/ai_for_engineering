"""Insert the RT-DETR markdown + self-contained code cell into both all-in-one
notebooks, just before the 'Part 4 - Summary' markdown (so RT-DETR results exist
before the summary). Idempotent: re-running does nothing if RT-DETR is already in.

The code cell body is read from notebooks/rtdetr_addon.py so the logic lives in one
reviewable place. Run:  python scripts/add_rtdetr_cells.py
"""
import json
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks"
ADDON = (NB_DIR / "rtdetr_addon.py").read_text(encoding="utf-8")
NOTEBOOKS = ["colab_train_evaluate.ipynb", "local_train_evaluate.ipynb"]

MARKDOWN = (
    "## Part 2.4 / 3.9 - RT-DETR (Real-Time Detection Transformer)\n\n"
    "A 4th comparison model: a **transformer-based** real-time detector (Ultralytics "
    "`RTDETR`, ~32M params). Unlike the YOLO family (convolutional, anchor-free) and "
    "SSDLite (anchor-based), RT-DETR uses **transformer object queries** for end-to-end "
    "detection (no NMS). It is trained on the **same** data.yaml, train/val/test split, "
    "image size (640) and epoch budget as the others for a fair comparison. Labels stay "
    "`0..4` - the `+1` background shift is **only** for torchvision SSDLite, never for "
    "RT-DETR. This cell is self-contained: train -> evaluate on the unseen test set -> "
    "per-class AP -> confidence sweep -> prediction samples -> merge into "
    "`runs/model_comparison.csv`."
)


def make_cell(cell_type, source):
    cell = {"cell_type": cell_type, "metadata": {}, "source": source.splitlines(keepends=True)}
    cell["id"] = uuid.uuid4().hex[:8]
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def patch(path: Path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    cells = nb["cells"]
    if any("RTDETR" in "".join(c.get("source", [])) for c in cells):
        print(f"  {path.name}: RT-DETR already present - skipped")
        return
    # insert before the 'Part 4' summary markdown; else before the last cell
    idx = next((i for i, c in enumerate(cells)
                if c["cell_type"] == "markdown" and "Part 4" in "".join(c["source"])),
               len(cells) - 1)
    cells[idx:idx] = [make_cell("markdown", MARKDOWN), make_cell("code", ADDON)]
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"  {path.name}: inserted RT-DETR cells at index {idx}")


if __name__ == "__main__":
    for name in NOTEBOOKS:
        p = NB_DIR / name
        if p.exists():
            patch(p)
        else:
            print(f"  {name}: NOT FOUND - skipped")
