"""Update the three all-in-one notebooks for the two-stage YOLO training (Phases 1-2).

Idempotent and self-migrating. Inlines code from scripts/prepare_training_data.py and
scripts/train_yolo_two_stage.py (Colab/Kaggle only upload data, so the cells must be self-contained):
  - "Part 1.6 Dataset repair and verification"           (before Part 2)
  - "Part 2.1 Training configuration"  = config + shared training functions
  - "Part 2.2 YOLOv11s baseline and fine-tuning"         = YOLOv11s run block
  - "Part 2.3 YOLOv8s baseline and fine-tuning"          = YOLOv8s run block (independent)

Parts 2.2 and 2.3 are pure run blocks that reuse the functions from Part 2.1, so YOLOv8s can run
without YOLOv11s. SSDLite, RT-DETR and Part 3 are left untouched (later phases). Run:
    python scripts/update_training_notebooks.py
"""
import json
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks"
NOTEBOOKS = ["local_train_evaluate.ipynb", "colab_train_evaluate.ipynb", "kaggle_train_evaluate.ipynb"]


def region(text, name):
    a, b = "# === BEGIN %s ===" % name, "# === END %s ===" % name
    return text[text.index(a) + len(a):text.index(b)].strip("\n")


REPAIR = region((ROOT / "scripts" / "prepare_training_data.py").read_text(encoding="utf-8"), "repair")
_TS = (ROOT / "scripts" / "train_yolo_two_stage.py").read_text(encoding="utf-8")
CONFIG = region(_TS, "config")
FUNCTIONS = region(_TS, "functions")
YOLOV11S_RUN = region(_TS, "yolov11s_run")
YOLOV8S_RUN = region(_TS, "yolov8s_run")
FINAL_TEST_EVAL = region(_TS, "final_test_eval")

P16_MD = (
    "## Part 1.6 Dataset repair and verification\n\n"
    "Repairs the dataset before training: convert/remove GIF-as-JPG, re-encode corrupt JPEGs, drop "
    "duplicate/invalid label lines, remove unopenable images. **`DRY_RUN=True` by default** (reports "
    "only). Set `DRY_RUN=False` to apply, or run `python scripts/prepare_training_data.py --apply`."
)
P16_DRIVER = """
# Part 1.6 driver
DRY_RUN = True   # set False to actually modify data/ (labels are git-tracked; images are not)
if DRY_RUN:
    print('DRY_RUN=True: no files were changed. Set DRY_RUN=False or run --apply to apply repairs.')
else:
    print('APPLYING changes to data/ (labels are git-tracked).')
_summary = prepare_dataset(DATA_DIR, apply=not DRY_RUN)
print('\\nRepair summary')
for _k in ['files_checked', 'images_fixed', 'images_removed',
           'duplicate_labels_removed', 'invalid_labels_removed']:
    print('  %-26s %d' % (_k, _summary[_k]))
print()
verify_dataset(DATA_DIR)
"""
P16_CODE = REPAIR + "\n\n" + P16_DRIVER.strip("\n")

P21_MD = (
    "## Part 2.1 Training configuration\n\n"
    "Run switches and shared training functions used by Parts 2.2 and 2.3. "
    "**Default workflow: `baseline_640` only.** The 768 fine-tune (`RUN_FINETUNE_768`) is disabled "
    "by default — it was tested on v2 and v3 and regressed vs the 640 baseline. Enable it only for "
    "ablation experiments. `CACHE_MODE=\"disk\"` speeds loading; `FORCE_RETRAIN=True` ignores "
    "existing checkpoints."
)
P21_CODE = CONFIG + "\n\n" + FUNCTIONS

P22_MD = (
    "## Part 2.2 YOLOv11s baseline_640\n\n"
    "Trains `baseline_640` (640px, 110 epochs, patience=25, AdamW, cosine LR). Checkpoints every "
    "5 epochs; re-run to resume from `last.pt`. Validation metrics go to "
    "`runs/experiment_tracker.csv`; test is scored only in Part 3. Runs when `RUN_YOLO11S=True`.\n\n"
    "**Note:** the 768 fine-tune stage was tested on v2 and v3 but did not improve validation mAP "
    "(v3: finetune_768 0.441 vs baseline_640 0.461). Available as optional ablation via "
    "`RUN_FINETUNE_768=True` but not part of the default workflow."
)
P22_CODE = YOLOV11S_RUN

P23_MD = (
    "## Part 2.3 YOLOv8s baseline_640\n\n"
    "Same `baseline_640` flow as Part 2.2 but for YOLOv8s, in its own independent folders "
    "(`runs/yolov8s/baseline_640`). **Independent of YOLOv11s** — runs whenever `RUN_YOLO8S=True`."
)
P23_CODE = YOLOV8S_RUN

P3_MD = (
    "## Part 3 Final test evaluation\n\n"
    "Scores the **test set once** for each trained model and writes `runs/model_comparison.csv`. "
    "Uses `baseline_640` as the selected stage (confirmed best on v3 validation). "
    "Falls back to best-by-validation-mAP if the tracker has a higher-scoring alternative. "
    "Run after Parts 2.2 / 2.3. Never mix test rows into `experiment_tracker.csv`."
)
P3_CODE = FINAL_TEST_EVAL

# Phase 4: severity_label moves to Part 0 (the optional RT-DETR cell uses it once the old Part 3 is gone)
SEVERITY_DEF = (
    "# severity from bbox area fraction (Low <5%, Medium 5-20%, High >20%)\n"
    "def severity_label(bw, bh):\n"
    "    a = bw * bh * 100\n"
    "    return 'Low' if a < 5 else ('Medium' if a <= 20 else 'High')"
)


def guard(src, switch, label):
    """Wrap a whole code cell in `if <switch>:` so it is skipped unless the switch is on."""
    body = "\n".join(("    " + ln) if ln.strip() else ln for ln in src.splitlines())
    return "if %s:\n%s\nelse:\n    print(%r)" % (switch, body, label)


def make_cell(cell_type, source):
    cell = {"cell_type": cell_type, "metadata": {}, "id": uuid.uuid4().hex[:8],
            "source": source.splitlines(keepends=True)}
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def find(cells, pred):
    return next((i for i, c in enumerate(cells) if pred(c)), None)


def find_code(cells, substr):
    return find(cells, lambda c: c["cell_type"] == "code" and substr in "".join(c["source"]))


def set_source(cell, text):
    if "".join(cell["source"]) == text:
        return False
    cell["source"] = text.splitlines(keepends=True)
    if cell["cell_type"] == "code":
        cell["outputs"] = []
        cell["execution_count"] = None
    return True


def set_heading(cells, code_idx, key, md_text):
    """Set the markdown cell immediately above code_idx if it's the matching heading."""
    j = code_idx - 1
    if j >= 0 and cells[j]["cell_type"] == "markdown" and key in "".join(cells[j]["source"]):
        return set_source(cells[j], md_text)
    return False


def _is_training_region_cell(c):
    """True for any markdown/code cell belonging to Part 2.1/2.2/2.3 (config, functions, run blocks).

    Used to wipe the region and re-insert it cleanly, so repeated runs cannot accumulate
    duplicate headings or leave the config/run cells out of order (the old per-cell patch did)."""
    s = "".join(c["source"])
    if c["cell_type"] == "markdown":
        return any(k in s for k in (
            "## Part 2.1 Training configuration",
            "## Part 2.2 YOLOv11s",
            "## Part 2.3 YOLOv8s",
        ))
    return any(k in s for k in (
        "RUN_FINETUNE_768",                  # config cell (definition or print)
        "def train_baseline_640",            # shared functions cell
        "train_baseline_640('yolo11s.pt'",   # YOLOv11s run block
        "train_baseline_640('yolov8s.pt'",   # YOLOv8s run block
    ))


def rebuild_training_region(cells, acts):
    """Deterministically rebuild Part 2.1/2.2/2.3 as one ordered, de-duplicated block.

    Removes every existing 2.1/2.2/2.3 cell, then inserts exactly:
      2.1 md -> config+functions code -> 2.2 md -> YOLOv11s run -> 2.3 md -> YOLOv8s run
    Anchored after the Part 2.0 helpers cell (or the '## Part 2' heading) on a fresh notebook."""
    want = [("markdown", P21_MD), ("code", P21_CODE), ("markdown", P22_MD),
            ("code", P22_CODE), ("markdown", P23_MD), ("code", P23_CODE)]
    idxs = [i for i, c in enumerate(cells) if _is_training_region_cell(c)]
    # No-op when the region is already exactly the 6 wanted cells, in order, with matching content.
    if len(idxs) == len(want) and idxs == list(range(idxs[0], idxs[0] + len(want))):
        if all(cells[idxs[k]]["cell_type"] == t and "".join(cells[idxs[k]]["source"]) == s
               for k, (t, s) in enumerate(want)):
            return
    if idxs:
        insert_at = idxs[0]
        for i in reversed(idxs):
            del cells[i]
    else:
        anchor = find_code(cells, "def train_yolo")
        if anchor is None:
            anchor = find(cells, lambda c: c["cell_type"] == "markdown" and "## Part 2 " in "".join(c["source"]))
        insert_at = (anchor + 1) if anchor is not None else len(cells)
    cells[insert_at:insert_at] = [make_cell(t, s) for t, s in want]
    acts.append("rebuilt Part 2.1-2.3 (%d removed, 6 inserted)" % len(idxs))


def patch(path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    cells = nb["cells"]
    acts = []

    # Part 1.6 - insert before Part 2 if missing
    if find_code(cells, "prepare_dataset(DATA_DIR") is None:
        i = find(cells, lambda c: c["cell_type"] == "markdown" and "## Part 2 " in "".join(c["source"]))
        if i is None:
            i = find(cells, lambda c: c["cell_type"] == "markdown" and "Part 2" in "".join(c["source"]))
        cells[i:i] = [make_cell("markdown", P16_MD), make_cell("code", P16_CODE)]
        acts.append("inserted 1.6")

    # Part 2.1/2.2/2.3 - rebuild the whole region in one deterministic, de-duplicated pass.
    # (The old per-cell patch could leave duplicate headings and omit the run cells entirely.)
    rebuild_training_region(cells, acts)

    # Part 3 - final test evaluation (update code if present; insert if missing)
    i3 = find_code(cells, "def evaluate_on_test")
    if i3 is None:
        i = find(cells, lambda c: c["cell_type"] == "markdown" and "## Part 3" in "".join(c["source"]))
        if i is not None:
            set_source(cells[i], P3_MD)
            cells[i + 1:i + 1] = [make_cell("code", P3_CODE)]
            acts.append("inserted Part 3 final test eval")
    else:
        if set_source(cells[i3], P3_CODE):
            acts.append("set Part 3 code")
        set_heading(cells, i3, "Part 3", P3_MD)

    # ---- Phase 4 cleanup ----
    # severity_label -> Part 0 (the optional RT-DETR cell needs it after the old Part 3 is removed)
    i = find_code(cells, "def write_data_yaml")
    if i is not None and "def severity_label" not in "".join(cells[i]["source"]):
        cells[i]["source"] = ("".join(cells[i]["source"]).rstrip("\n") + "\n\n"
                              + SEVERITY_DEF + "\n").splitlines(keepends=True)
        acts.append("severity_label -> Part 0")

    # remove the superseded basic-768 cell (markdown + code)
    rm = [k for k, c in enumerate(cells) if "finetune_hires" in "".join(c["source"])
          or (c["cell_type"] == "markdown" and "Steps 4-5 - 768-px" in "".join(c["source"]))]
    for k in sorted(rm, reverse=True):
        del cells[k]
    if rm:
        acts.append("removed basic-768 cell")

    # remove the old Part 3 eval cells (# 3.1 .. # 3.8) - superseded by the new Part 3
    import re as _re
    rm = [k for k, c in enumerate(cells) if c["cell_type"] == "code"
          and _re.match(r"# 3\.\d", "".join(c["source"]).lstrip())]
    for k in sorted(rm, reverse=True):
        del cells[k]
    if rm:
        acts.append("removed old Part 3 cells (%d)" % len(rm))

    # guard the SSDLite training cell and the RT-DETR cell with their RUN_* switch
    i = find_code(cells, "# 2.3b SSDLite training")
    if i is not None and not "".join(cells[i]["source"]).lstrip().startswith("if RUN_SSDLITE:"):
        set_source(cells[i], guard("".join(cells[i]["source"]), "RUN_SSDLITE",
                                   "RUN_SSDLITE=False - skipping SSDLite training"))
        acts.append("guarded SSDLite")
    i = find(cells, lambda c: c["cell_type"] == "code" and "RT-DETR (transformer detector)" in "".join(c["source"]))
    if i is not None and not "".join(cells[i]["source"]).lstrip().startswith("if RUN_RTDETR:"):
        set_source(cells[i], guard("".join(cells[i]["source"]), "RUN_RTDETR",
                                   "RUN_RTDETR=False - skipping RT-DETR"))
        acts.append("guarded RT-DETR")

    # renumber the optional-model headings (avoid the Part 2.3 collision; mark them optional)
    for c in cells:
        if c["cell_type"] != "markdown":
            continue
        s = "".join(c["source"])
        if "### 2.3 SSDLite320-MobileNetV3" in s:
            set_source(c, s.replace("### 2.3 SSDLite320-MobileNetV3",
                                    "## Part 2.4 SSDLite320-MobileNetV3 (optional - set RUN_SSDLITE=True)"))
            acts.append("renumbered SSDLite -> 2.4")
        elif "## Part 2.4 / 3.9 - " in s:
            set_source(c, s.replace("## Part 2.4 / 3.9 - ",
                                    "## Optional (set RUN_RTDETR=True) - "))
            acts.append("renumbered RT-DETR -> optional")
        elif s.startswith("## Part 2 ") and "all three models" in s:
            set_source(c, s.replace("(all three models, 110 epochs)",
                                    "(YOLOv11s + YOLOv8s two-stage; SSDLite/RT-DETR optional)"))
            acts.append("updated Part 2 heading")

    if acts:
        path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print("  %s: %s" % (path.name, "; ".join(acts) if acts else "already up to date"))


if __name__ == "__main__":
    for name in NOTEBOOKS:
        p = NB_DIR / name
        if p.exists():
            patch(p)
        else:
            print("  %s: NOT FOUND - skipped" % name)
