"""
Inference helpers used by the Streamlit UI.
Supports YOLOv11n, YOLOv8n (Ultralytics) and SSDLite320-MobileNetV3 (torchvision).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CLASSES = ['cracks', 'spalling', 'corrosion', 'potholes', 'paint_degradation']
CLASS_COLORS = [
    (231, 76,  60),   # red    — cracks
    ( 52,152,219),    # blue   — spalling
    ( 46,204,113),    # green  — corrosion
    (243,156, 18),    # orange — potholes
    (155, 89,182),    # purple — paint_degradation
]

ModelType = Literal['YOLOv11s', 'YOLOv8s', 'YOLOv11n', 'YOLOv8n', 'SSDLite']

# Maps the UI model name to its runs/ subfolder.
MODEL_DIRS: dict[str, str] = {
    'YOLOv11s': 'yolo11s',   # v3 small model — named-stage folder structure
    'YOLOv8s':  'yolov8s',   # v3 small model — named-stage folder structure
    'YOLOv11n': 'yolo11n',   # legacy nano — kept for old-data baseline comparison
    'YOLOv8n':  'yolov8n',   # legacy nano — kept for old-data baseline comparison
    'SSDLite':  'ssdlite',
}

# Named-stage models (v3): folders are 'baseline_640', 'finetune_768', not 'v1','v2',...
_NAMED_STAGE_MODELS = {'YOLOv11s', 'YOLOv8s'}


def _yolo_version_map50(version_dir: Path) -> float | None:
    """Best mAP@0.5 recorded in a YOLO run's results.csv, or None."""
    import csv as _csv
    results = version_dir / 'results.csv'
    if not results.exists() or results.stat().st_size == 0:
        return None
    try:
        with open(results) as f:
            vals = [float(r.get('metrics/mAP50(B)', 0)) for r in _csv.DictReader(f)]
        return max(vals) if vals else None
    except Exception:
        return None


def _ssd_version_map50(version_num: str) -> float | None:
    """mAP@0.5 for an SSDLite version, looked up in runs/run_log.csv."""
    log = PROJECT_ROOT / 'runs' / 'run_log.csv'
    if not log.exists():
        return None
    import csv as _csv
    try:
        with open(log) as f:
            for r in _csv.DictReader(f):
                if (r.get('model', '').lower() == 'ssdlite'
                        and str(r.get('version')) == str(version_num)):
                    return float(r.get('map50', 0))
    except Exception:
        return None
    return None


def _named_stage_map50(stage_dir: Path) -> float | None:
    """mAP@0.5 from a named-stage run's results.csv (e.g. baseline_640/)."""
    return _yolo_version_map50(stage_dir)


def list_versions(model_type: ModelType) -> list[tuple[str, str | None]]:
    """Returns [(label, weights_path_or_None)] for the UI version dropdown.
    The first entry is always ('Best (auto)', None), which defers to the
    automatic best-version selection in load_model(). Subsequent entries are
    concrete trained runs, annotated with mAP@0.5 when known."""
    out: list[tuple[str, str | None]] = [('Best (auto)', None)]
    model_dir = PROJECT_ROOT / 'runs' / MODEL_DIRS[model_type]
    if not model_dir.exists():
        return out

    if model_type in _NAMED_STAGE_MODELS:
        # v3 small models: named stages baseline_640 (default), finetune_768 (ablation)
        for stage_name in ('baseline_640', 'finetune_768'):
            d = model_dir / stage_name
            w = d / 'weights' / 'best.pt'
            if not w.exists():
                continue
            m = _named_stage_map50(d)
            suffix = '  (mAP@0.5 %.3f)' % m if m is not None else ''
            tag = '' if stage_name == 'baseline_640' else '  [ablation]'
            out.append((stage_name + suffix + tag, str(w)))
    elif model_type in ('YOLOv11n', 'YOLOv8n'):
        versions = sorted(
            [d for d in model_dir.iterdir() if d.is_dir() and d.name.startswith('v')],
            key=lambda d: int(d.name[1:]),
        )
        for v in versions:
            w = v / 'weights' / 'best.pt'
            if not w.exists():
                continue
            m = _yolo_version_map50(v)
            label = v.name + (f'  (mAP@0.5 {m:.3f})' if m is not None else '')
            out.append((label, str(w)))
    else:  # SSDLite
        ckpts = sorted(model_dir.glob('best_v*.pth'),
                       key=lambda p: int(p.stem.split('_v')[1]))
        for c in ckpts:
            num = c.stem.split('_v')[1]
            m = _ssd_version_map50(num)
            label = f'v{num}' + (f'  (mAP@0.5 {m:.3f})' if m is not None else '')
            out.append((label, str(c)))
    return out


def _best_yolo_weights(model_dir: Path) -> Path:
    """Return best.pt for a YOLO model directory.

    Named-stage models (YOLOv11s/YOLOv8s, v3): returns baseline_640/weights/best.pt.
    finetune_768 is excluded from auto-selection — it regressed on v3 validation.
    Legacy models (YOLOv11n/YOLOv8n): returns best.pt from the vN folder with the
    highest mAP@0.5 in results.csv.
    """
    # Named-stage (v3 small models): always prefer baseline_640
    baseline = model_dir / 'baseline_640' / 'weights' / 'best.pt'
    if baseline.exists():
        return baseline

    # Legacy vN structure
    import csv as _csv
    versions = sorted(
        [d for d in model_dir.iterdir() if d.is_dir() and d.name.startswith('v')],
        key=lambda d: int(d.name[1:])
    )
    if not versions:
        raise FileNotFoundError(f'No training runs found in {model_dir}')
    best_ver, best_map = versions[-1], -1.0
    for v in versions:
        results = v / 'results.csv'
        if not results.exists() or results.stat().st_size == 0:
            continue
        try:
            with open(results) as f:
                rows = list(_csv.DictReader(f))
            if not rows:
                continue
            map50s = [float(r.get('metrics/mAP50(B)', 0)) for r in rows]
            v_best = max(map50s)
            if v_best > best_map:
                best_map, best_ver = v_best, v
        except Exception:
            continue
    return best_ver / 'weights' / 'best.pt'


def _latest_torch_weights(model_dir: Path) -> Path:
    """Return the highest-numbered best_vN.pth file."""
    ckpts = sorted(
        model_dir.glob('best_v*.pth'),
        key=lambda p: int(p.stem.split('_v')[1])
    )
    if not ckpts:
        raise FileNotFoundError(f'No checkpoints found in {model_dir}')
    return ckpts[-1]


@dataclass
class Detection:
    class_id:   int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    severity:   str


def _compute_severity(x1: float, y1: float, x2: float, y2: float,
                      img_w: int, img_h: int) -> str:
    area_pct = ((x2 - x1) * (y2 - y1)) / (img_w * img_h) * 100
    if area_pct < 5:
        return 'Low'
    elif area_pct <= 20:
        return 'Medium'
    return 'High'


def load_model(model_type: ModelType, device: torch.device,
               weights: str | Path | None = None):
    """Loads the trained model for the given type. Returns (model, backend).

    weights: explicit path to a specific run's weights. When None, the best
    version is auto-selected (highest mAP@0.5 for YOLO, latest for SSDLite),
    preserving the previous behaviour. This is what powers UI version picking."""
    runs = PROJECT_ROOT / 'runs'

    if model_type in ('YOLOv11s', 'YOLOv8s', 'YOLOv11n', 'YOLOv8n'):
        from ultralytics import YOLO
        w = Path(weights) if weights else _best_yolo_weights(runs / MODEL_DIRS[model_type])
        if not w.exists():
            raise FileNotFoundError(f'{model_type} weights not found: {w}')
        return YOLO(str(w)), 'yolo'

    if model_type == 'SSDLite':
        from models.ssdlite_detector import build_model
        w = Path(weights) if weights else _latest_torch_weights(runs / 'ssdlite')
        if not w.exists():
            raise FileNotFoundError(f'SSDLite weights not found: {w}')
        model = build_model(num_classes=6, pretrained=False).to(device)
        ckpt = torch.load(str(w), map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        return model, 'torchvision'

    raise ValueError(f'Unknown model type: {model_type}')


def run_inference(
    image: Image.Image,
    model,
    model_backend: str,
    device: torch.device,
    conf_thresh: float = 0.25,
) -> list[Detection]:
    """Runs inference on a PIL image. Returns a list of Detection objects."""
    img_w, img_h = image.size

    if model_backend == 'yolo':
        results = model.predict(np.array(image), conf=conf_thresh, verbose=False)[0]
        detections = []
        for box in results.boxes:
            cls  = int(box.cls.item())
            conf = float(box.conf.item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(Detection(
                class_id=cls,
                class_name=CLASSES[cls] if cls < len(CLASSES) else 'unknown',
                confidence=round(conf, 3),
                x1=x1, y1=y1, x2=x2, y2=y2,
                severity=_compute_severity(x1, y1, x2, y2, img_w, img_h),
            ))
        return detections

    # torchvision models
    tensor = TF.to_tensor(image.resize((640, 640))).unsqueeze(0).to(device)
    scale_x = img_w / 640
    scale_y = img_h / 640
    with torch.no_grad():
        outputs = model(tensor)[0]

    detections = []
    for i in range(len(outputs['scores'])):
        score = float(outputs['scores'][i].item())
        if score < conf_thresh:
            continue
        cls = int(outputs['labels'][i].item()) - 1    # undo background offset
        if not (0 <= cls < len(CLASSES)):
            continue
        x1, y1, x2, y2 = outputs['boxes'][i].tolist()
        x1 *= scale_x; x2 *= scale_x
        y1 *= scale_y; y2 *= scale_y
        detections.append(Detection(
            class_id=cls,
            class_name=CLASSES[cls],
            confidence=round(score, 3),
            x1=x1, y1=y1, x2=x2, y2=y2,
            severity=_compute_severity(x1, y1, x2, y2, img_w, img_h),
        ))
    return detections


def draw_detections(image: Image.Image, detections: list[Detection]) -> Image.Image:
    """Draws bounding boxes with labels on a copy of the image."""
    import cv2
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    for det in detections:
        color = CLASS_COLORS[det.class_id % len(CLASS_COLORS)]
        bgr   = (color[2], color[1], color[0])
        pt1   = (int(det.x1), int(det.y1))
        pt2   = (int(det.x2), int(det.y2))
        cv2.rectangle(img_cv, pt1, pt2, bgr, 2)
        label = f"{det.class_name} {det.confidence:.2f} [{det.severity}]"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img_cv, (pt1[0], pt1[1] - th - 6), (pt1[0] + tw, pt1[1]), bgr, -1)
        cv2.putText(img_cv, label, (pt1[0], pt1[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
