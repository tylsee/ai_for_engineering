"""
Streamlit dashboard for structural defect detection.
Supports image upload and live webcam inference.
Run with: streamlit run ui/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from inference import (
    CLASSES,
    CLASS_COLORS,
    Detection,
    draw_detections,
    load_model,
    run_inference,
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEVERITY_COLORS = {'Low': '🟢', 'Medium': '🟡', 'High': '🔴'}
CLASS_HEX = [f'#{r:02x}{g:02x}{b:02x}' for r, g, b in CLASS_COLORS]

st.set_page_config(
    page_title='Structural Defect Detector',
    page_icon='🏗️',
    layout='wide',
)


@st.cache_resource(show_spinner='Loading model weights…')
def get_model(model_type: str):
    return load_model(model_type, DEVICE)


def render_detection_table(detections: list[Detection]):
    if not detections:
        st.info('No defects detected above the confidence threshold.')
        return
    rows = []
    for d in sorted(detections, key=lambda x: -x.confidence):
        rows.append({
            'Class':      d.class_name,
            'Confidence': f'{d.confidence:.3f}',
            'Severity':   f"{SEVERITY_COLORS[d.severity]} {d.severity}",
            'BBox (x1,y1,x2,y2)': f'({d.x1:.0f},{d.y1:.0f},{d.x2:.0f},{d.y2:.0f})',
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)


def render_severity_summary(detections: list[Detection]):
    counts = {'Low': 0, 'Medium': 0, 'High': 0}
    for d in detections:
        counts[d.severity] += 1
    col1, col2, col3 = st.columns(3)
    col1.metric('🟢 Low',    counts['Low'])
    col2.metric('🟡 Medium', counts['Medium'])
    col3.metric('🔴 High',   counts['High'])


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title('⚙️ Settings')
    model_choice = st.selectbox(
        'Detection model',
        ['YOLOv11n', 'ResNet50', 'EfficientNetB0'],
        help='YOLOv11n is fastest; ResNet50 is most accurate.'
    )
    conf_thresh = st.slider('Confidence threshold', 0.1, 0.9, 0.25, 0.05)
    st.divider()
    st.markdown('**Classes detected:**')
    for cls, hex_color in zip(CLASSES, CLASS_HEX):
        st.markdown(f'<span style="color:{hex_color}">■</span> {cls}', unsafe_allow_html=True)
    st.divider()
    st.caption(f'Running on: **{DEVICE}**')

# ── Main ──────────────────────────────────────────────────────────────────────
st.title('🏗️ Structural Defect Detection')
st.caption('COS40007 · AI for Engineering · Upload an infrastructure image to detect defects.')

tab_image, tab_compare, tab_about = st.tabs(['Image Analysis', 'Model Comparison', 'About'])

# ── Tab 1: Image Analysis ─────────────────────────────────────────────────────
with tab_image:
    uploaded = st.file_uploader(
        'Upload an image (JPG / PNG)',
        type=['jpg', 'jpeg', 'png'],
        help='Upload a photo of infrastructure to detect structural defects.'
    )

    if uploaded:
        image = Image.open(uploaded).convert('RGB')

        try:
            model, backend = get_model(model_choice)
        except FileNotFoundError as e:
            st.error(f'Model weights not found. Train the model first.\n\n{e}')
            st.stop()

        with st.spinner(f'Running {model_choice} inference…'):
            detections = run_inference(image, model, backend, DEVICE, conf_thresh)
            annotated  = draw_detections(image, detections)

        col_orig, col_pred = st.columns(2)
        with col_orig:
            st.subheader('Original')
            st.image(image, use_container_width=True)
        with col_pred:
            st.subheader(f'Detections ({len(detections)} found)')
            st.image(annotated, use_container_width=True)

        st.subheader('Severity Summary')
        render_severity_summary(detections)

        st.subheader('Detection Details')
        render_detection_table(detections)

    else:
        st.info('Upload an image in the area above to get started.')

# ── Tab 2: Model Comparison ───────────────────────────────────────────────────
with tab_compare:
    st.subheader('Model Performance Comparison')
    comparison_path = PROJECT_ROOT / 'runs' / 'model_comparison.csv'
    if comparison_path.exists():
        df = pd.read_csv(comparison_path, index_col=0)
        st.dataframe(df, use_container_width=True)

        img_path = PROJECT_ROOT / 'runs' / 'model_comparison.png'
        if img_path.exists():
            st.image(str(img_path), use_container_width=True)
    else:
        st.info('Run `notebooks/03_evaluation.ipynb` to generate comparison results.')

    run_log = PROJECT_ROOT / 'runs' / 'run_log.csv'
    if run_log.exists():
        log_df = pd.read_csv(run_log)
        if not log_df.empty:
            st.subheader('Training Run Log')
            st.dataframe(log_df, use_container_width=True)

# ── Tab 3: About ──────────────────────────────────────────────────────────────
with tab_about:
    st.subheader('About This Project')
    st.markdown("""
**COS40007 — Design Project: AI-Based Structural Defect Detection**

This dashboard detects five structural defect classes in infrastructure images:

| # | Class | Description |
|---|-------|-------------|
| 0 | `cracks` | Fractures on surfaces (concrete, asphalt, masonry) |
| 1 | `spalling` | Concrete chipping / surface layer detachment |
| 2 | `corrosion` | Surface staining from oxidation or moisture intrusion |
| 3 | `potholes` | Road surface depressions |
| 4 | `paint_degradation` | Paint peeling or delamination |

### Severity Estimation
Severity is calculated as the fraction of the image covered by each bounding box:

```
area% = (bbox_w × bbox_h) / (img_w × img_h) × 100
Low: < 5% | Medium: 5–20% | High: > 20%
```

### Models
| Model | Architecture | Backbone |
|-------|-------------|---------|
| YOLOv11n | One-stage anchor-free | CSPDarkNet (ImageNet) |
| ResNet50 | Faster R-CNN | ResNet-50 FPN (COCO) |
| EfficientNetB0 | Faster R-CNN | EfficientNet-B0 (ImageNet) |
""")
