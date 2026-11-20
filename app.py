"""Streamlit front end: Dense Retail Shelf Product Detection & Out-of-Stock Auditing."""
import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

from src.detector import detect, load_model
from src.gaps import find_gaps
from src.viz import draw_audit

st.set_page_config(page_title="Shelf Out-of-Stock Auditor", page_icon="🛒", layout="wide")


@st.cache_resource(show_spinner="Loading model...")
def get_model():
    return load_model()


st.title("🛒 Dense Retail Shelf Detection & Out-of-Stock Auditing")
st.caption("YOLOv8 detects tightly packed products; spatial clustering finds the empty shelf gaps.")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Detection")
    conf = st.slider("Confidence threshold", 0.05, 0.90, 0.25, 0.05)
    iou = st.slider("NMS IoU threshold", 0.30, 0.90, 0.60, 0.05,
                    help="Higher keeps more overlapping boxes - useful on dense shelves.")
    imgsz = st.select_slider("Inference size", [640, 800, 960, 1280], value=960)
    max_det = st.slider("Max detections", 100, 1000, 600, 50)

    st.header("Gap analysis")
    row_tol = st.slider("Row clustering tolerance", 0.2, 1.2, 0.6, 0.05,
                        help="Lower splits shelves into more rows.")
    gap_factor = st.slider("Gap size (x product width)", 0.3, 2.0, 0.6, 0.05,
                           help="Empty space must exceed this to count as out-of-stock.")
    min_row_items = st.slider("Min products per row", 2, 10, 3)
    check_edges = st.checkbox("Check shelf edges", True)
    show_boxes = st.checkbox("Show product boxes", True)

# ---------------- Input ----------------
model, source = get_model()
st.info(f"Model weights: **{source}**")

uploaded = st.file_uploader("Upload a shelf photo", type=["jpg", "jpeg", "png", "webp"])
sample_dir = Path(__file__).parent / "samples"
samples = sorted(p for p in sample_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
sample_choice = st.selectbox("...or try a sample", ["-"] + [p.name for p in samples]) if samples else "-"

image = None
if uploaded is not None:
    image = ImageOps.exif_transpose(Image.open(uploaded)).convert("RGB")
elif sample_choice != "-":
    image = Image.open(sample_dir / sample_choice).convert("RGB")

if image is None:
    st.stop()

# ---------------- Inference + audit ----------------
with st.spinner("Detecting products..."):
    boxes, scores = detect(model, np.array(image), conf, iou, imgsz, max_det)
gaps, rows = find_gaps(boxes, image.width, row_tol, gap_factor, min_row_items, check_edges)

audited = draw_audit(image, boxes, gaps, show_boxes)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Products detected", len(boxes))
c2.metric("Shelf rows", len(rows))
c3.metric("Gaps flagged", len(gaps))
c4.metric("Est. missing facings", int(sum(g.est_missing for g in gaps)))

left, right = st.columns(2)
left.subheader("Original")
left.image(image, use_container_width=True)
right.subheader("Audit result (red = out-of-stock gap)")
right.image(audited, use_container_width=True)

# ---------------- Report ----------------
st.subheader("Gap report")
if gaps:
    df = pd.DataFrame([g.as_dict() for g in gaps])
    df["row"] = df["row"] + 1
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button("Download gap report (CSV)", df.to_csv(index=False).encode(),
                       "out_of_stock_report.csv", "text/csv")
else:
    st.success("No out-of-stock gaps found with the current settings.")

buf = io.BytesIO()
audited.save(buf, format="PNG")
st.download_button("Download annotated image", buf.getvalue(), "shelf_audit.png", "image/png")
