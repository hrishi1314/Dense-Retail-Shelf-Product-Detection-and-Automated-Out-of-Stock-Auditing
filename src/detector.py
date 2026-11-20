"""Model loading and inference.

Weights are resolved in this order:
1. ``models/best.pt`` or ``models/best.onnx`` in the repo (your trained model)
2. Hugging Face Hub, if ``HF_REPO_ID`` is set (env var or Streamlit secret)
3. ``yolov8n.pt`` (generic COCO weights) as a placeholder so the app still runs
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
FALLBACK = "yolov8n.pt"


def resolve_weights() -> tuple[str, str]:
    """Return (path_or_name, human-readable source)."""
    for name in ("best.pt", "best.onnx"):
        p = MODEL_DIR / name
        if p.exists():
            return str(p), f"repo: models/{name}"

    repo_id = os.environ.get("HF_REPO_ID")
    if not repo_id:
        try:  # Streamlit secrets, if running inside Streamlit
            import streamlit as st
            repo_id = st.secrets.get("HF_REPO_ID")
        except Exception:
            repo_id = None
    if repo_id:
        from huggingface_hub import hf_hub_download
        filename = os.environ.get("HF_WEIGHTS_FILE", "best.pt")
        path = hf_hub_download(repo_id=repo_id, filename=filename)
        return path, f"Hugging Face: {repo_id}/{filename}"

    return FALLBACK, "placeholder: generic YOLOv8n (COCO), not shelf-trained"


def load_model():
    """Load the YOLO model. Returns (model, source_description)."""
    from ultralytics import YOLO
    path, source = resolve_weights()
    return YOLO(path), source


def detect(model, image_rgb: np.ndarray, conf: float = 0.25, iou: float = 0.6,
           imgsz: int = 960, max_det: int = 600):
    """Run detection on an RGB image.

    Returns (boxes_xyxy (N,4) float array, scores (N,) float array).
    ``iou`` is the NMS threshold and ``max_det`` the cap on detections; both
    matter on dense shelves where products overlap heavily.
    """
    res = model.predict(image_rgb, conf=conf, iou=iou, imgsz=imgsz,
                        max_det=max_det, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return np.zeros((0, 4)), np.zeros((0,))
    return res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()
