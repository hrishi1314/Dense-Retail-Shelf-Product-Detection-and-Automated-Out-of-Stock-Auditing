# Dense Retail Shelf Product Detection & Automated Out-of-Stock Auditing

**CSE471 (Deep Learning for Computer Vision) — Group 10**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/Model-YOLOv8-purple)](https://docs.ultralytics.com)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)](https://streamlit.io)

A computer-vision system for supermarket inventory management. A **YOLOv8** detector finds tightly packed products on a retail shelf, then spatial clustering locates the empty gaps and flags them as out-of-stock vacancies. A **Streamlit** app lets anyone upload a shelf photo and get an annotated audit plus a downloadable CSV report.

**Live app:** _add your share.streamlit.io URL here after deploying_

---

## How it works

```
Upload shelf image
       │
       ▼
  YOLOv8 detector  ──►  (N, 4) bounding boxes + confidence scores
  (src/detector.py)
       │
       ▼
  Row clustering   ──►  products grouped into horizontal shelf rows
  (src/gaps.py)
       │
       ▼
  Gap detection    ──►  empty spaces > gap_factor × median product width
  (src/gaps.py)             → estimated missing facings per gap
       │
       ▼
  Overlay + report ──►  annotated image (green = product, red = gap)
  (src/viz.py)              downloadable CSV
```

1. **Detection** (`src/detector.py`) — YOLOv8 fine-tuned on a Roboflow retail-shelf dataset. Dense-shelf settings: `imgsz=960`, `max_det=600`, tunable NMS IoU.
2. **Row clustering** (`src/gaps.py`) — detections are grouped into shelf rows by their vertical centre using a median-height threshold.
3. **Gap detection** (`src/gaps.py`) — within each row, products are sorted left-to-right; the empty horizontal space between neighbours (and shelf edges) is measured. Space wider than `gap_factor × median product width` is flagged, with an estimate of the missing facings.
4. **Visualisation** (`src/viz.py`) — annotated image with scalable text labels and translucent gap overlays.

> **Note on anchors:** YOLOv8 is anchor-free, so instead of anchor scales we tune `imgsz`, `max_det`, NMS IoU, and mosaic augmentation for dense scenes.

---

## Repo layout

```
app.py                          Streamlit front-end
src/
  detector.py                   Weight loading + inference wrapper
  gaps.py                       Row clustering + gap detection (pure NumPy)
  viz.py                        PIL-based overlay drawing
  __init__.py
notebooks/
  train_yolov8.ipynb            Full Colab training / eval / export notebook
models/
  .gitkeep                      Put best.pt or best.onnx here after training
samples/
  .gitkeep                      Optional demo images shown in the app
tests/
  test_gaps.py                  Unit tests for gap logic (pytest)
  __init__.py
requirements.txt
packages.txt
.streamlit/config.toml
```

---

## Run locally

```bash
git clone https://github.com/hrishi1314/Dense-Retail-Shelf-Product-Detection-and-Automated-Out-of-Stock-Auditing.git
cd Dense-Retail-Shelf-Product-Detection-and-Automated-Out-of-Stock-Auditing
pip install -r requirements.txt
streamlit run app.py
```

> Without trained weights the app falls back to generic `yolov8n` (COCO) so you can test the UI immediately. Results are only meaningful with the shelf-trained model.

---

## Train on Google Colab (free T4 GPU)

1. Open `notebooks/train_yolov8.ipynb` in [Google Colab](https://colab.research.google.com).
2. Set the runtime to **T4 GPU**: *Runtime → Change runtime type → T4 GPU*.
3. Create a free [Roboflow](https://roboflow.com) account and find a retail-shelf dataset on [Roboflow Universe](https://universe.roboflow.com).
4. Paste your Roboflow API key and dataset slug into Cell 3.
5. Run all cells. Download `best.pt` and `best.onnx` from Cell 10.

The notebook saves checkpoints to Google Drive so a Colab disconnect does not lose progress.

---

## Use your trained model

| Option | How |
|---|---|
| **In the repo** | Copy `best.pt` to `models/` (must be < 100 MB for GitHub). |
| **Hugging Face Hub** | Upload weights to a model repo; set `HF_REPO_ID = "username/repo"` in Streamlit secrets (and optionally `HF_WEIGHTS_FILE`). |

---

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** → pick this repo, branch `main`, main file `app.py`.
3. If using Hugging Face Hub, add `HF_REPO_ID` under **Advanced settings → Secrets**.
4. Update the **Live app** badge URL at the top of this README.

> The free Streamlit tier has ~1 GB RAM and no GPU. Use `yolov8s.pt` or the ONNX export if the app runs out of memory.

---

## Tests

```bash
pip install pytest
pytest
```

The unit tests cover row clustering, gap detection, edge gaps, overlapping boxes, and empty inputs — no model or GPU required.

---

## Team

| Reg. No. | Name |
|---|---|
| 12419862 | Rajveer Singh Pathania |
| 12400509 | Karthiksai Kumaraguru |
| 12400528 | Hrishikesh R |
