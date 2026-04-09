---
title: ATM Theft Detection
emoji: "🛡️"
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: "6.11.0"
python_version: "3.11"
app_file: app.py
suggested_hardware: cpu-basic
short_description: Detect covered and uncovered faces in ATM footage with a custom YOLOv8 model.
tags:
  - computer-vision
  - object-detection
  - yolo
  - gradio
  - security
fullWidth: true
pinned: false
startup_duration_timeout: 1h
---

# ATM Theft Detection

This repository contains a custom YOLOv8 model for ATM security monitoring. The model detects two classes:

- `Face_Covered`
- `Face_Uncovered`

The repo is now structured so it can be deployed directly as a Hugging Face Gradio Space while still keeping the original local training and quantization scripts.

## Hugging Face Space Setup

The Space entry point is [`app.py`](app.py). The Space metadata is the YAML block at the top of this [`README.md`](README.md), which Hugging Face reads automatically.

This setup assumes:

- the model weights are available at [`weights/best.pt`](weights/best.pt)
- the Space installs Python dependencies from [`requirements.txt`](requirements.txt)
- inference runs on CPU by default, which fits `cpu-basic`

### Deploy

1. Create a new Hugging Face Space and choose the `Gradio` SDK.
2. Push this repository to the Space repo.
3. Hugging Face will read the YAML front matter in [`README.md`](README.md), install [`requirements.txt`](requirements.txt), and launch [`app.py`](app.py).

The web app supports:

- photo inference
- short video inference
- live browser webcam inference

## Local Development

Use the single [`requirements.txt`](requirements.txt) file for both the Hugging Face Space and local development:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

For the legacy desktop OpenCV webcam script in [`live_inference.py`](live_inference.py), replace `opencv-python-headless` with `opencv-python` in your local environment if you need native windowed webcam display.

## Original Training Workflow

### 1. Configure Roboflow access

Copy [`.env.example`](.env.example) to `.env` and set:

```text
ROBOFLOW_API_KEY="YOUR_API_KEY"
```

### 2. Download dataset and model assets

```bash
python setup.py
```

This downloads the Roboflow dataset and ensures the model weights are available.

### 3. Merge classes

```bash
python merge_classes.py
```

This converts the source dataset into the two final labels used by the model.

### 4. Train

```bash
python train.py
```

### 5. Local webcam inference

If you want the original desktop webcam loop, keep `opencv-python` installed locally and run:

```bash
python live_inference.py
```

## Repository Layout

- [`app.py`](app.py): Hugging Face Space Gradio app
- [`requirements.txt`](requirements.txt): shared dependencies for the Space and local workflows
- [`setup.py`](setup.py): Roboflow dataset download and weight bootstrap
- [`merge_classes.py`](merge_classes.py): class remapping for the dataset
- [`train.py`](train.py): YOLOv8 training script
- [`live_inference.py`](live_inference.py): local OpenCV webcam inference
- [`live_quantize.py`](live_quantize.py): local TFLite webcam inference
- [`quantize.py`](quantize.py): model export for quantized deployment
- [`check_quantization.py`](check_quantization.py): TFLite quantization verification

## Notes

- The Hugging Face Space path is inference-only. Training and Roboflow dataset download are not part of the Space startup.
- The current model file is small enough to live in the repo directly. If future weights grow substantially, move them to a dedicated model repo or Git LFS.
- Short video clips are the best fit for `cpu-basic`; long videos will be slow.
