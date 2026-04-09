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
short_description: YOLOv8 ATM face-cover detection demo.
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

This folder is meant to be pushed as a standalone Hugging Face Space repository.

## Runtime Requirements

- `app.py` is the Space entry point.
- `requirements.txt` contains the Space-only dependencies.
- The app looks for model weights in this order:
  1. `MODEL_PATH`
  2. `weights/best.pt`
  3. the sibling monorepo path `../huggingface-model/weights/best.pt`
  4. a Hugging Face model repo when `HF_MODEL_REPO_ID` is configured

## Recommended Space Variables

- `HF_MODEL_REPO_ID`: Hugging Face model repo that stores `weights/best.pt`
- `HF_MODEL_FILENAME`: optional override, defaults to `weights/best.pt`

If you prefer bundling the checkpoint directly inside the Space repo, place it at `weights/best.pt`.

## Local Test

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```
