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

# ATM face-cover detection

[Open website](https://vaibhavsh0120-atm-theft-detection.hf.space) · [Model and checkpoint](https://huggingface.co/Vaibhavsh0120/ATM-Theft-Detection) · [GitHub source](https://github.com/Vaibhavsh0120/ATM-Theft-Detection)

A CPU Gradio research demo for photo, short-video and streamed webcam inference. Detects `Face_Covered` and `Face_Uncovered`; it does not establish theft, intent or identity. Camera frames and uploaded media are processed on this server. Use media you have permission to share.

## Usage

- **Photo:** upload or capture an image, then run inference.
- **Short Video:** upload or record a clip up to 30 seconds / 25 MB. CPU processing can take time.
- **Live Inference:** allow browser camera access, then stream frames. HTTPS is required remotely.

Inference modes share one model queue. Gradio expires served cached files hourly. Original annotated videos older than an hour are cleaned on the next video request; do not assume immediate deletion.

## Model configuration

Precedence: explicit `MODEL_PATH` → explicit `HF_MODEL_REPO_ID` → bundled `weights/best.pt` → sibling monorepo model folder. `HF_MODEL_FILENAME` defaults to `weights/best.pt`. Configure these as Space variables; use a Space secret `HF_TOKEN` for private model downloads. The standalone app does not load a root `.env`.

## Run locally

Use Python 3.11 and an activated virtual environment in this folder:

```bash
python -m pip install -r requirements.txt
python app.py
```

Bundled weights require Git LFS when cloning. The free CPU Space may sleep when idle and take time to start. See the [source publishing guide](https://github.com/Vaibhavsh0120/ATM-Theft-Detection/blob/main/docs/publishing.md) for deployment and training instructions.
