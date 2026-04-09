# Training Workspace

This folder contains the local-only training workflow for the ATM theft detection model.

## Layout

- `configs/`: dataset YAML files used by Ultralytics
- `data/`: downloaded Roboflow datasets
- `models/pretrained/`: optional local base checkpoints
- `models/exports/`: generated quantized exports
- `outputs/`: Ultralytics training runs and checkpoints
- `scripts/`: setup, remapping, training, and local inference scripts

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r training/requirements.txt
```

Create a root-level `.env` from `.env.example` and set:

```text
ROBOFLOW_API_KEY="YOUR_API_KEY"
```

## Commands

### 1. Download dataset and published model

```bash
python training/scripts/bootstrap_assets.py
```

This populates `training/data/` and ensures `deploy/huggingface-model/weights/best.pt` exists.

### 2. Merge labels into the final 2-class dataset

```bash
python training/scripts/remap_labels.py
```

This step mutates the downloaded label files in place once per fresh dataset
download. If you need to rerun it, run `python training/scripts/bootstrap_assets.py`
first to restore the original Roboflow export.

### 3. Train YOLOv8

```bash
python training/scripts/train_yolo.py
python training/scripts/train_yolo.py --epochs 5
python training/scripts/train_yolo.py --resume
```

Fresh runs are written under incrementing run folders such as:

`training/outputs/runs/detect/train/weights/best.pt`
`training/outputs/runs/detect/train2/weights/best.pt`

Use `--resume` to continue from the latest `last.pt` checkpoint after an interrupted run.
Starting without `--resume` always creates a new run by default.

If `yolov8n.pt` is not already cached, the training script downloads it into
`training/models/pretrained/`.

### 4. Export a quantized TFLite model

```bash
python training/scripts/export_tflite.py
python training/scripts/verify_tflite.py
```

`verify_tflite.py` expects TensorFlow to be available in your local environment.

### 5. Push the Hugging Face model and Space repos

```bash
python scripts/push_hf.py
```

Set `HF_MODEL_REPO_ID` and `HF_SPACE_REPO_ID` in the root `.env` if you want
`push_hf.py` to publish both standalone repos in one command. It pushes the
current deploy folders as-is by default. If you want to refresh
`deploy/huggingface-model/` from the latest local training outputs first, use
`python scripts/push_hf.py --refresh-model-from-training`.

## Local Inference

Run the desktop webcam scripts after you have a `.pt` or `.tflite` artifact available:

```bash
python training/scripts/run_live_inference.py
python training/scripts/run_live_tflite.py
```

Set `WEBCAM_INDEX` if your camera is not on device `0`.
