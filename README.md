<h1 align="center">Real-Time ATM Theft Detection with YOLOv8</h1>

<p align="center">
  <img src="https://media.licdn.com/dms/image/sync/v2/D5627AQEQ41fkNOIq8A/articleshare-shrink_800/articleshare-shrink_800/0/1722446223640?e=2147483647&amp;v=beta&amp;t=IpkZR8u2Y5LepFrMrfZo-UjZLkOoFPgaiXqtNU9WUMc" alt="ATM Theft Detection Demo" width="700"/>
</p>

<p align="center">
  <b>Detect suspicious ATM activity in real time using a custom YOLOv8n model trained to separate covered and uncovered faces.</b>
</p>

---

## Overview

This project focuses on ATM security monitoring with a custom YOLOv8 pipeline built around a Roboflow dataset and a lightweight YOLOv8n detector. The repository is structured as a small monorepo so you can:

- download and prepare the dataset locally
- train and resume YOLOv8 runs
- run webcam inference on desktop
- package the latest checkpoint for a Hugging Face Model repo
- push both the Hugging Face Model repo and Hugging Face Space repo with one command

### Class Merging

The current Roboflow export contains **21 source classes**. They are remapped into two final detection classes by [`training/scripts/remap_labels.py`](training/scripts/remap_labels.py):

- **Face_Covered:** `Helmet`, `balaclava`, `concealing glasses`, `cover`, `hand`, `mask`, `medicine mask`, `person-with-mask`, `scarf`, `thief_mask`
- **Face_Uncovered:** `Cuong`, `Hung`, `Lau-Ka-Fai`, `Trung`, `Tuan`, `Vu`, `face`, `non-concealing glasses`, `normal`, `nothing`, `person-without-mask`

---

## Quick Start

### 1. Prerequisites

- Python 3.11
- NVIDIA GPU with CUDA recommended for training speed
- Git LFS recommended if you plan to publish model weights

### 2. Clone And Install

```bash
git clone https://github.com/Vaibhavsh0120/ATM-Theft-Detection.git
cd ATM-Theft-Detection
python -m venv .venv
```

Activate the environment and install the training dependencies:

```bash
# Windows
.\.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -r training/requirements.txt
```

### 3. Roboflow API Key And Dataset

1. Create a root `.env` file from `.env.example`.
2. Add your Roboflow private API key:

   ```text
   ROBOFLOW_API_KEY="YOUR_API_KEY"
   ```

3. Download the dataset and published model assets:

   ```bash
   python training/scripts/bootstrap_assets.py
   ```

4. Remap the source labels into the final two-class dataset:

   ```bash
   python training/scripts/remap_labels.py
   ```

5. Start training:

   ```bash
   python training/scripts/train_yolo.py --epochs 5
   ```

6. Resume an interrupted run when needed:

   ```bash
   python training/scripts/train_yolo.py --resume
   ```

Dataset source: [Roboflow Universe - ATM Theft Detection](https://universe.roboflow.com/vaibhav-7tcrm/atm-theft-detection-f8ezg)

---

## Project Structure

```text
ATM-THEFT-DETECTION/
├── .env.example
├── README.md
├── deploy/
│   ├── huggingface-model/      # Standalone Hugging Face Model repo contents
│   └── huggingface-space/      # Standalone Hugging Face Space repo contents
├── docs/
│   └── publishing.md
├── scripts/
│   └── push_hf.py              # Refresh deploy assets and push to Hugging Face
└── training/
    ├── configs/
    ├── data/                   # Downloaded Roboflow dataset
    ├── models/
    │   ├── exports/
    │   └── pretrained/
    ├── outputs/                # YOLO training runs
    ├── requirements.txt
    └── scripts/
        ├── bootstrap_assets.py
        ├── remap_labels.py
        ├── train_yolo.py
        ├── export_tflite.py
        ├── run_live_inference.py
        └── run_live_tflite.py
```

---

## Usage

### 1. Train A Fresh Model

```bash
python training/scripts/train_yolo.py --epochs 80
```

Fresh runs are written to incrementing folders such as:

```text
training/outputs/runs/detect/train/
training/outputs/runs/detect/train2/
training/outputs/runs/detect/train3/
```

### 2. Resume From The Last Checkpoint

```bash
python training/scripts/train_yolo.py --resume
```

You can also resume a specific checkpoint:

```bash
python training/scripts/train_yolo.py --resume-from "training/outputs/runs/detect/train2/weights/last.pt"
```

### 3. Live Webcam Inference

Run desktop webcam inference with the latest trained model or the published fallback checkpoint:

```bash
python training/scripts/run_live_inference.py
```

If you want TFLite export and TFLite webcam inference:

```bash
python training/scripts/export_tflite.py
python training/scripts/verify_tflite.py
python training/scripts/run_live_tflite.py
```

### 4. Publish To Hugging Face

Add these optional values to your root `.env`:

```text
HF_MODEL_REPO_ID="your-username/your-model-repo"
HF_SPACE_REPO_ID="your-username/your-space-repo"
```

Log in once:

```bash
huggingface-cli login
```

Then push both standalone deploy targets in one command:

```bash
python scripts/push_hf.py
```

If you want the Space repo to include `weights/best.pt` directly instead of downloading from the model repo:

```bash
python scripts/push_hf.py --bundle-space-model
```

---

## Validation Snapshot

Recent local 5-epoch smoke-run results:

- **Face_Covered:** Precision `0.894`, Recall `0.827`, mAP50 `0.898`, mAP50-95 `0.558`
- **Face_Uncovered:** Precision `0.823`, Recall `0.800`, mAP50 `0.855`, mAP50-95 `0.599`

These numbers come from a short verification run, not a final long-training benchmark.

---

## Notes

- [`training/scripts/remap_labels.py`](training/scripts/remap_labels.py) mutates label files in place once per fresh dataset download.
- If you need to remap again, rerun `python training/scripts/bootstrap_assets.py` first to restore a clean Roboflow export.
- The Hugging Face Space can either bundle weights locally or load them from the Hugging Face Model repo through `HF_MODEL_REPO_ID`.

---

## Contributing

Pull requests and suggestions are welcome. Open an issue first for major changes so the scope is clear before implementation.
