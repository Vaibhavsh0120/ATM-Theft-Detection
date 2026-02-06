<h1 align="center">🚨 Real-Time ATM Theft Detection with YOLOv8</h1>

<p align="center">
  <img src="https://media.licdn.com/dms/image/sync/v2/D5627AQEQ41fkNOIq8A/articleshare-shrink_800/articleshare-shrink_800/0/1722446223640?e=2147483647&v=beta&t=IpkZR8u2Y5LepFrMrfZo-UjZLkOoFPgaiXqtNU9WUMc" alt="ATM Theft Detection Demo" width="700"/>
</p>

<p align="center">
  <b>Detect suspicious ATM activities in real-time using a custom YOLOv8n model, focusing on individuals with covered faces.</b>
</p>

---

## 📖 Overview

This project aims to enhance ATM security by automatically detecting potential threats, such as individuals concealing their identity. The YOLOv8n model is trained on a custom Roboflow dataset and optimized for fast, real-time inference on webcam feeds.

### 🎭 Class Merging

The original dataset had **13 classes**. To simplify detection, these were merged into two categories using `merge_classes.py`:

- **Face_Covered:** `balaclava`, `concealing glasses`, `cover`, `hand`, `mask`, `medicine mask`, `person-with-mask`, `scarf`, `thief_mask`
- **Face_Uncovered:** `non-concealing glasses`, `normal`, `nothing`, `person-without-mask`

---

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.8+
- NVIDIA GPU with CUDA (recommended for real-time performance)

### 2. Setup

Clone the repository and enter the project directory:

```bash
git clone https://github.com/Vaibhav0120/ATM-Theft-Detection.git
cd ATM-Theft-Detection
```

Create a virtual environment and install dependencies:

```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Roboflow API Key & Dataset

1. [Sign up at Roboflow](https://roboflow.com/)
2. Get your API key from workspace settings.
3. Copy `.env.example` to `.env` and add your API key:
    ```
    ROBOFLOW_API_KEY="YOUR_API_KEY"
    ```
4. Download dataset and model weights:
    ```bash
    python setup.py
    ```
5. [View dataset on Roboflow](https://universe.roboflow.com/vaibhav-7tcrm/atm-theft-detection-f8ezg)

---

## 📂 Project Structure

```
ATM-THEFT-DETECTION/
├── .env                # Your API key (local only)
├── .env.example        # Template for .env
├── .gitignore
├── README.md
├── requirements.txt
├── setup.py            # Downloads dataset & weights
├── train.py            # Model training script
├── live_inference.py   # Real-time webcam inference
├── merge_classes.py    # Merges original classes
│
├── ATM-Theft-Detection-2/ # Dataset (created by setup.py)
│   ├── data.yaml
│   ├── train/
│   ├── valid/
│   └── test/
│
└── weights/
    └── best.pt         # Pre-trained model weights
```

---

## 💻 Usage

### 1. Live Inference (Pre-trained Model)

**Step 1:** Switch to full OpenCV for webcam support:

```bash
pip uninstall opencv-python-headless
pip install opencv-python
```

**Step 2:** Run live inference:

```bash
python live_inference.py
```

A window will show your webcam feed with real-time detections. Press **'q'** to exit.

> **Tip:** Use separate Python environments for `opencv-python` and `opencv-python-headless` if needed.

### 2. Train Your Own Model

**Step 1:** Complete setup steps above.

**Step 2:** Merge classes:

```bash
python merge_classes.py
```

**Step 3:** (Optional) Edit `train.py` for hyperparameters (epochs, image size, device).

**Step 4:** Start training:

```bash
python train.py
```

New weights will be saved in `runs/detect/train/weights/`.

**Step 5:** Replace `weights/best.pt` with your new model.

**Step 6:** Run live inference as above.

---

## 📊 Model Performance

- **mAP@50:** 94.5%
- **Precision:** 94.0%
- **Recall:** 88.0%

---

## 🙌 Contributing

Pull requests and suggestions are welcome! Please open an issue for major changes.
