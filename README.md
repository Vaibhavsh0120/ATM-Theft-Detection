# Real-Time ATM Theft Detection with YOLOv8

This project uses a custom-trained YOLOv8n model to detect suspicious activities at ATMs in real-time, focusing on identifying individuals with covered faces.

![ATM Theft Detection Inference](https://i.imgur.com/your-image-code.jpg)
*Replace this link with a URL to a screenshot of your `live_inference.py` running!*

---

## About The Project

The primary goal is to enhance ATM security by automatically identifying potential threats, such as individuals concealing their identity. The model was trained on a custom dataset from Roboflow and is optimized for fast, real-time inference on a live webcam feed.

### Class Merging

The original dataset contained **13 distinct classes**. To simplify the problem into a binary classification task ("covered" vs "uncovered"), these classes were merged into two final categories using the `merge_classes.py` script:

* **`Face_Covered`**: Includes `balaclava`, `concealing glasses`, `cover`, `hand`, `mask`, `medicine mask`, `person-with-mask`, `scarf`, and `thief_mask`.
* **`Face_Uncovered`**: Includes `non-concealing glasses`, `normal`, `nothing`, and `person-without-mask`.

---

## Getting Started

Follow these steps to set up and run the project on your local machine.

### 1. Prerequisites

* Python 3.8+
* An NVIDIA GPU with CUDA support is highly recommended for real-time performance.

### 2. Setup

First, clone the repository and navigate into the project directory:
```bash
git clone [https://github.com/your-username/atm-theft-detection.git](https://github.com/your-username/atm-theft-detection.git)
cd atm-theft-detection
````

Next, create a Python virtual environment and install the required packages:

```bash
# Create a virtual environment
python -m venv venv

# Activate it (on Windows)
.\venv\Scripts\activate

# Activate it (on macOS/Linux)
source venv/bin/activate

# Install dependencies from the requirements file
pip install -r requirements.txt
```

### 3\. Environment and Downloads

Before running, you need your Roboflow API key and the necessary project files.

1.  **Create a `.env` file** by copying the `.env.example` template.
2.  **Add your API key** to the `.env` file. You can find this in your Roboflow account settings.
3.  **Run the setup script** to download the dataset from Roboflow and the pre-trained `best.pt` model weights.
    ```bash
    python setup.py
    ```
-----

<br>

## Usage

This repository is configured for two main use-cases: running live inference with the provided pre-trained model, or training your own model from scratch using the same dataset.

### 1\. Using the Pre-trained Model (Live Inference)

The `requirements.txt` file installs `opencv-python-headless` by default to ensure compatibility with cloud services like Roboflow. To run live inference on your local machine, you must switch to the full version of OpenCV.

**Step 1: Swap OpenCV Version**

First, uninstall the headless version and install the full desktop version:

```bash
pip uninstall opencv-python-headless
pip install opencv-python
```

**Step 2: Run the Inference Script**

Once the correct OpenCV version is installed, run the `live_inference.py` script:

```bash
python live_inference.py
```

A window will open showing your webcam feed with real-time bounding boxes. Press **'q'** to quit.

> **Note:** If you later need to use a script that requires the headless version, you can always switch back by running `pip install -r requirements.txt`.

### 2\. Training a New Model

If you want to train your own model using this dataset, follow these steps.

**Step 1: (Optional) Modify Training Parameters**

You can open the `train.py` script to adjust hyperparameters like the number of `epochs`, `img_size` (image size), or `device` (e.g., changing from `0` for GPU to `'cpu'`).

**Step 2: Run the Training Script**

The `train.py` script is pre-configured to start training from the official `yolov8n.pt` weights and use the downloaded dataset.

```bash
python train.py
```

Training progress will be displayed in the console. When finished, the new weights (including `best.pt`) will be saved in a new directory inside the `runs/detect/` folder.

## Performance

The trained model achieved the following metrics on the validation set:

  - **mAP@50:** 94.5%
  - **Precision:** 94.0%
  - **Recall:** 88.0%

