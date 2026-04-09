# ATM Theft Detection YOLOv8 Model

This folder is intended to be pushed as a standalone Hugging Face model repository.

## Model Summary

The model detects two classes for ATM monitoring:

- `Face_Covered`
- `Face_Uncovered`

## Files

- `weights/best.pt`: published YOLOv8 checkpoint
- `exports/best_full_integer_quant.tflite`: optional quantized export copied from local training

## Local Usage

```python
from ultralytics import YOLO

model = YOLO("weights/best.pt")
results = model("example.jpg")
```

## Source Workflow

The canonical local training flow lives in the root repository under `training/`. After training, push the latest model and Space snapshots with:

```bash
python scripts/push_hf.py
```
