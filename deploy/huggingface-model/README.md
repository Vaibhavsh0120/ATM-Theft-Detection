---
library_name: ultralytics
pipeline_tag: object-detection
tags:
  - yolov8
  - face-cover-detection
  - computer-vision
  - research
model-index: []
---

# ATM face-cover detection — YOLOv8n

Two-class detector: `0: Face_Covered`, `1: Face_Uncovered`.

[Live website](https://vaibhavsh0120-atm-theft-detection.hf.space) · [Interactive Space](https://huggingface.co/spaces/Vaibhavsh0120/ATM-Theft-Detection) · [Source and training](https://github.com/Vaibhavsh0120/ATM-Theft-Detection) · [Dataset version 4](https://universe.roboflow.com/vaibhav-7tcrm/atm-theft-detection-f8ezg/dataset/4)

## Use

```python
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

checkpoint = hf_hub_download("Vaibhavsh0120/ATM-Theft-Detection", "weights/best.pt")
model = YOLO(checkpoint)
results = model.predict("example.jpg", conf=0.25)
results[0].save(filename="annotated.jpg")
```

Developed with Python 3.11, Ultralytics 8.3.211 and PyTorch 2.5.1. Only load trusted PyTorch checkpoints.

## Training data and interpretation

The Roboflow version 4 export has 21 original classes merged into two classes by the source repository's remapping script. The covered class includes masks, helmets, scarves and other obstruction labels. Original named-person classes are merged into the uncovered class; the model does not perform identity recognition.

This research checkpoint is not a theft classifier. Face coverings do not establish intent, and uncovered faces do not establish safety. Performance may vary with lighting, occlusion, camera angle and dataset bias. No reproducible held-out benchmark report is bundled; historical short-run metrics are not treated as verified production performance.

## Files

- `weights/best.pt`: YOLOv8 checkpoint.
- `exports/`: optional local INT8 TFLite exports, present only when published after conversion.

Check the source repository, dataset terms and Ultralytics licensing before reuse. No additional model license grant is asserted by this card.
