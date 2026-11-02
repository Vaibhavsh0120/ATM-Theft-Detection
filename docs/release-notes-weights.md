# Pretrained YOLOv8n face-cover detector

Detects `Face_Covered` (class 0) and `Face_Uncovered` (class 1). This checkpoint is intended for research and demonstration. Face covering alone does not establish theft or intent.

- [Open the live website](https://vaibhavsh0120-atm-theft-detection.hf.space)
- [Hugging Face Space](https://huggingface.co/spaces/Vaibhavsh0120/ATM-Theft-Detection)
- [Hugging Face model card](https://huggingface.co/Vaibhavsh0120/ATM-Theft-Detection)
- [Setup and training instructions](https://github.com/Vaibhavsh0120/ATM-Theft-Detection#run-the-website-locally)

## Assets

- `best.pt`: original published YOLOv8n checkpoint, preserved without retraining.
- `SHA256SUMS`: integrity checksum for `best.pt`.

SHA-256: `f300d94325772555e8e22ffe6f88755fe9c696d75e4d4494dbbe2cfc0debf937`

Verify with `sha256sum -c SHA256SUMS` on Linux, or `Get-FileHash best.pt -Algorithm SHA256` in PowerShell.

Load with Python 3.11, `ultralytics==8.3.211` and `torch==2.5.1`:

```python
from ultralytics import YOLO
model = YOLO('best.pt')
results = model.predict('example.jpg', conf=0.25)
results[0].save(filename='annotated.jpg')
```

The original release described a 100-epoch training run; a reproducible training/evaluation report is not bundled. No new training or accuracy claim is made by this release update. Download checkpoints only from trusted sources and review the dataset and Ultralytics license terms before reuse.
