# Training workspace

Use Python 3.11 and run these commands from the repository root. Follow the [root README](../README.md) to create and activate `.venv`.

```bash
python -m pip install -r training/requirements.txt
```

CPU PyTorch is the default. For GPU training, use the CUDA build matching your hardware from the official PyTorch installation selector. Set `ROBOFLOW_API_KEY` in a root `.env` copied from `.env.example`.

## Prepare and train

```bash
python training/scripts/bootstrap_assets.py
python training/scripts/remap_labels.py
python training/scripts/train_yolo.py --epochs 80
python training/scripts/train_yolo.py --resume
python training/scripts/train_yolo.py --resume-from training/outputs/runs/detect/train2/weights/last.pt
```

Bootstrap uses the Roboflow export REST API, downloads to staging and validates before replacing existing data. It preserves the previous dataset if downloading or extraction fails. The API key is never printed in download errors.

Remapping supports the original version 4 class order. It validates all bounding boxes before writing, saves original labels and `data.yaml` under `.original-labels/`, updates metadata to the two target classes, and creates `.labels_remapped` only on completion. If interrupted after writes begin, restore a fresh dataset using bootstrap before retrying. Never delete the marker and remap already converted labels: original class 0 and class 1 have different meanings.

Fresh runs are auto-incremented under `training/outputs/runs/detect/train*/`. Resume uses the most recently modified `last.pt`; an explicit checkpoint is safer when multiple experiments exist. Resume preserves the saved epoch target. `YOLO_BASE_MODEL`, `YOLO_EPOCHS`, `YOLO_IMAGE_SIZE` and `YOLO_RUN_NAME` configure fresh runs through environment variables. The default base is `yolov8n.pt`, downloaded to `training/models/pretrained/`.

## Desktop inference

```bash
python training/scripts/run_live_inference.py
```

Uses the newest local `best.pt`, then the bundled published checkpoint. Set `WEBCAM_INDEX` if the camera is not device 0. Press `q` to quit. This requires a desktop session and OpenCV with GUI support.

## Optional INT8 TFLite export

Use a separate Python 3.11 environment on Linux/WSL for TensorFlow conversion dependencies; this is heavier than running the website. Install the training requirements and TensorFlow there. Ultralytics may install additional conversion dependencies during export.

```bash
python -m pip install 'tensorflow>=2.16,<2.20'
python training/scripts/export_tflite.py
python training/scripts/verify_tflite.py
python training/scripts/run_live_tflite.py
```

Export needs the prepared dataset for calibration. The exporter selects `format=tflite, int8=True`, then copies the **full integer I/O** sibling to `training/models/exports/best_full_integer_quant.tflite`. It does not invoke the Coral Edge TPU compiler. Verification checks actual input and output tensor dtypes; successful construction of a YOLO wrapper alone does not validate a TFLite file. Conversion and hardware inference must be tested in the target environment; CI does not perform conversion.

## Publish

```bash
python scripts/push_hf.py --refresh-model-from-training --bundle-space-model
```

See [publishing](../docs/publishing.md) for authentication and safe Hub uploads. Datasets, base models, generated exports, and training outputs stay ignored by Git. Publishable checkpoints live in `deploy/`.
