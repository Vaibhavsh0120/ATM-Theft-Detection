from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

os.environ.setdefault("YOLO_CONFIG_DIR", str(SCRIPT_DIR.parents[1] / ".runtime"))

from ultralytics import YOLO  # type: ignore

from repo_paths import MERGED_DATASET_YAML, TFLITE_EXPORT_PATH, ensure_layout, resolve_preferred_model

EXPORT_FORMAT = "edgetpu"
IMAGE_SIZE = 640


def main() -> None:
    ensure_layout()

    if not MERGED_DATASET_YAML.exists():
        print(f"Dataset YAML not found at '{MERGED_DATASET_YAML}'.")
        print("Run 'python training/scripts/remap_labels.py' first.")
        raise SystemExit(1)

    try:
        source_model = resolve_preferred_model()
    except FileNotFoundError as exc:
        print(exc)
        raise SystemExit(1) from exc

    print(f"Loading YOLO model from '{source_model}'...")
    model = YOLO(str(source_model))

    try:
        exported_model = Path(
            model.export(
                format=EXPORT_FORMAT,
                data=str(MERGED_DATASET_YAML),
                imgsz=IMAGE_SIZE,
            )
        ).resolve()
    except Exception as exc:
        print(f"Quantization export failed: {exc}")
        raise SystemExit(1) from exc

    if not exported_model.exists():
        print(f"Expected exported model was not created: '{exported_model}'")
        raise SystemExit(1)

    TFLITE_EXPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(exported_model, TFLITE_EXPORT_PATH)

    print(f"Copied exported model to '{TFLITE_EXPORT_PATH}'.")
    print("Verifying that Ultralytics can load the copied TFLite model...")

    try:
        verified_model = YOLO(str(TFLITE_EXPORT_PATH))
    except Exception as exc:
        print(f"Verification failed while loading '{TFLITE_EXPORT_PATH}': {exc}")
        raise SystemExit(1) from exc

    verified_model.info()
    print("Quantization and verification complete.")


if __name__ == "__main__":
    main()
