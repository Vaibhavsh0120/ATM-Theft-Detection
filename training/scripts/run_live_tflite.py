from __future__ import annotations

import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

os.environ.setdefault("YOLO_CONFIG_DIR", str(SCRIPT_DIR.parents[1] / ".runtime"))

import cv2  # type: ignore
from ultralytics import YOLO  # type: ignore

from repo_paths import TFLITE_EXPORT_PATH


def main() -> None:
    webcam_index = int(os.getenv("WEBCAM_INDEX", "0"))

    if not TFLITE_EXPORT_PATH.exists():
        print(f"TFLite model not found at '{TFLITE_EXPORT_PATH}'.")
        print("Run 'python training/scripts/export_tflite.py' first.")
        raise SystemExit(1)

    print(f"Loading quantized model from '{TFLITE_EXPORT_PATH}'...")
    model = YOLO(str(TFLITE_EXPORT_PATH))

    cap = cv2.VideoCapture(webcam_index)
    if not cap.isOpened():
        print(f"Could not open webcam index {webcam_index}.")
        raise SystemExit(1)

    print("Starting live inference with the quantized model. Press 'q' to quit.")

    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to capture frame from webcam.")
            break

        for result in model(frame, stream=True, verbose=False):
            cv2.imshow("YOLOv8 Live Inference", result.plot())

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Inference stopped.")


if __name__ == "__main__":
    main()
