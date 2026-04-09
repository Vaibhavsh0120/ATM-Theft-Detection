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

from repo_paths import resolve_preferred_model


def main() -> None:
    webcam_index = int(os.getenv("WEBCAM_INDEX", "0"))

    try:
        model_path = resolve_preferred_model()
    except FileNotFoundError as exc:
        print(exc)
        raise SystemExit(1) from exc

    print(f"Loading model from '{model_path}'...")
    model = YOLO(str(model_path))

    cap = cv2.VideoCapture(webcam_index)
    if not cap.isOpened():
        print(f"Could not open webcam index {webcam_index}.")
        raise SystemExit(1)

    print("Starting live inference. Press 'q' to quit.")

    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to capture frame from webcam.")
            break

        results = model(frame, verbose=False)
        annotated_frame = results[0].plot()
        cv2.imshow("YOLOv8 Live Inference", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Inference stopped.")


if __name__ == "__main__":
    main()
