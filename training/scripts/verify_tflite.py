from __future__ import annotations

import numpy as np
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from repo_paths import TFLITE_EXPORT_PATH


def main() -> None:
    try:
        import tensorflow as tf  # type: ignore
    except ModuleNotFoundError as exc:
        print("TensorFlow is required for verify_tflite.py.")
        raise SystemExit(1) from exc

    print(f"--- Checking model: {TFLITE_EXPORT_PATH} ---")
    tf.get_logger().setLevel("ERROR")

    if not TFLITE_EXPORT_PATH.exists():
        print(f"Model file not found at '{TFLITE_EXPORT_PATH}'.")
        raise SystemExit(1)

    try:
        interpreter = tf.lite.Interpreter(model_path=str(TFLITE_EXPORT_PATH))
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
    except Exception as exc:
        print(f"Failed to inspect the TFLite model: {exc}")
        raise SystemExit(1) from exc

    details = input_details + output_details
    for tensor in details:
        print(f"Tensor {tensor['name']}: {tensor['dtype']}")
    if details and all(tensor["dtype"] in (np.int8, np.uint8) and tensor["quantization"][0] > 0 for tensor in details):
        print("Success: all input and output tensors use quantized integer I/O.")
        return

    print("Failed: expected quantized int8/uint8 input and output tensors with a nonzero scale.")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
