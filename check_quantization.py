import tensorflow as tf # type: ignore
import numpy as np
import os

# --- THIS PATH IS NOW CORRECT ---
MODEL_PATH = 'weights/best_saved_model/best_full_integer_quant.tflite'
# ---------------------------------

print(f"--- Checking Model: {MODEL_PATH} ---")
tf.get_logger().setLevel('ERROR') # Hide warnings

if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file not found at '{MODEL_PATH}'")
    exit()

try:
    # Load the TFLite model interpreter
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    
    # Get details about the input tensor
    input_details = interpreter.get_input_details()
    input_dtype = input_details[0]['dtype']
    
    print(f"Model Input Tensor Type:  {input_dtype}")

    # Check if the data type is 8-bit integer
    if input_dtype == np.int8 or input_dtype == np.uint8:
        print("\n[SUCCESS] 👍 Result: The model IS fully quantized (INT8).")
    else:
        print(f"\n[FAILED] 👎 Result: The model is NOT fully quantized ({input_dtype}).")

except Exception as e:
    print(f"\nAn error occurred: {e}")