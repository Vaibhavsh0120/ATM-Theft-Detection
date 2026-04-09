import os
from ultralytics import YOLO # type: ignore

# --- Configuration ---
INPUT_MODEL_PATH = 'weights/best.pt'

# We use 'edgetpu' to force full INT8 quantization
EXPORT_FORMAT = 'edgetpu' 

# --- THIS IS THE FIXED PATH ---
# The script saves the model we want for the Pi (CPU) here:
EXPORT_MODEL_NAME = 'weights/best_saved_model/best_full_integer_quant.tflite' 

CALIBRATION_DATA_YAML = 'merged_dataset.yaml'
IMAGE_SIZE = 640
# ---------------------

print(f"Loading YOLOv8 model from '{INPUT_MODEL_PATH}'...")

try:
    model = YOLO(INPUT_MODEL_PATH)
except Exception as e:
    print(f"Error loading base model '{INPUT_MODEL_PATH}': {e}")
    exit()

print(f"--- Starting FULL INT8 Quantization ---")
print(f"Exporting with format='{EXPORT_FORMAT}' to force INT8 inputs/outputs.")
print(f"Using '{CALIBRATION_DATA_YAML}' for calibration...")

try:
    # 2. Export the model
    model.export(
        format=EXPORT_FORMAT, # This forces full INT8
        data=CALIBRATION_DATA_YAML,
        imgsz=IMAGE_SIZE
    )
    
    # This message is just for our info
    print(f"\nUltralytics export process complete.")
    print(f"Our script will now verify the INT8 model at: {EXPORT_MODEL_NAME}")


except Exception as e:
    print(f"\nAn error occurred during export: {e}")
    print("Please ensure '{CALIBRATION_DATA_YAML}' is at the correct path.")
    print("Also ensure your dataset images are accessible.")
    exit() # Exit if export failed

# --- VERIFICATION CHECKS ---
print("\n--- Starting Verification ---")

# 1. Check if the file exists
if not os.path.exists(EXPORT_MODEL_NAME):
    print(f"VERIFICATION FAILED: Model file '{EXPORT_MODEL_NAME}' was not created.")
    print("Please check the export logs above for the correct path.")
else:
    print(f"Verification Check 1/2: File '{EXPORT_MODEL_NAME}' found. [SUCCESS]")
    
    # 2. Check if the model can be loaded by Ultralytics
    try:
        print("Attempting to load the new TFLite model...")
        # We verify the model by loading it
        verified_model = YOLO(EXPORT_MODEL_NAME)
        
        print("Verification Check 2/2: Model loaded successfully. [SUCCESS]")
        print("\nModel Info:")
        verified_model.info()
        print("\n--- QUANTIZATION AND VERIFICATION COMPLETE ---")
        print("✅ You are all set!")
        print(f"You can now copy this file to your Raspberry Pi 5:")
        print(f"'{EXPORT_MODEL_NAME}'")

    except Exception as e:
        print(f"Verification Check 2/2: Failed to load TFLite model. [FAILED]")
        print(f"Error: {e}")
        print("The file may be corrupt or invalid for use with the YOLO class.")