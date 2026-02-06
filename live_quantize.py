import cv2 # type: ignore
from ultralytics import YOLO # type: ignore
import os

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# --- MODEL AND WEBCAM SETUP ---

# Path to your fully quantized TFLite model
# This is the correct model for the Raspberry Pi CPU
model_path = os.path.join(script_dir, 'weights', 'best_saved_model', 'best_full_integer_quant.tflite')

# Check if the model file exists
if not os.path.exists(model_path):
    print(f"Error: Model file not found at '{model_path}'")
    print("Please ensure the quantized model file exists at this path.")
    exit()

# Load the YOLOv8 TFLite model
# The YOLO() class will automatically use the TFLite runtime
model = YOLO(model_path)

# Open a connection to the webcam (0 is usually the default webcam)
cap = cv2.VideoCapture(0)

# Check if the webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    print("Please ensure your webcam is connected and not in use by another application.")
    exit()

print("--- Starting Live Inference with Quantized Model ---")
print("Press 'q' to quit.")

# --- REAL-TIME INFERENCE LOOP ---

while True:
    # Read a frame from the webcam
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        # stream=True is more efficient for video
        results = model(frame, stream=True, verbose=False)

        # Visualize the results on the frame
        for r in results:
            annotated_frame = r.plot()

            # Display the annotated frame
            cv2.imshow("YOLOv8 Live Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        print("Error: Failed to capture frame from webcam.")
        break

# --- CLEANUP ---
cap.release()
cv2.destroyAllWindows()
print("Inference stopped.")