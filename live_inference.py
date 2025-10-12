import cv2
from ultralytics import YOLO # type: ignore
import os

# --- MODEL AND WEBCAM SETUP ---

# Path to your downloaded custom-trained model
model_path = os.path.join('weights', 'best.pt')

# Check if the model file exists
if not os.path.exists(model_path):
    print(f"Error: Model file not found at '{model_path}'")
    print("Please run 'python setup.py' to download the model weights.")
    exit()

# Load the YOLOv8 model
model = YOLO(model_path)

# Open a connection to the webcam (0 is usually the default webcam)
cap = cv2.VideoCapture(0)

# Check if the webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    print("Please ensure your webcam is connected and not in use by another application.")
    exit()

print("--- Starting Live Inference ---")
print("Press 'q' to quit.")

# --- REAL-TIME INFERENCE LOOP ---

while True:
    # Read a frame from the webcam
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame, disable verbose output for a cleaner console
        results = model(frame, verbose=False)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

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