import cv2
from ultralytics import YOLO

# --- MODEL AND WEBCAM SETUP ---

# Load your custom-trained YOLOv8 model
# Make sure the path is correct
model_path = 'runs/detect/train/weights/best.pt'
model = YOLO(model_path)

# Open a connection to the webcam (0 is usually the default webcam)
cap = cv2.VideoCapture(0)

# Check if the webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# --- REAL-TIME INFERENCE LOOP ---

while True:
    # Read a frame from the webcam
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        # The model will automatically use your RTX 4050 GPU if available
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame in a window
        cv2.imshow("YOLOv8 Live Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if reading the frame fails
        print("Error: Failed to capture frame.")
        break

# --- CLEANUP ---

# Release the webcam and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()

print("Inference stopped.")