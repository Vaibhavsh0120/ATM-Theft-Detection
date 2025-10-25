#
# FILENAME: live_inference_action.py
#
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
from collections import deque

# Import the model class from train.py
# We can do this because they are in the same folder.
from train import ActionModel, SEQUENCE_LENGTH, IMG_SIZE, CLASSES_LIST, DEVICE

def run_live_inference():
    """
    Loads the trained action model and runs live inference on a webcam feed.
    """
    
    # --- 1. Load Model ---
    print(f"Loading model to {DEVICE}...")
    MODEL_PATH = "action_model.pth"
    NUM_CLASSES = len(CLASSES_LIST)
    
    # Initialize model definition
    model = ActionModel(num_classes=NUM_CLASSES, sequence_length=SEQUENCE_LENGTH).to(DEVICE)
    
    # Load trained weights
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except FileNotFoundError:
        print(f"Error: Model file not found at '{MODEL_PATH}'")
        print("Please run 'python train.py' first to train the model.")
        return
        
    model.eval() # Set model to evaluation mode
    
    print("Model loaded successfully.")

    # --- 2. Define Transforms (must match validation transforms) ---
    data_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # --- 3. Setup Webcam and Frame Buffer ---
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
        
    # Frame buffer
    frame_buffer = deque(maxlen=SEQUENCE_LENGTH)
    
    # Text display properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_pos = (20, 40)
    font_scale = 1
    font_color = (255, 255, 255) # White
    line_type = 2
    bg_color = (0, 0, 0) # Black
    
    current_action = "Buffering..."
    
    print("--- Starting Live Inference ---")
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
            
        # --- 4. Pre-process Frame ---
        # Convert BGR (OpenCV) to RGB (PIL)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        
        # Apply transforms and add to buffer
        frame_tensor = data_transform(frame_pil)
        frame_buffer.append(frame_tensor)
        
        # --- 5. Run Prediction (if buffer is full) ---
        if len(frame_buffer) == SEQUENCE_LENGTH:
            # Prepare sequence tensor
            # Stack frames: (N, C, H, W) -> (1, N, C, H, W)
            sequence_tensor = torch.stack(list(frame_buffer), dim=0).unsqueeze(0)
            sequence_tensor = sequence_tensor.to(DEVICE)
            
            with torch.no_grad():
                outputs = model(sequence_tensor)
                
                # Get probabilities and prediction
                probabilities = torch.softmax(outputs, dim=1)
                confidence, pred_idx = torch.max(probabilities, 1)
                
                predicted_class = CLASSES_LIST[pred_idx.item()] # type: ignore
                conf_score = confidence.item()
                
                current_action = f"{predicted_class.upper()} ({conf_score*100:.1f}%)"

        # --- 6. Display Results ---
        # Draw a black rectangle background for the text
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 60), bg_color, -1)
        # Put action text on the frame
        cv2.putText(frame, current_action,
                    text_pos,
                    font,
                    font_scale,
                    font_color,
                    line_type)
        
        cv2.imshow("Live Action Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- 7. Cleanup ---
    cap.release()
    cv2.destroyAllWindows()
    print("Inference stopped.")


if __name__ == "__main__":
    run_live_inference()