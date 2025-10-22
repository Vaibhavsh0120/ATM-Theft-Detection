import os
import torch
from ultralytics import YOLO # type: ignore

def main():
    # Get the directory where the script is located
    script_dir = os.path.dirname(__file__)

    # --- Training Configuration ---
    # Path to the dataset YAML file (relative to this script)
    data_yaml = os.path.join(script_dir, 'merged_dataset.yaml')

    # Pre-trained model to start from (relative to this script)
    pretrained_model = os.path.join(script_dir, 'yolov8n.pt')

    # Training parameters
    epochs = 5
    img_size = 640

    # Automatically select device: prioritizes GPU if available, otherwise uses CPU
    device = 0 if torch.cuda.is_available() else 'cpu'

    # --- Verification ---
    if not os.path.exists(data_yaml):
        print(f"Error: Dataset YAML not found at '{data_yaml}'")
        print("Please ensure 'merged_dataset.yaml' is in the same directory as train.py")
        print("Run 'python merge_classes.py' if needed.")
        return

    if not os.path.exists(pretrained_model):
        print(f"Error: Base model '{os.path.basename(pretrained_model)}' not found at '{pretrained_model}'")
        print("Please ensure 'yolov8n.pt' is in the same directory as train.py.")
        # You could add logic here to download yolov8n.pt if missing
        return

    print("--- Starting Model Training ---")
    print(f"Running from directory: {os.getcwd()}") # Show current working directory
    print(f"Using device: {str(device).upper()}")
    print(f"Dataset YAML: {os.path.relpath(data_yaml)}")
    print(f"Base Model: {os.path.relpath(pretrained_model)}")
    print(f"Epochs: {epochs}")
    print(f"Image Size: {img_size}")
    print("-----------------------------")

    # Load the model
    model = YOLO(pretrained_model)

    # Train the model
    # Note: Ultralytics handles the 'runs' directory relative to the execution location
    try:
        model.train(
            data=data_yaml, # Use the absolute or correct relative path
            epochs=epochs,
            imgsz=img_size,
            device=device,
            exist_ok=True, # Allows re-running training in the same directory
            project=script_dir, # Optional: force runs dir creation inside script dir
            name='train' # Optional: sets the subfolder name in 'runs/detect/'
        )
        print("\n✅ --- Training Complete ---")
        print(f"Best model weights saved in '{os.path.join(script_dir, 'runs/detect/train/weights/')}' directory.")
    except Exception as e:
        print(f"\n❌ An error occurred during training: {e}")

if __name__ == '__main__':
    # Change directory to the script's location before running main
    # This makes relative paths (like './ATM-Theft-Detection-2' in YAML) work reliably
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()