import os
import torch
from ultralytics import YOLO # type: ignore

def main():
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir) # IMPORTANT: Change working directory to script's dir

    # --- Training Configuration ---
    # Path to the dataset YAML file (now relative to the script dir)
    data_yaml = 'merged_dataset.yaml'

    # Pre-trained model to start from (now relative to the script dir)
    pretrained_model = 'yolov8n.pt'

    # Training parameters
    epochs = 15
    img_size = 640

    # Automatically select device: prioritizes GPU if available, otherwise uses CPU
    device = 0 if torch.cuda.is_available() else 'cpu'

    # --- Verification ---
    if not os.path.exists(data_yaml):
        print(f"Error: Dataset YAML not found at '{os.path.abspath(data_yaml)}'")
        print("Please ensure 'merged_dataset.yaml' is in the same directory as train.py.")
        print("Run 'python setup.py' and 'python merge_classes.py' first.")
        return

    if not os.path.exists(pretrained_model):
        print(f"Error: Base model '{pretrained_model}' not found at '{os.path.abspath(pretrained_model)}'")
        print("Please ensure 'yolov8n.pt' is in the same directory as train.py.")
        # You could add logic here to download yolov8n.pt if missing
        return

    print("--- Starting Model Training ---")
    print(f"Working Directory: {os.getcwd()}") # Confirm working directory
    print(f"Using device: {str(device).upper()}")
    print(f"Dataset YAML: {data_yaml}")
    print(f"Base Model: {pretrained_model}")
    print(f"Epochs: {epochs}")
    print(f"Image Size: {img_size}")
    print("-----------------------------")

    # Load the model
    model = YOLO(pretrained_model)

    # Train the model
    # Note: Ultralytics creates 'runs' relative to the current working directory
    try:
        model.train(
            data=data_yaml, # Path is now relative to script_dir due to os.chdir()
            epochs=epochs,
            imgsz=img_size,
            device=device,
            exist_ok=True, # Allows re-running training in the same directory
            project='.', # Create runs folder in the current directory (ATM_Face_Recognition)
            name='runs/detect/train' # Define the experiment path structure
        )
        print("\n✅ --- Training Complete ---")
        print("Best model weights saved in './runs/detect/train/weights/' directory.")
    except Exception as e:
        print(f"\n❌ An error occurred during training: {e}")

if __name__ == '__main__':
    main()