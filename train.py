import os
import torch
from ultralytics import YOLO # type: ignore

def main():
    # --- Training Configuration ---
    # Path to the dataset YAML file after merging classes
    data_yaml = 'merged_dataset.yaml'
    
    # Pre-trained model to start from
    pretrained_model = 'yolov8n.pt'
    
    # Training parameters
    epochs = 5
    img_size = 640
    
    # Automatically select device: prioritizes GPU if available, otherwise uses CPU
    device = 0 if torch.cuda.is_available() else 'cpu'

    # --- Verification ---
    if not os.path.exists(data_yaml):
        print(f"Error: Dataset YAML not found at '{data_yaml}'")
        print("Please run 'python setup.py' first to download the dataset.")
        return
        
    print("--- Starting Model Training ---")
    print(f"Using device: {str(device).upper()}")
    print(f"Dataset: {data_yaml}")
    print(f"Model: {pretrained_model}")
    print(f"Epochs: {epochs}")
    print(f"Image Size: {img_size}")
    print("-----------------------------")

    # Load the model
    model = YOLO(pretrained_model)

    # Train the model
    try:
        model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=img_size,
            device=device,
            exist_ok=True # Allows re-running training in the same directory
        )
        print("\n✅ --- Training Complete ---")
        print("Best model weights saved in the 'runs/detect/train/weights/' directory.")
    except Exception as e:
        print(f"\n❌ An error occurred during training: {e}")

if __name__ == '__main__':
    main()