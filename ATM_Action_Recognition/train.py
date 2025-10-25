#
# FILENAME: train.py
#
import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models
from tqdm import tqdm
import numpy as np
from PIL import Image

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Model & Training Hyperparameters
SEQUENCE_LENGTH = 30  # Number of frames per video clip
IMG_SIZE = 224        # Image size for the CNN
BATCH_SIZE = 8        # Your 6GB RTX 4050 should handle 8-16
EPOCHS = 20           # Number of epochs to train
LEARNING_RATE = 1e-4
NUM_CLASSES = 2       # 'Violence' and 'NonViolence'

# !!! THIS IS THE UPDATED CLASS LIST !!!
# Must match the folder names: 0: NonViolence, 1: Violence
CLASSES_LIST = ["NonViolence", "Violence"] 

# Paths
TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
MODEL_SAVE_PATH = "action_model.pth"

# --- 1. Custom Dataset ---

class VideoFrameDataset(Dataset):
    """
    A custom PyTorch Dataset to load video frames.
    It samples SEQUENCE_LENGTH frames evenly from each video.
    """
    def __init__(self, data_dir, sequence_length, transform, classes_list):
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.transform = transform
        self.classes_list = classes_list
        self.video_files = self._get_video_files()

    def _get_video_files(self):
        video_files = []
        for class_index, class_name in enumerate(self.classes_list):
            class_dir = os.path.join(self.data_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"Warning: Directory not found: {class_dir}")
                continue
            for video_name in os.listdir(class_dir):
                if video_name.endswith('.mp4') or video_name.endswith('.avi'):
                    video_path = os.path.join(class_dir, video_name)
                    video_files.append((video_path, class_index))
        return video_files

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path, label = self.video_files[idx]
        frames = self._load_video_frames(video_path)
        
        # Apply transforms
        frames_transformed = torch.stack([self.transform(frame) for frame in frames])
        
        return frames_transformed, label

    def _load_video_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            # Return a stack of black frames if video fails to open
            black_frame = Image.new('RGB', (IMG_SIZE, IMG_SIZE))
            return [black_frame] * self.sequence_length

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Handle videos with fewer frames than SEQUENCE_LENGTH
        if total_frames < self.sequence_length:
            indices = np.arange(total_frames)
            # Repeat last frame to fill sequence
            indices = np.pad(indices, (0, self.sequence_length - total_frames), 'edge')
        else:
            # Evenly sample frames
            indices = np.linspace(0, total_frames - 1, self.sequence_length, dtype=int)
        
        frames = []
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame)
                frames.append(frame_pil)
            else:
                # Handle case where frame read fails (e.g., use last good frame)
                if frames:
                    frames.append(frames[-1].copy())
                else:
                    # If first frame fails, create a black frame
                    frames.append(Image.new('RGB', (IMG_SIZE, IMG_SIZE)))

        cap.release()
        
        # Final check to ensure correct length
        if len(frames) < self.sequence_length:
            frames.extend([frames[-1].copy()] * (self.sequence_length - len(frames)))
        
        return frames[:self.sequence_length]


# --- 2. Model Definition (CNN + LSTM) ---
# (This class is identical to before, as it's the correct model)
class ActionModel(nn.Module):
    def __init__(self, num_classes, sequence_length, hidden_size=256, num_layers=1):
        super(ActionModel, self).__init__()
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        
        # Feature Extractor (EfficientNet-B0)
        self.cnn = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        
        # Freeze all layers in CNN
        for param in self.cnn.parameters():
            param.requires_grad = False
            
        # Get the input feature size of the classifier
        cnn_out_features = self.cnn.classifier[1].in_features
        
        # Replace the classifier with an Identity layer
        self.cnn.classifier = nn.Identity() # type: ignore
        
        # LSTM Layer
        self.lstm = nn.LSTM(
            input_size=cnn_out_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True # Input shape: (batch_size, seq_len, features)
        )
        
        # Final Classifier
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_len, C, H, W)
        batch_size, seq_len, C, H, W = x.shape
        
        # Reshape for CNN: (batch_size * seq_len, C, H, W)
        x_reshaped = x.view(batch_size * seq_len, C, H, W)
        
        # Get features from CNN
        features = self.cnn(x_reshaped)
        
        # Reshape for LSTM: (batch_size, seq_len, cnn_out_features)
        features_reshaped = features.view(batch_size, seq_len, -1)
        
        # Pass features through LSTM
        lstm_out, (h_n, c_n) = self.lstm(features_reshaped)
        
        # We only need the output from the last time step
        last_time_step_out = lstm_out[:, -1, :]
        
        # Pass to final classifier
        output = self.classifier(last_time_step_out)
        
        return output

# --- 3. Training & Validation ---

def main():
    # Define Transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
    }

    # Create Datasets and DataLoaders
    print("Loading datasets...")
    train_dataset = VideoFrameDataset(TRAIN_DIR, SEQUENCE_LENGTH, data_transforms['train'], CLASSES_LIST)
    val_dataset = VideoFrameDataset(VAL_DIR, SEQUENCE_LENGTH, data_transforms['val'], CLASSES_LIST)
    
    # Use num_workers=2 or 4 if you have a good CPU, 0 if you get errors
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    # Initialize Model, Optimizer, and Loss
    model = ActionModel(num_classes=NUM_CLASSES, sequence_length=SEQUENCE_LENGTH).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    best_val_accuracy = 0.0

    print("--- Starting Training ---")
    
    for epoch in range(EPOCHS):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            _, preds = torch.max(outputs, 1)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset) # type: ignore
        
        print(f"Epoch {epoch+1}/{EPOCHS} [Train] Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                _, preds = torch.max(outputs, 1)
                
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
        
        val_epoch_loss = val_loss / len(val_dataset)
        val_epoch_acc = val_corrects.double() / len(val_dataset) # type: ignore
        
        print(f"Epoch {epoch+1}/{EPOCHS} [Val]   Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}")

        # Save the best model
        if val_epoch_acc > best_val_accuracy:
            best_val_accuracy = val_epoch_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"New best model saved with accuracy: {best_val_accuracy:.4f}")
            
    print("--- Training Complete ---")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"Model saved to '{MODEL_SAVE_PATH}'")

if __name__ == "__main__":
    main()