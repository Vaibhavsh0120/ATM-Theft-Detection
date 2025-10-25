#
# FILENAME: setup_dataset.py
#
import os
import zipfile
import shutil
from sklearn.model_selection import train_test_split
from kaggle.api.kaggle_api_extended import KaggleApi
from dotenv import load_dotenv
from tqdm import tqdm

def setup_dataset():
    """
    Downloads, extracts, and splits the Real Life Violence Situations dataset.
    """
    # --- 1. Setup Kaggle API ---
    print("Loading Kaggle credentials...")
    load_dotenv() # Load .env file

    if 'KAGGLE_USERNAME' not in os.environ or 'KAGGLE_KEY' not in os.environ:
        print("Error: KAGGLE_USERNAME and KAGGLE_KEY not found in .env file.")
        print("Please create a .env file in this directory with your Kaggle API credentials.")
        return
    
    api = KaggleApi()
    api.authenticate()

    # --- 2. Download Dataset ---
    # !!! THIS IS THE NEW, WORKING DATASET SLUG !!!
    dataset_slug = 'mohamedmustafa/real-life-violence-situations-dataset'
    download_path = '.'
    zip_file = os.path.join(download_path, 'real-life-violence-situations-dataset.zip')
    extracted_dir = 'real_life_violence_raw' # Temp extraction folder

    if not os.path.exists(zip_file):
        print(f"Downloading dataset '{dataset_slug}'...")
        try:
            api.dataset_download_files(dataset_slug, path=download_path, quiet=False)
            print("Download complete.")
        except Exception as e:
            print(f"\nError downloading from Kaggle: {e}")
            print("This can be a 403 Forbidden error. Please go to the dataset page, log in,")
            print("and accept the rules, then try again:")
            print(f"https://www.kaggle.com/datasets/{dataset_slug}")
            return
    else:
        print("Dataset zip file already exists.")

    # --- 3. Extract Dataset ---
    if os.path.exists(extracted_dir):
        print(f"Removing existing extracted directory: {extracted_dir}")
        shutil.rmtree(extracted_dir)
        
    print(f"Extracting '{zip_file}'...")
    try:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(extracted_dir)
        print("Extraction complete.")
    except zipfile.BadZipFile:
        print(f"Error: Bad zip file. The download may be corrupted. Please delete '{zip_file}' and try again.")
        return
        
    # --- 4. Prepare Train/Val Split ---
    
    # This dataset has a simple structure:
    # real_life_violence_raw/
    #   NonViolence/ (1000 videos)
    #   Violence/ (1000 videos)
    raw_video_dir = extracted_dir
    base_data_dir = 'data'
    
    if os.path.exists(base_data_dir):
        print(f"Removing existing data split directory: {base_data_dir}")
        shutil.rmtree(base_data_dir)

    # !!! THESE ARE THE NEW CLASS NAMES !!!
    classes = ['NonViolence', 'Violence']
    split_ratio = 0.2 # 20% for validation

    print("Creating train/val splits...")
    for class_name in classes:
        # Create destination folders
        train_dir = os.path.join(base_data_dir, 'train', class_name)
        val_dir = os.path.join(base_data_dir, 'val', class_name)
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)

        source_dir = os.path.join(raw_video_dir, class_name)
        if not os.path.exists(source_dir):
            print(f"Warning: Could not find source directory for class {class_name}")
            continue

        videos = [f for f in os.listdir(source_dir) if f.endswith('.mp4') or f.endswith('.avi')]
        
        # Split the list
        train_videos, val_videos = train_test_split(videos, test_size=split_ratio, random_state=42)

        # Copy files
        print(f"Processing class '{class_name}': {len(train_videos)} train, {len(val_videos)} val")
        for video in tqdm(train_videos, desc=f"Copying {class_name} train"):
            shutil.copy(os.path.join(source_dir, video), os.path.join(train_dir, video))
        
        for video in tqdm(val_videos, desc=f"Copying {class_name} val"):
            shutil.copy(os.path.join(source_dir, video), os.path.join(val_dir, video))

    print("\n--- Dataset Setup Complete ---")
    print(f"Data is split into '{base_data_dir}/train' and '{base_data_dir}/val'")
    
    # --- 5. Cleanup ---
    print("Cleaning up raw files...")
    os.remove(zip_file)
    shutil.rmtree(extracted_dir)
    print("Cleanup complete.")

if __name__ == "__main__":
    setup_dataset()