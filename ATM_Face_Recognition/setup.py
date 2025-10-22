import os
import requests
from roboflow import Roboflow # type: ignore
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables from .env file (expected in the same directory)
load_dotenv()

def download_file(url, filename):
    """Downloads a file from a URL with a progress bar."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte

        print(f"Downloading {filename}...")
        # Save relative to the current script directory
        save_path = os.path.join(os.path.dirname(__file__), filename)
        with open(save_path, 'wb') as file, tqdm(
            desc=filename,
            total=total_size_in_bytes,
            unit='iB',
            unit_scale=True,
        ) as bar:
            for data in response.iter_content(block_size):
                bar.update(len(data))
                file.write(data)
        print(f"Downloaded {filename} successfully.")
        return save_path # Return the actual path where it was saved
    except requests.exceptions.RequestException as e:
        print(f"Failed to download {filename}. Error: {e}")
        exit()

def main():
    # Get the directory where the script is located
    script_dir = os.path.dirname(__file__)

    # --- Download Dataset from Roboflow ---
    print("--- Downloading Dataset ---")
    api_key = os.getenv("ROBOFLOW_API_KEY")

    if not api_key or api_key == "YOUR_API_KEY":
        print("Error: ROBOFLOW_API_KEY not found in .env file (expected in the same directory as setup.py).")
        api_key = input("Please enter your Roboflow Private API Key to continue: ")

    # Define dataset path relative to the script directory
    dataset_dir = os.path.join(script_dir, "ATM-Theft-Detection-2")

    try:
        rf = Roboflow(api_key=api_key)
        project = rf.workspace("vaibhav-7tcrm").project("atm-theft-detection-f8ezg")
        # Check if dataset directory already exists
        if not os.path.exists(dataset_dir):
            print(f"Downloading dataset to: {dataset_dir}")
            # Set location explicitly for download
            version_obj = project.version(2)
            version_obj.download("yolov8", location=script_dir) # Download to script's dir
            print("Dataset downloaded successfully.")
        else:
            print(f"Dataset '{os.path.basename(dataset_dir)}' already exists in {script_dir}. Skipping download.")
    except Exception as e:
        print(f"An error occurred while downloading the dataset: {e}")
        print("Please ensure your API key is correct and you have access to the project.")
        return

    # --- Download Model Weights ---
    print("\n--- Downloading Model Weights ---")

    # Define weights path relative to the script directory
    weights_dir = os.path.join(script_dir, "weights")
    weights_path = os.path.join(weights_dir, "best.pt")

    # Check if the weights file already exists
    if os.path.exists(weights_path):
        print(f"Model weights '{os.path.relpath(weights_path)}' already exist. Skipping download.")
    else:
        # Direct download link from your GitHub Release
        weights_url = "https://github.com/Vaibhav0120/ATM-Theft-Detection/releases/download/weights/best.pt"

        os.makedirs(weights_dir, exist_ok=True)
        # Pass the full desired path to download_file
        download_file(weights_url, weights_path)

    print("\n✅ Setup complete! You are now ready to run training or inference from the ATM_Face_Recognition directory.")

if __name__ == "__main__":
    main()