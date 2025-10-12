import os
import requests
from roboflow import Roboflow
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def download_file(url, filename):
    """Downloads a file from a URL with a progress bar."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte

        print(f"Downloading {filename}...")
        with open(filename, 'wb') as file, tqdm(
            desc=filename,
            total=total_size_in_bytes,
            unit='iB',
            unit_scale=True,
        ) as bar:
            for data in response.iter_content(block_size):
                bar.update(len(data))
                file.write(data)
        print(f"Downloaded {filename} successfully.")
    except requests.exceptions.RequestException as e:
        print(f"Failed to download {filename}. Error: {e}")
        exit()

def main():
    # --- Download Dataset from Roboflow ---
    print("--- Downloading Dataset ---")
    api_key = os.getenv("ROBOFLOW_API_KEY")

    if not api_key or api_key == "YOUR_API_KEY":
        print("Error: ROBOFLOW_API_KEY not found in .env file.")
        api_key = input("Please enter your Roboflow Private API Key to continue: ")

    try:
        rf = Roboflow(api_key=api_key)
        project = rf.workspace("vaibhav-7tcrm").project("atm-theft-detection-f8ezg")
        # Check if dataset directory already exists
        if not os.path.exists("ATM-Theft-Detection-2"):
            project.version(2).download("yolov8")
            print("Dataset downloaded successfully.")
        else:
            print("Dataset 'ATM-Theft-Detection-2' already exists. Skipping download.")
    except Exception as e:
        print(f"An error occurred while downloading the dataset: {e}")
        print("Please ensure your API key is correct and you have access to the project.")
        return

    # --- Download Model Weights ---
    print("\n--- Downloading Model Weights ---")
    
    weights_dir = "weights"
    weights_path = os.path.join(weights_dir, "best.pt")
    
    # Check if the weights file already exists
    if os.path.exists(weights_path):
        print("Model weights 'weights/best.pt' already exist. Skipping download.")
    else:
        # Direct download link from your GitHub Release
        weights_url = "https://github.com/Vaibhav0120/ATM-Theft-Detection/releases/download/weights/best.pt"
            
        os.makedirs(weights_dir, exist_ok=True)
        download_file(weights_url, weights_path)
    
    print("\n✅ Setup complete! You are now ready to run training or inference.")

if __name__ == "__main__":
    main()