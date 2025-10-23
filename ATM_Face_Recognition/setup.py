import os
import requests
from roboflow import Roboflow # type: ignore
from tqdm import tqdm
from dotenv import load_dotenv
import shutil
import time # Import time for a small delay

# Load environment variables from .env file (expected in the same directory)
load_dotenv()

def download_file(url, filename):
    """Downloads a file from a URL with a progress bar."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status() # Raise an exception for bad status codes
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024 # 1 Kibibyte

        if total_size_in_bytes == 0:
             print(f"Warning: Content-Length is 0 for {os.path.basename(filename)}. Download might be empty or skipped.")

        print(f"Downloading {os.path.basename(filename)}...")
        # Save relative to the script's directory
        with open(filename, 'wb') as file, tqdm(
            desc=os.path.basename(filename),
            total=total_size_in_bytes,
            unit='iB',
            unit_scale=True,
        ) as bar:
            for data in response.iter_content(block_size):
                bar.update(len(data))
                file.write(data)
        # Add check for file size after download
        downloaded_size = os.path.getsize(filename)
        if total_size_in_bytes != 0 and downloaded_size < total_size_in_bytes * 0.9: # Check if at least 90% downloaded
             print(f"Warning: Downloaded file size ({downloaded_size} bytes) is much smaller than expected ({total_size_in_bytes} bytes).")
        elif downloaded_size == 0 and total_size_in_bytes == 0:
             print(f"Warning: Downloaded file size is 0 for {os.path.basename(filename)}.")
        else:
             print(f"Downloaded {os.path.basename(filename)} successfully ({downloaded_size} bytes).")
        return filename # Return the actual path where it was saved
    except requests.exceptions.RequestException as e:
        print(f"Failed to download {os.path.basename(filename)}. Error: {e}")
        exit()
    except Exception as e:
        print(f"An unexpected error occurred during file download: {e}")
        exit()


def main():
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir) # Change current directory to script's directory

    # --- Download Dataset from Roboflow ---
    print("--- Downloading Dataset ---")
    api_key = os.getenv("ROBOFLOW_API_KEY")

    if not api_key or api_key == "YOUR_API_KEY":
        print("Error: ROBOFLOW_API_KEY not found in .env file (expected in the same directory as setup.py).")
        api_key = input("Please enter your Roboflow Private API Key to continue: ")

    # Define dataset path relative to the script directory
    expected_dataset_dir_name = "ATM-Theft-Detection-2"
    dataset_dir = os.path.join(script_dir, expected_dataset_dir_name)
    download_success_flag = False # Flag to track download status

    try:
        print("Initializing Roboflow...")
        rf = Roboflow(api_key=api_key)
        print("Accessing workspace 'vaibhav-7tcrm'...")
        project = rf.workspace("vaibhav-7tcrm").project("atm-theft-detection-f8ezg")
        print("Accessing project version 2...")
        version_obj = project.version(2)

        # Check if dataset directory already exists
        if os.path.exists(dataset_dir):
            print(f"Dataset directory '{expected_dataset_dir_name}' already exists. Removing it to force fresh download.")
            try:
                shutil.rmtree(dataset_dir)
                print(f"Removed existing directory: {dataset_dir}")
                time.sleep(1) # Small pause after removing
            except OSError as e:
                print(f"Error removing directory {dataset_dir}: {e}. Please remove manually and retry.")
                return

        print(f"\nAttempting to download dataset via Roboflow API to: {script_dir}")
        print("This might take several minutes depending on dataset size and connection speed...")

        # Initiate download
        # *** MODIFIED LINE: Removed location=script_dir ***
        version_obj.download("yolov8") # Download to current working dir (which is script_dir)
        print("Roboflow download command finished.") # Added message

        # --- MORE ROBUST VERIFICATION ---
        print("Verifying download...")
        time.sleep(2) # Give filesystem a moment

        actual_downloaded_folder = None
        possible_folder_name = expected_dataset_dir_name

        # Check if the exact expected folder name exists
        if os.path.exists(dataset_dir):
            print(f"Found expected directory: '{expected_dataset_dir_name}'")
            actual_downloaded_folder = dataset_dir
        else:
            # Check for variations Roboflow might create (less likely with explicit location, but good to check)
            print(f"Expected directory '{expected_dataset_dir_name}' not found directly. Checking for variations...")
            for item in os.listdir(script_dir):
                item_path = os.path.join(script_dir, item)
                # Check if it's a directory and contains typical dataset subfolders
                if os.path.isdir(item_path) and all(os.path.exists(os.path.join(item_path, sub)) for sub in ['train', 'valid', 'test']):
                    print(f"Found potential dataset folder: '{item}'")
                    # Try renaming if it's different
                    if item != expected_dataset_dir_name:
                        print(f"Attempting to rename '{item}' to '{expected_dataset_dir_name}'...")
                        try:
                            os.rename(item_path, dataset_dir)
                            print("Rename successful.")
                            actual_downloaded_folder = dataset_dir
                        except OSError as e:
                            print(f"Error renaming folder: {e}. Using actual name '{item}'.")
                            actual_downloaded_folder = item_path
                    else:
                        actual_downloaded_folder = dataset_dir # Already correct name
                    break # Stop after finding the first likely match

        # Final check if we found and verified the folder structure
        if actual_downloaded_folder and \
           os.path.exists(os.path.join(actual_downloaded_folder, 'train')) and \
           os.path.exists(os.path.join(actual_downloaded_folder, 'valid')) and \
           os.path.exists(os.path.join(actual_downloaded_folder, 'test')):
            print(f"Dataset structure verified in: {actual_downloaded_folder}")
            download_success_flag = True
        else:
             print("\n--- DOWNLOAD VERIFICATION FAILED ---")
             print("Could not find or verify the downloaded dataset directory with 'train', 'valid', 'test' subfolders.")
             print(f"Please check the contents of '{script_dir}' manually.")
             print("Possible issues:")
             print("- Roboflow API key might be invalid or lack permissions.")
             print("- Network issues during download.")
             print("- Roboflow library might have downloaded to an unexpected location or failed silently.")
             return # Stop if download verification failed

    except Exception as e:
        print(f"\n--- An error occurred during the dataset process ---")
        print(f"Error details: {e}")
        print("Please ensure your API key is correct and you have network access.")
        return

    # Proceed only if download flag is true
    if download_success_flag:
        # --- Download Model Weights ---
        print("\n--- Downloading Model Weights ---")
        weights_dir = os.path.join(script_dir, "weights")
        weights_path = os.path.join(weights_dir, "best.pt")
        if os.path.exists(weights_path):
            print(f"Model weights '{os.path.relpath(weights_path)}' already exist. Skipping download.")
        else:
            weights_url = "https://github.com/Vaibhav0120/ATM-Theft-Detection/releases/download/weights/best.pt"
            os.makedirs(weights_dir, exist_ok=True)
            download_file(weights_url, weights_path)

        print("\n✅ Setup potentially complete (Dataset download verified). Please check folder contents.")
    else:
         print("\n❌ Setup incomplete due to dataset download/verification failure.")


if __name__ == "__main__":
    main()