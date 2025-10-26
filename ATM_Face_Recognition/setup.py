import os
import requests
from roboflow import Roboflow # type: ignore
from tqdm import tqdm
from dotenv import load_dotenv
import shutil
import time

# Load environment variables from .env file (expected in the same directory)
load_dotenv()

def download_file(url, filename):
    """Downloads a file from a URL with a progress bar."""
    try:
        response = requests.get(url, stream=True, timeout=30) 
        response.raise_for_status() # Raise an exception for bad status codes
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024 # 1 Kibibyte

        if total_size_in_bytes == 0:
            print(f"Warning: Content-Length is 0 for {os.path.basename(filename)}. Download might be empty or skipped.")

        print(f"Downloading {os.path.basename(filename)}...")
        with open(filename, 'wb') as file, tqdm(
            desc=os.path.basename(filename),
            total=total_size_in_bytes,
            unit='iB',
            unit_scale=True,
        ) as bar:
            for data in response.iter_content(block_size):
                bar.update(len(data))
                file.write(data)
        
        downloaded_size = os.path.getsize(filename)
        if total_size_in_bytes != 0 and downloaded_size < total_size_in_bytes * 0.9: 
            print(f"Warning: Downloaded file size ({downloaded_size} bytes) is much smaller than expected ({total_size_in_bytes} bytes).")
        elif downloaded_size == 0 and total_size_in_bytes == 0:
            print(f"Warning: Downloaded file size is 0 for {os.path.basename(filename)}.")
        else:
            print(f"Downloaded {os.path.basename(filename)} successfully ({downloaded_size} bytes).")
        return filename 
    
    except requests.exceptions.ConnectTimeout:
        print(f"Failed to download {os.path.basename(filename)}. Error: Connection timed out.")
        exit()
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

    expected_dataset_dir_name = "ATM-Theft-Detection-3"
    dataset_dir = os.path.join(script_dir, expected_dataset_dir_name)
    download_success_flag = False 

    try:
        print("Initializing Roboflow...")
        rf = Roboflow(api_key=api_key)
        print("Accessing workspace 'vaibhav-7tcrm'...")
        project = rf.workspace("vaibhav-7tcrm").project("atm-theft-detection-f8ezg")
        print("Accessing project version 3...")
        version_obj = project.version(3)

        if os.path.exists(dataset_dir):
            print(f"Dataset directory '{expected_dataset_dir_name}' already exists. Removing it to force fresh download.")
            try:
                shutil.rmtree(dataset_dir)
                print(f"Removed existing directory: {dataset_dir}")
                time.sleep(1) 
            except OSError as e:
                print(f"Error removing directory {dataset_dir}: {e}. Please remove manually and retry.")
                return

        print(f"\nAttempting to download dataset via Roboflow API to: {script_dir}")
        print("This might take several minutes depending on dataset size and connection speed...")
        
        final_dataset_path = None
        try:
            dataset = version_obj.download("yolov8") 
            actual_downloaded_path = dataset.location
            print(f"Roboflow download finished. Actual downloaded location: {actual_downloaded_path}")

            desired_path = dataset_dir 
            final_dataset_path = actual_downloaded_path 

            if actual_downloaded_path == desired_path:
                print(f"Dataset is already at the desired location: {desired_path}")
                final_dataset_path = desired_path
            else:
                print(f"Attempting to rename '{os.path.basename(actual_downloaded_path)}' to '{expected_dataset_dir_name}'...")
                try:
                    os.rename(actual_downloaded_path, desired_path)
                    print(f"Rename successful. Dataset is now at: {desired_path}")
                    final_dataset_path = desired_path 
                except OSError as e:
                    print(f"Error renaming folder: {e}. Using original downloaded path: {actual_downloaded_path}")
                    final_dataset_path = actual_downloaded_path

        except (requests.exceptions.ConnectTimeout, requests.exceptions.ConnectionError) as e:
            print("\n--- DATASET DOWNLOAD FAILED ---")
            print("A network connection error occurred while downloading the dataset.")
            print("Please check your connection and ensure your firewall is not blocking Python.")
            print(f"Error details: {e}")
            return
        except Exception as e:
            print(f"\n--- An error occurred during the dataset download process ---")
            print(f"Error details: {e}")
            return

        # -----------------------------------------------------------------
        # --- FIXED VERIFICATION LOGIC ---
        # -----------------------------------------------------------------
        
        print(f"\nVerifying dataset structure in: {final_dataset_path}...")
        
        if not final_dataset_path or not os.path.exists(final_dataset_path):
             print(f"--- VERIFICATION FAILED ---")
             print(f"Final dataset path '{final_dataset_path}' does not exist.")
             return

        # Check for each split individually
        train_exists = os.path.exists(os.path.join(final_dataset_path, 'train'))
        valid_exists = os.path.exists(os.path.join(final_dataset_path, 'valid'))
        test_exists = os.path.exists(os.path.join(final_dataset_path, 'test'))

        print(f"  - 'train' folder: {'Found' if train_exists else 'MISSING'}")
        print(f"  - 'valid' folder: {'Found' if valid_exists else 'MISSING'}")
        print(f"  - 'test' folder:  {'Found' if test_exists else 'MISSING'}")

        # Consider it a success if at least the 'train' folder exists
        if train_exists:
            print("Dataset structure verification successful (found 'train' folder).")
            download_success_flag = True
            if not valid_exists or not test_exists:
                 print("Warning: Dataset seems incomplete. Missing 'valid' or 'test' split.")
        else:
            print("\n--- VERIFICATION FAILED ---")
            print("The essential 'train' folder is missing from the dataset.")
            print("Please check your dataset export on Roboflow.")
            return

        # -----------------------------------------------------------------
        # --- END OF FIXED LOGIC ---
        # -----------------------------------------------------------------

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

        print("\n✅ Setup complete. Dataset and weights are ready.")
    else:
        print("\n❌ Setup incomplete due to dataset download/verification failure.")


if __name__ == "__main__":
    main()