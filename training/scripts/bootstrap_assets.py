from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv
from roboflow import Roboflow  # type: ignore
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from repo_paths import (
    DATASET_DIR,
    DATASET_DIR_NAME,
    DATASET_VERSION,
    DATA_DIR,
    PUBLISHED_MODEL_PATH,
    REPO_ROOT,
    ensure_layout,
)

WEIGHTS_URL = "https://github.com/Vaibhav0120/ATM-Theft-Detection/releases/download/weights/best.pt"
ROBOFLOW_WORKSPACE = "vaibhav-7tcrm"
ROBOFLOW_PROJECT = "atm-theft-detection-f8ezg"


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to download '{destination.name}': {exc}") from exc

    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024

    print(f"Downloading '{destination.name}'...")
    with destination.open("wb") as handle, tqdm(
        desc=destination.name,
        total=total_size,
        unit="iB",
        unit_scale=True,
    ) as progress:
        for chunk in response.iter_content(block_size):
            progress.update(len(chunk))
            handle.write(chunk)

    if destination.stat().st_size == 0:
        raise RuntimeError(f"Downloaded file '{destination}' is empty.")


def load_api_key() -> str:
    load_dotenv(REPO_ROOT / ".env")
    api_key = os.getenv("ROBOFLOW_API_KEY")

    if api_key and api_key != "YOUR_API_KEY":
        return api_key

    entered_key = input("Enter your Roboflow Private API Key: ").strip()
    if not entered_key:
        raise RuntimeError("ROBOFLOW_API_KEY is required to download the dataset.")
    return entered_key


def download_dataset(api_key: str) -> None:
    print("--- Downloading dataset from Roboflow ---")
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
    version = project.version(DATASET_VERSION)

    if DATASET_DIR.exists():
        print(f"Removing existing dataset at '{DATASET_DIR}'...")
        shutil.rmtree(DATASET_DIR)

    original_cwd = Path.cwd()
    os.chdir(DATA_DIR)
    try:
        dataset = version.download("yolov8")
    finally:
        os.chdir(original_cwd)

    downloaded_path = Path(dataset.location).resolve()
    expected_path = DATASET_DIR.resolve()

    if downloaded_path != expected_path:
        if expected_path.exists():
            shutil.rmtree(expected_path)
        downloaded_path.rename(expected_path)

    for split in ("train", "valid", "test"):
        status = "found" if (DATASET_DIR / split).exists() else "missing"
        print(f"  - {split}: {status}")

    if not (DATASET_DIR / "train").exists():
        raise RuntimeError(
            f"Downloaded dataset is missing the 'train' split in '{DATASET_DIR_NAME}'."
        )


def download_published_weights() -> None:
    print("--- Ensuring published model weights exist ---")
    if PUBLISHED_MODEL_PATH.exists():
        print(f"Model weights already present at '{PUBLISHED_MODEL_PATH}'.")
        return

    download_file(WEIGHTS_URL, PUBLISHED_MODEL_PATH)
    print(f"Saved published weights to '{PUBLISHED_MODEL_PATH}'.")


def main() -> None:
    ensure_layout()

    try:
        api_key = load_api_key()
        download_dataset(api_key)
        download_published_weights()
    except Exception as exc:
        print(f"Setup failed: {exc}")
        raise SystemExit(1) from exc

    print("Setup complete. Dataset and published weights are ready.")


if __name__ == "__main__":
    main()
