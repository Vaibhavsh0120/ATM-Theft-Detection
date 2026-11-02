from __future__ import annotations

import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import requests
from dotenv import load_dotenv
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from repo_paths import (
    DATASET_DIR,
    DATASET_VERSION,
    DATA_DIR,
    PUBLISHED_MODEL_PATH,
    REPO_ROOT,
    ensure_layout,
)

WEIGHTS_URL = "https://huggingface.co/Vaibhavsh0120/ATM-Theft-Detection/resolve/main/weights/best.pt"
ROBOFLOW_WORKSPACE = "vaibhav-7tcrm"
ROBOFLOW_PROJECT = "atm-theft-detection-f8ezg"


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
    except requests.RequestException:
        raise RuntimeError(
            f"Failed to download '{destination.name}'. Check connectivity and download access."
        ) from None

    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024

    print(f"Downloading '{destination.name}'...")
    partial = destination.with_suffix(destination.suffix + ".part")
    with (
        response,
        partial.open("wb") as handle,
        tqdm(
            desc=destination.name,
            total=total_size,
            unit="iB",
            unit_scale=True,
        ) as progress,
    ):
        for chunk in response.iter_content(block_size):
            progress.update(len(chunk))
            handle.write(chunk)

    if partial.stat().st_size == 0:
        raise RuntimeError(f"Downloaded file '{destination}' is empty.")
    partial.replace(destination)


def load_api_key() -> str:
    load_dotenv(REPO_ROOT / ".env")
    api_key = os.getenv("ROBOFLOW_API_KEY")

    if api_key and api_key != "YOUR_API_KEY":
        return api_key

    if not sys.stdin.isatty():
        raise RuntimeError("Set ROBOFLOW_API_KEY in the environment or root .env before downloading.")
    entered_key = input("Enter your Roboflow Private API Key: ").strip()
    if not entered_key:
        raise RuntimeError("ROBOFLOW_API_KEY is required to download the dataset.")
    return entered_key


def download_dataset(api_key: str) -> None:
    print("--- Downloading dataset from Roboflow ---")
    endpoint = f"https://api.roboflow.com/{ROBOFLOW_WORKSPACE}/{ROBOFLOW_PROJECT}/{DATASET_VERSION}/yolov8"
    try:
        with requests.get(endpoint, params={"api_key": api_key}, timeout=30) as response:
            response.raise_for_status()
            link = response.json()["export"]["link"]
    except (requests.RequestException, ValueError, KeyError):
        # Do not print the request URL: Roboflow credentials are query parameters.
        raise RuntimeError("Could not obtain the Roboflow export. Check the API key and dataset access.") from None
    with tempfile.TemporaryDirectory(prefix="atm-dataset-", dir=DATA_DIR) as temp:
        stage = Path(temp)
        archive = stage / "dataset.zip"
        download_file(link, archive)
        extracted = stage / "dataset"
        extracted.mkdir()
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                target = (extracted / member.filename).resolve()
                if not target.is_relative_to(extracted.resolve()):
                    raise RuntimeError("The dataset archive contains an unsafe path.")
            bundle.extractall(extracted)
        if not (extracted / "train/images").is_dir() or not (extracted / "data.yaml").is_file():
            raise RuntimeError("Downloaded dataset is missing train/images or data.yaml; existing data was preserved.")
        backup = DATASET_DIR.with_name(DATASET_DIR.name + ".previous")
        if backup.exists():
            raise RuntimeError(
                f"A previous dataset backup already exists at '{backup}'. Resolve it before downloading again."
            )
        if DATASET_DIR.exists():
            DATASET_DIR.rename(backup)
        try:
            extracted.rename(DATASET_DIR)
        except Exception:
            if backup.exists():
                backup.rename(DATASET_DIR)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    for split in ("train", "valid", "test"):
        status = "found" if (DATASET_DIR / split).exists() else "missing"
        print(f"  - {split}: {status}")


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
