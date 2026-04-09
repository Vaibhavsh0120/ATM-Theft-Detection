from __future__ import annotations

from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
TRAINING_ROOT = SCRIPT_DIR.parent
REPO_ROOT = TRAINING_ROOT.parent

CONFIGS_DIR = TRAINING_ROOT / "configs"
DATA_DIR = TRAINING_ROOT / "data"
MODELS_DIR = TRAINING_ROOT / "models"
PRETRAINED_DIR = MODELS_DIR / "pretrained"
EXPORTS_DIR = MODELS_DIR / "exports"
OUTPUTS_DIR = TRAINING_ROOT / "outputs"

HF_MODEL_ROOT = REPO_ROOT / "deploy" / "huggingface-model"
HF_MODEL_WEIGHTS_DIR = HF_MODEL_ROOT / "weights"
HF_MODEL_EXPORTS_DIR = HF_MODEL_ROOT / "exports"

DATASET_VERSION = 4
DATASET_DIR_NAME = f"ATM-Theft-Detection-{DATASET_VERSION}"
DATASET_DIR = DATA_DIR / DATASET_DIR_NAME
MERGED_DATASET_YAML = CONFIGS_DIR / "merged_dataset.yaml"

DETECT_RUNS_DIR = OUTPUTS_DIR / "runs" / "detect"
TRAINING_RUN_DIR = DETECT_RUNS_DIR / "train"
BEST_TRAINED_MODEL_PATH = TRAINING_RUN_DIR / "weights" / "best.pt"
LAST_TRAINED_MODEL_PATH = TRAINING_RUN_DIR / "weights" / "last.pt"

PUBLISHED_MODEL_PATH = HF_MODEL_WEIGHTS_DIR / "best.pt"
TFLITE_EXPORT_PATH = EXPORTS_DIR / "best_full_integer_quant.tflite"
PUBLISHED_TFLITE_PATH = HF_MODEL_EXPORTS_DIR / TFLITE_EXPORT_PATH.name


def ensure_layout() -> None:
    for path in (
        DATA_DIR,
        PRETRAINED_DIR,
        EXPORTS_DIR,
        OUTPUTS_DIR,
        HF_MODEL_WEIGHTS_DIR,
        HF_MODEL_EXPORTS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def find_latest_run_artifact(filename: str) -> Path | None:
    candidates = [path for path in DETECT_RUNS_DIR.glob(f"*/weights/{filename}") if path.is_file()]
    if not candidates:
        return None

    return max(candidates, key=lambda path: path.stat().st_mtime)


def resolve_latest_trained_model() -> Path | None:
    return find_latest_run_artifact("best.pt")


def resolve_latest_resume_checkpoint() -> Path | None:
    return find_latest_run_artifact("last.pt")


def resolve_preferred_model() -> Path:
    latest_trained_model = resolve_latest_trained_model()
    if latest_trained_model is not None:
        return latest_trained_model

    if PUBLISHED_MODEL_PATH.exists():
        return PUBLISHED_MODEL_PATH

    raise FileNotFoundError(
        "No model weights were found. Train a checkpoint first or run "
        "'python training/scripts/bootstrap_assets.py' to fetch the published weights."
    )
