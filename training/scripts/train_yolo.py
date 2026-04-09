from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

os.environ.setdefault("YOLO_CONFIG_DIR", str(SCRIPT_DIR.parents[1] / ".runtime"))

import torch
from ultralytics import YOLO  # type: ignore
from ultralytics.utils.downloads import attempt_download_asset  # type: ignore

from repo_paths import (
    DETECT_RUNS_DIR,
    MERGED_DATASET_YAML,
    PRETRAINED_DIR,
    REPO_ROOT,
    ensure_layout,
    resolve_latest_resume_checkpoint,
)

DEFAULT_BASE_MODEL = "yolov8n.pt"
DEFAULT_EPOCHS = int(os.getenv("YOLO_EPOCHS", "80"))
IMAGE_SIZE = int(os.getenv("YOLO_IMAGE_SIZE", "640"))
DEFAULT_RUN_NAME = os.getenv("YOLO_RUN_NAME", "train")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the ATM theft detection YOLO model.")
    parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        help="Number of training epochs for a fresh run.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest available 'last.pt' checkpoint under training/outputs/runs/detect.",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="Resume from a specific 'last.pt' checkpoint path. Implies --resume.",
    )
    parser.add_argument(
        "--name",
        default=DEFAULT_RUN_NAME,
        help="Base run name for fresh training. Existing names are auto-incremented by Ultralytics.",
    )

    args = parser.parse_args(argv)
    if args.epochs is not None and args.epochs <= 0:
        parser.error("--epochs must be a positive integer.")
    args.resume = args.resume or args.resume_from is not None
    return args


def resolve_resume_checkpoint(resume_from: Path | None) -> Path:
    if resume_from is not None:
        checkpoint = resume_from.expanduser().resolve()
        if not checkpoint.exists():
            raise FileNotFoundError(f"Resume checkpoint not found at '{checkpoint}'.")
        return checkpoint

    checkpoint = resolve_latest_resume_checkpoint()
    if checkpoint is None:
        raise FileNotFoundError(
            "No resumable checkpoint was found under 'training/outputs/runs/detect'. "
            "Start a fresh run first or pass '--resume-from <path-to-last.pt>'."
        )
    return checkpoint


def resolve_base_model() -> str:
    override = os.getenv("YOLO_BASE_MODEL")
    if override:
        override_path = Path(override).expanduser()
        return str(override_path.resolve()) if override_path.exists() else override

    local_base_model = PRETRAINED_DIR / DEFAULT_BASE_MODEL
    legacy_root_model = REPO_ROOT / DEFAULT_BASE_MODEL

    if not local_base_model.exists() and legacy_root_model.exists():
        print(f"Moving existing base model into '{local_base_model}'...")
        local_base_model.parent.mkdir(parents=True, exist_ok=True)
        legacy_root_model.replace(local_base_model)

    if local_base_model.exists():
        return str(local_base_model)

    print(f"Downloading base model to '{local_base_model}'...")
    downloaded_model = Path(attempt_download_asset(local_base_model)).resolve()
    return str(downloaded_model)


def main() -> None:
    args = parse_args(sys.argv[1:])
    ensure_layout()

    if not MERGED_DATASET_YAML.exists():
        print(f"Dataset YAML not found at '{MERGED_DATASET_YAML}'.")
        print("Run 'python training/scripts/bootstrap_assets.py' and 'python training/scripts/remap_labels.py' first.")
        raise SystemExit(1)

    device = 0 if torch.cuda.is_available() else "cpu"
    epochs = args.epochs if args.epochs is not None else DEFAULT_EPOCHS
    resume_requested = args.resume or args.resume_from is not None

    print("--- Starting YOLO training ---")
    print(f"Dataset config: {MERGED_DATASET_YAML}")
    print(f"Device: {str(device).upper()}")
    print(f"Image size: {IMAGE_SIZE}")
    print(f"Runs directory: {DETECT_RUNS_DIR}")

    if resume_requested:
        try:
            resume_checkpoint = resolve_resume_checkpoint(args.resume_from)
        except FileNotFoundError as exc:
            print(exc)
            raise SystemExit(1) from exc

        print("Resume mode: enabled")
        print(f"Resume checkpoint: {resume_checkpoint}")
        if args.epochs is not None:
            print("Ignoring explicit epoch override while resuming. Ultralytics resumes with the checkpoint's saved epoch target.")

        model = YOLO(str(resume_checkpoint))
        train_kwargs = {
            "resume": True,
            "device": device,
        }
    else:
        base_model = resolve_base_model()
        print("Resume mode: disabled")
        print(f"Base model: {base_model}")
        print(f"Epochs: {epochs}")
        print(f"Run name: {args.name}")

        model = YOLO(base_model)
        train_kwargs = {
            "data": str(MERGED_DATASET_YAML),
            "epochs": epochs,
            "imgsz": IMAGE_SIZE,
            "device": device,
            "exist_ok": False,
            "project": str(DETECT_RUNS_DIR),
            "name": args.name,
        }

    try:
        model.train(**train_kwargs)
    except Exception as exc:
        print(f"Training failed: {exc}")
        raise SystemExit(1) from exc

    save_dir = Path(model.trainer.save_dir).resolve()
    best_checkpoint = Path(model.trainer.best).resolve()
    last_checkpoint = Path(model.trainer.last).resolve()

    print("Training complete.")
    print(f"Run directory: '{save_dir}'")
    print(f"Best model checkpoint: '{best_checkpoint}'")
    print(f"Last checkpoint: '{last_checkpoint}'")


if __name__ == "__main__":
    main()
