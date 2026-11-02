"""Publish deploy folders as normal Hub commits, preserving remote history."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv
from huggingface_hub import HfApi
from huggingface_hub.utils import validate_repo_id

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = REPO_ROOT / "deploy/huggingface-model"
SPACE_ROOT = REPO_ROOT / "deploy/huggingface-space"
IGNORE_PATTERNS = [
    ".git/**",
    ".runtime/**",
    "**/.runtime/**",
    "**/__pycache__/**",
    "*.pyc",
    "**/*.pyc",
    ".env",
    ".env.*",
    "**/.env",
    "**/.env.*",
]


def resolve_repo_id(value: str, repo_type: str) -> str:
    remote = subprocess.run(["git", "remote", "get-url", value], cwd=REPO_ROOT, capture_output=True, text=True)
    value = remote.stdout.strip() if remote.returncode == 0 else value
    if "://" in value:
        url = urlparse(value)
        if url.scheme != "https" or url.hostname != "huggingface.co" or url.username or url.query or url.fragment:
            raise ValueError("Use HTTPS Hugging Face repository URLs without credentials.")
        value = url.path.strip("/").removesuffix(".git")
        if repo_type == "space":
            if not value.startswith("spaces/"):
                raise ValueError("Space URLs must include /spaces/.")
            value = value.removeprefix("spaces/")
    validate_repo_id(value)
    if value.count("/") != 1:
        raise ValueError("Use an owner/repository ID.")
    return value


def parse_args() -> argparse.Namespace:
    load_dotenv(REPO_ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("model", "space", "both"), default="both")
    parser.add_argument("--model-repo", default=os.getenv("HF_MODEL_REPO_ID") or os.getenv("HF_MODEL_REMOTE"))
    parser.add_argument("--space-repo", default=os.getenv("HF_SPACE_REPO_ID") or os.getenv("HF_SPACE_REMOTE"))
    parser.add_argument("--branch", default="main", help="Existing Hub branch (default: main).")
    parser.add_argument(
        "--bundle-space-model", action="store_true", help="Refresh the Space checkpoint from the model folder."
    )
    parser.add_argument("--refresh-model-from-training", action="store_true")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the plan without changing files or contacting the Hub."
    )
    return parser.parse_args()


def refresh_model_folder(refresh_from_training: bool) -> None:
    if not refresh_from_training:
        return
    runs = REPO_ROOT / "training/outputs/runs/detect"
    candidates = list(runs.glob("*/weights/best.pt"))
    if not candidates:
        raise FileNotFoundError(f"No trained checkpoint found under {runs}.")
    latest = max(candidates, key=lambda path: path.stat().st_mtime)
    shutil.copy2(latest, MODEL_ROOT / "weights/best.pt")
    export = REPO_ROOT / "training/models/exports/best_full_integer_quant.tflite"
    if export.is_file():
        shutil.copy2(export, MODEL_ROOT / "exports" / export.name)


def publish(api: HfApi, source: Path, repo_id: str, repo_type: str, args: argparse.Namespace) -> None:
    parent = api.repo_info(repo_id=repo_id, repo_type=repo_type, revision=args.branch).sha
    result = api.upload_folder(
        repo_id=repo_id,
        repo_type=repo_type,
        folder_path=str(source),
        revision=args.branch,
        parent_commit=parent,
        ignore_patterns=IGNORE_PATTERNS,
        commit_message=f"Update ATM detection {repo_type} from GitHub source",
    )
    print(result.commit_url)


def main() -> None:
    args = parse_args()
    try:
        targets = []
        for kind, source, value in (("model", MODEL_ROOT, args.model_repo), ("space", SPACE_ROOT, args.space_repo)):
            if args.target not in (kind, "both"):
                continue
            if not value:
                raise ValueError(f"Set HF_{kind.upper()}_REPO_ID or pass --{kind}-repo.")
            targets.append((kind, source, resolve_repo_id(value, kind)))
        for kind, source, repo_id in targets:
            print(f"{'[dry-run] ' if args.dry_run else ''}{source} -> {kind}:{repo_id}@{args.branch}")
        if args.dry_run:
            print(
                f"Refresh training weights: {args.refresh_model_from_training}; refresh Space bundle: {args.bundle_space_model}"
            )
            return
        api = HfApi()
        api.whoami()
        for kind, _, repo_id in targets:
            api.repo_info(repo_id=repo_id, repo_type=kind, revision=args.branch)
        refresh_model_folder(args.refresh_model_from_training)
        for kind, source, repo_id in targets:
            if kind == "space" and args.bundle_space_model:
                with tempfile.TemporaryDirectory(prefix="atm-publish-") as temp:
                    snapshot = Path(temp) / "space"
                    shutil.copytree(
                        source, snapshot, ignore=shutil.ignore_patterns(".git", ".runtime", "__pycache__", ".env*")
                    )
                    (snapshot / "weights").mkdir(exist_ok=True)
                    shutil.copy2(MODEL_ROOT / "weights/best.pt", snapshot / "weights/best.pt")
                    publish(api, snapshot, repo_id, kind, args)
            else:
                publish(api, source, repo_id, kind, args)
    except Exception as exc:
        raise SystemExit(f"Publish failed: {exc}") from exc


if __name__ == "__main__":
    main()
