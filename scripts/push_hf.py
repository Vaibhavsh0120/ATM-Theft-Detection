from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAINING_RUNS_ROOT = REPO_ROOT / "training" / "outputs" / "runs" / "detect"
TRAINING_TFLITE = REPO_ROOT / "training" / "models" / "exports" / "best_full_integer_quant.tflite"
HF_MODEL_ROOT = REPO_ROOT / "deploy" / "huggingface-model"
HF_MODEL_WEIGHTS = HF_MODEL_ROOT / "weights" / "best.pt"
HF_MODEL_EXPORTS = HF_MODEL_ROOT / "exports" / "best_full_integer_quant.tflite"
SPACE_ROOT = REPO_ROOT / "deploy" / "huggingface-space"
TEMP_PUBLISH_ROOT = REPO_ROOT / ".runtime" / "hf-push-temp"
DEFAULT_BRANCH = "main"
DEFAULT_TARGET = "both"
SKIP_DIR_NAMES = {".git", ".runtime", "__pycache__", ".pytest_cache"}
SKIP_FILE_SUFFIXES = {".pyc", ".pyo"}


def run_git(*args: str, cwd: Path, capture_output: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        text=True,
        capture_output=capture_output,
    )


def git_remote_url(name: str) -> str | None:
    result = subprocess.run(
        ["git", "remote", "get-url", name],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        return None

    return result.stdout.strip() or None


def resolve_repo_url(value: str, repo_type: str) -> str:
    remote_url = git_remote_url(value)
    if remote_url is not None:
        return remote_url

    if value.startswith(("https://", "http://")):
        return value

    if "/" not in value:
        raise ValueError(
            f"Could not resolve '{value}' as a git remote or Hugging Face repo id for the {repo_type} repo."
        )

    if repo_type == "model":
        return f"https://huggingface.co/{value}"
    if repo_type == "space":
        return f"https://huggingface.co/spaces/{value}"

    raise ValueError(f"Unsupported repo type '{repo_type}'.")


def parse_args() -> argparse.Namespace:
    load_dotenv(REPO_ROOT / ".env")

    parser = argparse.ArgumentParser(
        description="Push the monorepo's Hugging Face model and/or Space folders as standalone repos."
    )
    parser.add_argument(
        "--target",
        choices=("model", "space", "both"),
        default=DEFAULT_TARGET,
        help="Which standalone deploy surface to push.",
    )
    parser.add_argument(
        "--model-repo",
        default=os.getenv("HF_MODEL_REPO_ID") or os.getenv("HF_MODEL_REMOTE"),
        help="Model repo id, full URL, or existing git remote name.",
    )
    parser.add_argument(
        "--space-repo",
        default=os.getenv("HF_SPACE_REPO_ID") or os.getenv("HF_SPACE_REMOTE"),
        help="Space repo id, full URL, or existing git remote name.",
    )
    parser.add_argument(
        "--branch",
        default=DEFAULT_BRANCH,
        help="Remote branch to push to. Defaults to 'main'.",
    )
    parser.add_argument(
        "--bundle-space-model",
        action="store_true",
        help="Copy deploy/huggingface-model/weights/best.pt into the pushed Space snapshot as weights/best.pt.",
    )
    parser.add_argument(
        "--refresh-model-from-training",
        action="store_true",
        help="Refresh deploy/huggingface-model from the latest local training outputs before pushing.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned actions without pushing anything.",
    )
    return parser.parse_args()


def require_repo_value(value: str | None, label: str) -> str:
    if value:
        return value

    raise ValueError(
        f"No {label} was provided. Pass it explicitly or set the matching value in '.env'."
    )


def resolve_latest_trained_model() -> Path | None:
    candidates = [path for path in TRAINING_RUNS_ROOT.glob("*/weights/best.pt") if path.is_file()]
    if not candidates:
        return None

    return max(candidates, key=lambda path: path.stat().st_mtime)


def normalize_model_repo_weights() -> None:
    if HF_MODEL_WEIGHTS.exists():
        return

    candidates = [path for path in HF_MODEL_WEIGHTS.parent.glob("*.pt") if path.is_file()]
    if not candidates:
        return

    if len(candidates) > 1:
        raise FileExistsError(
            f"Expected a single checkpoint under '{HF_MODEL_WEIGHTS.parent}', found: "
            f"{', '.join(str(path.name) for path in candidates)}"
        )

    candidate = candidates[0]
    print(f"Normalizing model checkpoint name: '{candidate.name}' -> '{HF_MODEL_WEIGHTS.name}'.")
    candidate.replace(HF_MODEL_WEIGHTS)


def copy_if_present(source: Path, destination: Path) -> bool:
    if not source.exists():
        print(f"Skipping '{source.name}': source file not found at '{source}'.")
        return False

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    print(f"Copied '{source}' -> '{destination}'.")
    return True


def copy_snapshot(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)

    for item in source.iterdir():
        if item.name in SKIP_DIR_NAMES:
            continue
        if item.is_file() and item.suffix in SKIP_FILE_SUFFIXES:
            continue

        target = destination / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)


def configure_local_git_identity(repo_dir: Path) -> None:
    values: dict[str, str] = {}
    for key in ("user.name", "user.email"):
        result = subprocess.run(
            ["git", "config", "--get", key],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
        )
        value = result.stdout.strip()
        if value:
            values[key] = value

    values.setdefault("user.name", "ATM-Theft-Detection Publisher")
    values.setdefault("user.email", "atm-theft-detection-publisher@example.invalid")

    for key, value in values.items():
        run_git("config", key, value, cwd=repo_dir)


def enable_git_lfs_if_needed(repo_dir: Path) -> None:
    gitattributes = repo_dir / ".gitattributes"
    if not gitattributes.exists():
        return

    if "filter=lfs" not in gitattributes.read_text(encoding="utf-8"):
        return

    result = subprocess.run(
        ["git", "lfs", "install", "--local"],
        cwd=repo_dir,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "git-lfs is required for this push but could not be initialized in the temporary publish repo.\n"
            f"{result.stderr.strip()}"
        )


def snapshot_uses_lfs(source_dir: Path) -> bool:
    gitattributes = source_dir / ".gitattributes"
    if not gitattributes.exists():
        return False

    return "filter=lfs" in gitattributes.read_text(encoding="utf-8")


def stage_without_lfs(repo_dir: Path) -> None:
    gitattributes = repo_dir / ".gitattributes"
    if not gitattributes.exists():
        run_git("add", "-A", cwd=repo_dir)
        return

    backup = repo_dir.parent / f"{repo_dir.name}.gitattributes.publish-backup"
    gitattributes.replace(backup)
    try:
        run_git("add", "-A", cwd=repo_dir)
        backup.replace(gitattributes)
        run_git("add", ".gitattributes", cwd=repo_dir)
    finally:
        if backup.exists():
            backup.replace(gitattributes)


def create_temp_repo(source_dir: Path, commit_message: str, use_lfs: bool) -> Path:
    TEMP_PUBLISH_ROOT.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix="atm-hf-push-", dir=TEMP_PUBLISH_ROOT))
    try:
        copy_snapshot(source_dir, temp_dir)
        run_git("init", "--initial-branch", DEFAULT_BRANCH, cwd=temp_dir)
        configure_local_git_identity(temp_dir)
        if use_lfs:
            enable_git_lfs_if_needed(temp_dir)
            run_git("add", "-A", cwd=temp_dir)
        else:
            stage_without_lfs(temp_dir)
        run_git("commit", "-m", commit_message, cwd=temp_dir)
        return temp_dir
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def prepare_temp_repo(source_dir: Path, commit_message: str) -> Path:
    try:
        return create_temp_repo(source_dir, commit_message, use_lfs=True)
    except subprocess.CalledProcessError:
        if not snapshot_uses_lfs(source_dir):
            raise

        print("git-lfs staging failed for the temporary publish repo. Retrying without LFS filters for this snapshot.")
        return create_temp_repo(source_dir, commit_message, use_lfs=False)


def bundle_space_model_if_requested(temp_space_dir: Path) -> None:
    if not HF_MODEL_WEIGHTS.exists():
        raise FileNotFoundError(
            f"Cannot bundle the Space model because '{HF_MODEL_WEIGHTS}' does not exist."
        )

    bundled_weights = temp_space_dir / "weights" / "best.pt"
    bundled_weights.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(HF_MODEL_WEIGHTS, bundled_weights)
    print(f"Bundled model weights into the Space snapshot at '{bundled_weights}'.")


def push_snapshot(source_dir: Path, remote_url: str, branch: str, commit_message: str, dry_run: bool) -> None:
    print(f"Preparing snapshot from '{source_dir}'...")
    temp_repo = prepare_temp_repo(source_dir, commit_message)

    try:
        if dry_run:
            print(f"[dry-run] Would push '{source_dir}' to '{remote_url}' on branch '{branch}'.")
            return

        run_git("remote", "add", "origin", remote_url, cwd=temp_repo)
        print(f"Pushing snapshot to '{remote_url}' ({branch})...")
        run_git("push", "origin", f"HEAD:{branch}", "--force", cwd=temp_repo)
    finally:
        shutil.rmtree(temp_repo, ignore_errors=True)


def refresh_model_folder(refresh_from_training: bool) -> None:
    if not refresh_from_training:
        normalize_model_repo_weights()
        print("Using the current deploy/huggingface-model contents as-is.")
        return

    print("Refreshing deploy/huggingface-model from the latest training artifacts...")
    copied_count = 0

    latest_model = resolve_latest_trained_model()
    if latest_model is None:
        print(f"Skipping '{HF_MODEL_WEIGHTS.name}': no trained best.pt file was found under '{TRAINING_RUNS_ROOT}'.")
    elif copy_if_present(latest_model, HF_MODEL_WEIGHTS):
        copied_count += 1

    if copy_if_present(TRAINING_TFLITE, HF_MODEL_EXPORTS):
        copied_count += 1

    if copied_count == 0:
        normalize_model_repo_weights()
        print("No new training artifacts were found. Using the current deploy/huggingface-model contents as-is.")
        return

    normalize_model_repo_weights()
    print(f"Updated '{HF_MODEL_ROOT}' with {copied_count} artifact(s) from local training outputs.")


def push_model(args: argparse.Namespace) -> None:
    model_repo = resolve_repo_url(require_repo_value(args.model_repo, "model repo"), "model")
    refresh_model_folder(args.refresh_model_from_training)
    push_snapshot(
        source_dir=HF_MODEL_ROOT,
        remote_url=model_repo,
        branch=args.branch,
        commit_message="Publish Hugging Face model snapshot",
        dry_run=args.dry_run,
    )


def push_space(args: argparse.Namespace) -> None:
    space_repo = resolve_repo_url(require_repo_value(args.space_repo, "Space repo"), "space")
    refresh_model_folder(args.refresh_model_from_training)

    print(f"Preparing snapshot from '{SPACE_ROOT}'...")
    temp_repo = prepare_temp_repo(SPACE_ROOT, "Publish Hugging Face Space snapshot")

    try:
        if args.bundle_space_model:
            bundle_space_model_if_requested(temp_repo)
            run_git("add", "-A", cwd=temp_repo)
            run_git("commit", "--amend", "--no-edit", cwd=temp_repo)
        elif not args.dry_run:
            print(
                "Space snapshot does not bundle weights. Make sure the Space has "
                "'HF_MODEL_REPO_ID' configured in its variables, or rerun with "
                "'--bundle-space-model'."
            )

        if args.dry_run:
            print(f"[dry-run] Would push '{SPACE_ROOT}' to '{space_repo}' on branch '{args.branch}'.")
            return

        run_git("remote", "add", "origin", space_repo, cwd=temp_repo)
        print(f"Pushing snapshot to '{space_repo}' ({args.branch})...")
        run_git("push", "origin", f"HEAD:{args.branch}", "--force", cwd=temp_repo)
    finally:
        shutil.rmtree(temp_repo, ignore_errors=True)


def main() -> None:
    args = parse_args()

    try:
        if args.target in {"model", "both"}:
            push_model(args)

        if args.target in {"space", "both"}:
            push_space(args)
    except (RuntimeError, ValueError, FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(f"Publish failed: {exc}")
        raise SystemExit(1) from exc

    print("Publish complete.")


if __name__ == "__main__":
    main()
