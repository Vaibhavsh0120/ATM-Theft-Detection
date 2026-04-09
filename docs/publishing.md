# Publishing To Hugging Face

The simplest workflow is the helper script in `scripts/push_hf.py`. It pushes
the current `deploy/huggingface-model/` and `deploy/huggingface-space/`
snapshots as standalone repos without requiring you to manually run subtree
commands.

## One-Time Setup

1. Authenticate git with Hugging Face.

   `huggingface-cli login`

2. Add your repo ids to the root `.env`:

   ```text
   HF_MODEL_REPO_ID="your-username/your-model-repo"
   HF_SPACE_REPO_ID="your-username/your-space-repo"
   ```

## Push Commands

Push both repos:

`python scripts/push_hf.py`

Push only the model repo:

`python scripts/push_hf.py --target model`

Push only the Space repo:

`python scripts/push_hf.py --target space`

Push the Space repo with a bundled `weights/best.pt` so it works without setting
`HF_MODEL_REPO_ID` in the Space settings:

`python scripts/push_hf.py --bundle-space-model`

## Notes

- `push_hf.py` pushes the current `deploy/huggingface-model/` and `deploy/huggingface-space/` contents by default.
- If you want to refresh `deploy/huggingface-model/` from the latest local training outputs first, use `--refresh-model-from-training`.
- The script accepts Hugging Face repo ids, full URLs, or existing git remote names via `--model-repo` and `--space-repo`.
- The script force-pushes a snapshot of each standalone deploy folder, which matches the old subtree-based publish behavior.
- If you do not bundle weights into the Space repo, configure `HF_MODEL_REPO_ID` in the Space variables so the app can download the model from Hugging Face.
