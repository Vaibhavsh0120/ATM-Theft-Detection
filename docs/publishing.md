# Publishing to Hugging Face

GitHub is the canonical source. Publish `deploy/huggingface-model/` to a **model** repository and `deploy/huggingface-space/` to a **Gradio Space**. The root README is for GitHub; the Space README contains its required YAML settings.

## Authenticate and preview

```bash
python -m pip install huggingface_hub==1.9.2 python-dotenv==1.1.1
hf auth login
python scripts/push_hf.py --dry-run
```

Copy `.env.example` to `.env` first, or pass destination IDs:

```bash
python scripts/push_hf.py --model-repo owner/model --space-repo owner/space --dry-run
```

The CLI accepts `owner/repo`, HTTPS Hugging Face URLs, or existing Git remote names. It rejects GitHub URLs. Create the target repositories on Hugging Face first; for Spaces select Gradio. `--branch` selects an existing branch.

## Publish

```bash
# Publish both current deploy folders; refresh Space weights from model weights
python scripts/push_hf.py --bundle-space-model

# Publish only one target
python scripts/push_hf.py --target model
python scripts/push_hf.py --target space

# Publish the newest local training checkpoint to both targets
python scripts/push_hf.py --refresh-model-from-training --bundle-space-model
```

The publisher uses the Hub upload API with parent-commit conflict detection. It preserves history and unrelated remote files, excludes secrets/caches, and lets the Hub handle binary storage. Removed local files are not deleted remotely; remove obsolete remote files explicitly when needed. Each destination is a separate commit, so a model upload can succeed even if the following Space upload fails. Rerunning is safe.

A dry run resolves local configuration and prints a plan without refreshing, copying or uploading assets; it does not check remote permissions. No Git LFS client is needed for publishing through the API, but Git LFS is needed to obtain real checkpoints when cloning GitHub.

## Model selection in the Space

Precedence is explicit `MODEL_PATH`, explicit `HF_MODEL_REPO_ID` + `HF_MODEL_FILENAME` (default `weights/best.pt`), bundled `weights/best.pt`, then the sibling monorepo model folder. An explicitly configured but unavailable model reports an error instead of silently falling back to old weights.

Set runtime variables in **Space Settings → Variables and secrets**. The root `.env` configures local preparation/publishing; the standalone app does not load it. Private Hub downloads require `HF_TOKEN` as a Space secret. Only use trusted PyTorch checkpoints.

## GitHub Actions

- `Checks` runs on pushes to main and pull requests, plus manual dispatch.
- `Publish Hugging Face` is manual to avoid surprise deployments.
- Add a GitHub repository **secret** `HF_TOKEN` with write access to both targets.
- Optional repository **variables** `HF_MODEL_REPO_ID` and `HF_SPACE_REPO_ID` override the current owner/repo defaults.
- The publish workflow uploads the committed deploy assets and refreshes the uploaded Space bundle. It does not train or select uncommitted local results.

After publishing, check the [Space runtime](https://huggingface.co/spaces/Vaibhavsh0120/ATM-Theft-Detection), open the [website](https://vaibhavsh0120-atm-theft-detection.hf.space), and run photo/video inference. A successful upload alone does not prove the app is ready. Free CPU Spaces may sleep and need a cold start.
