"""Regression coverage for data preparation, publishing, and the Space."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training" / "scripts"))


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bad_label_leaves_file_unchanged(tmp_path):
    remap = load("remap", "training/scripts/remap_labels.py")
    path = tmp_path / "sample.txt"
    original = "1 0.5 0.5 0.2 0.2\n99 0.5 0.5 0.2 0.2\n"
    path.write_text(original)
    with pytest.raises(ValueError):
        remap.remap_file(path, set())
    assert path.read_text() == original


def test_label_mapping(tmp_path):
    remap = load("remap", "training/scripts/remap_labels.py")
    path = tmp_path / "sample.txt"
    path.write_text("1 0.5 0.5 0.2 0.2\n0 0.5 0.5 0.2 0.2\n")
    remap.remap_file(path, set())
    assert path.read_text().splitlines() == ["0 0.5 0.5 0.2 0.2", "1 0.5 0.5 0.2 0.2"]


@pytest.fixture(scope="module")
def app():
    return load("space_app", "deploy/huggingface-space/app.py")


def test_explicit_hub_model_overrides_bundled(app, monkeypatch, tmp_path):
    import huggingface_hub

    model = tmp_path / "remote.pt"
    model.touch()
    monkeypatch.setattr(app, "MODEL_PATH_ENV", None)
    monkeypatch.setattr(app, "HF_MODEL_REPO_ID", "owner/new-model")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda **kwargs: str(model))
    app.resolve_model_path.cache_clear()
    try:
        assert app.resolve_model_path() == model
    finally:
        app.resolve_model_path.cache_clear()


def test_inference_events_share_one_queue(app):
    events = [
        d for d in app.demo.fns.values() if d.fn in (app.predict_photo, app.predict_video, app.predict_live_frame)
    ]
    assert len(events) == 3
    assert len({event.concurrency_id for event in events}) == 1
    assert all(event.queue and event.concurrency_limit == 1 for event in events)


def test_empty_inputs(app):
    assert app.predict_photo(None, 0.25, 0.45)[0] is None
    assert app.predict_video(None, 0.25, 0.45)[0] is None
    assert app.predict_live_frame(None, 0.25, 0.45)[0] is None


def test_publisher_rejects_non_hub_target():
    publish = load("publish", "scripts/push_hf.py")
    with pytest.raises(ValueError):
        publish.resolve_repo_id("https://github.com/owner/repo", "space")


def test_publisher_dry_run_has_no_side_effects(monkeypatch):
    publish = load("publish", "scripts/push_hf.py")
    monkeypatch.setattr(
        publish,
        "parse_args",
        lambda: SimpleNamespace(
            target="both",
            model_repo="owner/model",
            space_repo="owner/space",
            branch="main",
            bundle_space_model=True,
            refresh_model_from_training=True,
            dry_run=True,
        ),
    )
    monkeypatch.setattr(publish, "refresh_model_folder", lambda *args: pytest.fail("dry run mutated assets"))
    publish.main()


def test_checkpoint_photo_and_live(app):
    from PIL import Image
    import ultralytics

    sample = Image.open(Path(ultralytics.__file__).parent / "assets/zidane.jpg")
    for predict in (app.predict_photo, app.predict_live_frame):
        image, summary = predict(sample, 0.25, 0.45)
        assert image is not None, summary
        assert image.size == sample.size
        assert "Error" not in summary
    assert app.load_model().names == {0: "Face_Covered", 1: "Face_Uncovered"}


def test_real_video_inference(app, tmp_path):
    import cv2
    import numpy as np

    path = tmp_path / "short.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5, (96, 96))
    assert writer.isOpened()
    for _ in range(3):
        writer.write(np.zeros((96, 96, 3), dtype=np.uint8))
    writer.release()
    output, summary = app.predict_video(str(path), 0.25, 0.45)
    assert output is not None, summary
    result = cv2.VideoCapture(output)
    try:
        assert result.isOpened()
        assert int(result.get(cv2.CAP_PROP_FRAME_COUNT)) == 3
    finally:
        result.release()
    assert Path(output).stat().st_size > 0


def test_remap_validates_whole_dataset_before_writing(monkeypatch, tmp_path):
    import yaml

    remap = load("remap_batch", "training/scripts/remap_labels.py")
    labels = tmp_path / "train/labels"
    labels.mkdir(parents=True)
    first = labels / "a.txt"
    first.write_text("1 0.5 0.5 0.2 0.2\n")
    (labels / "b.txt").write_text("99 0.5 0.5 0.2 0.2\n")
    names = [
        "Cuong",
        "Helmet",
        "Hung",
        "Lau-Ka-Fai",
        "Trung",
        "Tuan",
        "Vu",
        "balaclava",
        "concealing glasses",
        "cover",
        "face",
        "hand",
        "mask",
        "medicine mask",
        "non-concealing glasses",
        "normal",
        "nothing",
        "person-with-mask",
        "person-without-mask",
        "scarf",
        "thief_mask",
    ]
    (tmp_path / "data.yaml").write_text(yaml.safe_dump({"names": names}))
    monkeypatch.setattr(remap, "DATASET_DIR", tmp_path)
    monkeypatch.setattr(remap, "REMAPPED_MARKER", tmp_path / ".labels_remapped")
    with pytest.raises(ValueError):
        remap.main()
    assert first.read_text().startswith("1 ")
    assert not remap.REMAPPED_MARKER.exists()


def test_publisher_preserves_history(monkeypatch, tmp_path):
    publish = load("publish_safe", "scripts/push_hf.py")
    calls = []
    api = SimpleNamespace(
        repo_info=lambda **kwargs: SimpleNamespace(sha="current-sha"),
        upload_folder=lambda **kwargs: calls.append(kwargs) or SimpleNamespace(commit_url="https://hf.co/commit"),
    )
    publish.publish(api, tmp_path, "owner/repo", "space", SimpleNamespace(branch="main"))
    assert calls[0]["parent_commit"] == "current-sha"
    assert "delete_patterns" not in calls[0]
    assert ".env" in calls[0]["ignore_patterns"]


def test_standalone_production_path():
    # Evaluate path setup with POSIX semantics even on the Windows dev host.
    import ast
    from pathlib import PurePosixPath

    source = (ROOT / "deploy/huggingface-space/app.py").read_text(encoding="utf-8")
    namespace = {"ROOT": PurePosixPath("/app")}
    for node in ast.parse(source).body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id in ("REPO_ROOT", "DEFAULT_LOCAL_MODEL_CANDIDATES"):
                expression = ast.Expression(node.value)
                namespace[node.target.id] = eval(compile(expression, "<path-setup>", "eval"), namespace)
    assert namespace["DEFAULT_LOCAL_MODEL_CANDIDATES"][0] == PurePosixPath("/app/weights/best.pt")


@pytest.mark.parametrize("valid", [True, False])
def test_dataset_download_preserves_existing_until_validated(monkeypatch, tmp_path, valid):
    import zipfile

    setup = load("bootstrap_test", "training/scripts/bootstrap_assets.py")
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "original.txt").write_text("keep")
    monkeypatch.setattr(setup, "DATA_DIR", tmp_path)
    monkeypatch.setattr(setup, "DATASET_DIR", dataset)

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def raise_for_status(self):
            pass

        def json(self):
            return {"export": {"link": "https://example.com/export.zip"}}

    monkeypatch.setattr(setup.requests, "get", lambda *args, **kwargs: Response())

    def download(url, path):
        with zipfile.ZipFile(path, "w") as archive:
            if valid:
                archive.writestr("train/images/", "")
                archive.writestr("data.yaml", "nc: 21\n")
            else:
                archive.writestr("unexpected.txt", "bad export")

    monkeypatch.setattr(setup, "download_file", download)
    if valid:
        setup.download_dataset("test-key")
        assert (dataset / "data.yaml").is_file()
        assert not (dataset / "original.txt").exists()
        assert not dataset.with_name("dataset.previous").exists()
    else:
        with pytest.raises(RuntimeError):
            setup.download_dataset("test-key")
        assert (dataset / "original.txt").read_text() == "keep"
