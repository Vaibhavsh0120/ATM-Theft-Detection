from __future__ import annotations

import os
import shutil
import tempfile
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Final, TypeAlias

ROOT: Final[Path] = Path(__file__).resolve().parent
APP_CACHE_DIR = ROOT / ".runtime"
APP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(APP_CACHE_DIR))

import gradio as gr
import cv2
import numpy as np
from PIL import Image as PILImage
from ultralytics import YOLO
from ultralytics.engine.results import Results

ImagePrediction: TypeAlias = tuple[PILImage.Image | None, str]
VideoPrediction: TypeAlias = tuple[str | None, str]

MODEL_PATH_ENV = os.getenv("MODEL_PATH")
HF_MODEL_REPO_ID = os.getenv("HF_MODEL_REPO_ID")
HF_MODEL_FILENAME = os.getenv("HF_MODEL_FILENAME", "weights/best.pt")
DEFAULT_LOCAL_MODEL_CANDIDATES: Final[tuple[Path, ...]] = (
    ROOT / "weights" / "best.pt",
    ROOT.parent / "huggingface-model" / "weights" / "best.pt",
)
TITLE: Final[str] = "ATM Theft Detection"
MODEL_LOCK = Lock()
MAX_VIDEO_SECONDS = 30

THEME = gr.themes.Soft(
    primary_hue="amber",
    secondary_hue="rose",
    neutral_hue="stone",
).set(
    body_background_fill="#f8f1e7",
    body_background_fill_dark="#f8f1e7",
    background_fill_primary="#f8f1e7",
    background_fill_primary_dark="#f8f1e7",
    background_fill_secondary="#fffaf6",
    background_fill_secondary_dark="#fffaf6",
    block_background_fill="#fffaf6",
    block_background_fill_dark="#fffaf6",
    panel_background_fill="#fffaf6",
    panel_background_fill_dark="#fffaf6",
    body_text_color="#221711",
    body_text_color_dark="#221711",
    body_text_color_subdued="#4d3d31",
    body_text_color_subdued_dark="#4d3d31",
    block_title_text_color="#2b1b14",
    block_title_text_color_dark="#2b1b14",
    block_label_text_color="#2b1b14",
    block_label_text_color_dark="#2b1b14",
    block_info_text_color="#4d3d31",
    block_info_text_color_dark="#4d3d31",
    accordion_text_color="#2b1b14",
    accordion_text_color_dark="#2b1b14",
    checkbox_label_text_color="#221711",
    checkbox_label_text_color_dark="#221711",
    checkbox_label_text_color_selected="#2b1b14",
    checkbox_label_text_color_selected_dark="#2b1b14",
    table_text_color="#221711",
    table_text_color_dark="#221711",
    link_text_color="#6e241d",
    link_text_color_dark="#6e241d",
    link_text_color_active="#6e241d",
    link_text_color_active_dark="#6e241d",
    link_text_color_hover="#9b392a",
    link_text_color_hover_dark="#9b392a",
    link_text_color_visited="#6e241d",
    link_text_color_visited_dark="#6e241d",
    button_primary_text_color="#fff7f0",
    button_primary_text_color_dark="#fff7f0",
    button_primary_text_color_hover="#fff7f0",
    button_primary_text_color_hover_dark="#fff7f0",
    button_secondary_text_color="#2b1b14",
    button_secondary_text_color_dark="#2b1b14",
    button_secondary_text_color_hover="#2b1b14",
    button_secondary_text_color_hover_dark="#2b1b14",
    input_placeholder_color="#6b5748",
    input_placeholder_color_dark="#6b5748",
)

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

:root {
  --page-bg: linear-gradient(180deg, #f8f1e7 0%, #f1e4d4 56%, #ead7c4 100%);
  --card-bg: #fffaf6;
  --card-border: rgba(123, 67, 42, 0.14);
  --card-shadow: 0 10px 24px rgba(98, 53, 32, 0.08);
  --accent: #9b392a;
  --accent-dark: #6e241d;
  --accent-soft: rgba(155, 57, 42, 0.10);
  --text-main: #221711;
  --text-strong: #2b1b14;
  --text-muted: #4d3d31;
  --text-soft: #6b5748;
  --gold: #c58d2a;
}

html,
body,
#root {
  background: var(--page-bg);
}

html,
body,
#root,
.gradio-container {
  background: var(--page-bg) !important;
  color: var(--text-main);
  font-family: 'IBM Plex Sans', ui-sans-serif, sans-serif;
  text-rendering: auto;
  color-scheme: light;
}

.gradio-container > .main {
  padding: 0 !important;
}

html,
body,
#root {
  width: 100%;
  min-height: 100%;
  margin: 0;
}

body {
  overflow-x: hidden;
}

.gradio-container {
  width: 100% !important;
  max-width: none !important;
  margin: 0 !important;
  box-sizing: border-box !important;
  padding-top: 24px !important;
  padding-left: clamp(16px, 2vw, 28px) !important;
  padding-right: clamp(16px, 2vw, 28px) !important;
  padding-bottom: 40px !important;
  --body-background-fill: #f8f1e7;
  --body-background-fill-dark: #f8f1e7;
  --background-fill-primary: #f8f1e7;
  --background-fill-primary-dark: #f8f1e7;
  --background-fill-secondary: #fffaf6;
  --background-fill-secondary-dark: #fffaf6;
  --block-background-fill: #fffaf6;
  --block-background-fill-dark: #fffaf6;
  --panel-background-fill: #fffaf6;
  --panel-background-fill-dark: #fffaf6;
  --body-text-color: var(--text-main);
  --body-text-color-subdued: var(--text-muted);
  --block-title-text-color: var(--text-strong);
  --block-label-text-color: var(--text-strong);
  --block-info-text-color: var(--text-muted);
  --accordion-text-color: var(--text-strong);
  --checkbox-label-text-color: var(--text-main);
  --checkbox-label-text-color-selected: var(--text-strong);
  --table-text-color: var(--text-main);
  --link-text-color: var(--accent-dark);
  --link-text-color-active: var(--accent-dark);
  --link-text-color-hover: var(--accent);
  --link-text-color-visited: var(--accent-dark);
  --button-primary-text-color: #fff7f0;
  --button-primary-text-color-hover: #fff7f0;
  --button-secondary-text-color: var(--text-strong);
  --button-secondary-text-color-hover: var(--text-strong);
  --input-placeholder-color: var(--text-soft);
}

h1, h2, h3, .hero-title, .metric-value {
  font-family: 'Space Grotesk', ui-sans-serif, sans-serif !important;
  color: var(--text-strong);
}

button,
input,
textarea,
select {
  font-family: 'IBM Plex Sans', ui-sans-serif, sans-serif !important;
}

.gradio-container p,
.gradio-container li,
.gradio-container label,
.gradio-container legend,
.gradio-container .prose,
.gradio-container .prose p,
.gradio-container .prose li,
.gradio-container .prose strong,
.gradio-container .prose th,
.gradio-container .prose td,
.gradio-container .prose code,
.gradio-container th,
.gradio-container td,
.gradio-container .helper,
.gradio-container .footer-note {
  color: var(--text-main);
}

.gradio-container .prose p,
.gradio-container .prose li,
.gradio-container .helper,
.gradio-container .footer-note {
  color: var(--text-muted);
}

.gradio-container label,
.gradio-container legend,
.gradio-container .prose strong,
.gradio-container th,
.gradio-container td,
.gradio-container h1,
.gradio-container h2,
.gradio-container h3 {
  color: var(--text-strong);
}

.gradio-container input,
.gradio-container textarea,
.gradio-container select {
  color: var(--text-main) !important;
  background: rgba(255, 251, 247, 0.96) !important;
}

.gradio-container ::placeholder {
  color: var(--text-soft) !important;
}

.gradio-container a,
.gradio-container a:visited {
  color: var(--accent-dark) !important;
}

.gradio-container a:hover {
  color: var(--accent) !important;
}

.hero-shell {
  position: relative;
  overflow: hidden;
  padding: 28px 30px;
  border: 1px solid rgba(147, 73, 43, 0.18);
  border-radius: 28px;
  background: linear-gradient(180deg, #fff6eb 0%, #fffaf5 100%);
  box-shadow: var(--card-shadow);
}

.hero-shell::after {
  content: none;
}

.hero-kicker {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 8px 14px;
  border-radius: 999px;
  background: rgba(255, 245, 233, 0.92);
  border: 1px solid rgba(155, 57, 42, 0.14);
  color: var(--accent-dark);
  font-size: 0.9rem;
  font-weight: 600;
  margin-bottom: 18px;
}

.hero-title {
  margin: 0;
  font-size: clamp(2.4rem, 5vw, 4.3rem);
  line-height: 0.96;
  letter-spacing: -0.06em;
  color: var(--text-strong);
}

.hero-copy {
  max-width: 760px;
  margin: 16px 0 0 0;
  color: var(--text-muted);
  font-size: 1.02rem;
  line-height: 1.7;
}

.metric-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 14px;
  margin-top: 24px;
}

.metric-card {
  padding: 18px 18px 16px;
  border-radius: 20px;
  background: #fffaf6;
  border: 1px solid rgba(123, 67, 42, 0.10);
}

.metric-label {
  color: var(--text-soft);
  font-size: 0.88rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  font-weight: 600;
}

.metric-value {
  margin-top: 10px;
  font-size: 1.45rem;
  font-weight: 700;
  color: var(--text-strong);
}

.metric-copy {
  margin-top: 8px;
  color: var(--text-muted);
  font-size: 0.92rem;
  line-height: 1.5;
}

.dashboard-card,
.panel {
  border: 1px solid var(--card-border);
  border-radius: 24px;
  background: var(--card-bg);
  box-shadow: var(--card-shadow);
}

.dashboard-card {
  padding: 18px;
}

.panel {
  padding: 22px 20px;
}

.legend-grid {
  display: grid;
  gap: 12px;
}

.legend-item {
  display: flex;
  gap: 12px;
  align-items: flex-start;
  padding: 14px 14px;
  border-radius: 18px;
  background: #fff8f2;
  border: 1px solid rgba(123, 67, 42, 0.08);
}

.legend-swatch {
  width: 14px;
  height: 14px;
  margin-top: 4px;
  border-radius: 999px;
  flex: 0 0 14px;
}

.legend-copy strong {
  display: block;
  margin-bottom: 4px;
  font-family: 'Space Grotesk', ui-sans-serif, sans-serif;
}

.legend-copy span {
  color: var(--text-muted);
  font-size: 0.93rem;
  line-height: 1.5;
}

.helper {
  color: var(--text-muted);
  font-size: 0.95rem;
  line-height: 1.6;
}

.guide-strip {
  padding: 14px 16px;
  border-radius: 18px;
  background: #f5e8d6;
  border: 1px solid rgba(155, 57, 42, 0.10);
  color: var(--text-main);
}

.tabs {
  gap: 12px;
}

button.primary,
.primary {
  background: linear-gradient(135deg, #b3442f, #7a261d) !important;
  border: none !important;
  box-shadow: 0 8px 16px rgba(122, 38, 29, 0.16) !important;
  color: #fff7f0 !important;
  font-weight: 600 !important;
}

button.secondary {
  border-color: rgba(123, 67, 42, 0.18) !important;
  color: var(--text-strong) !important;
  background: rgba(255, 250, 246, 0.92) !important;
}

button.primary *,
.primary *,
button.secondary *,
.gr-tab-nav button *,
.gradio-container [role="tab"] * {
  color: inherit !important;
}

.gr-tab-nav {
  gap: 10px !important;
  margin-bottom: 18px !important;
}

.gr-tab-nav button {
  border-radius: 999px !important;
  border: 1px solid rgba(123, 67, 42, 0.14) !important;
  background: rgba(255, 250, 246, 0.75) !important;
  color: var(--text-strong) !important;
  font-weight: 600 !important;
}

.gr-tab-nav button.selected {
  background: linear-gradient(135deg, rgba(196, 123, 40, 0.18), rgba(155, 57, 42, 0.16)) !important;
  border-color: rgba(155, 57, 42, 0.18) !important;
  color: var(--accent-dark) !important;
}

.gr-box,
.gr-form,
.gr-group,
.gr-panel,
.gr-accordion {
  border-color: rgba(123, 67, 42, 0.14) !important;
}

.gr-accordion summary,
.gr-accordion button,
.gradio-container [role="tab"],
.gradio-container [role="tablist"] button {
  color: var(--text-strong) !important;
}

.gradio-container .block-title,
.gradio-container .block-info,
.gradio-container .gr-form > label,
.gradio-container .gr-block > label,
.gradio-container [data-testid="block-label"] {
  color: var(--text-strong) !important;
}

.gradio-container .block-info,
.gradio-container .gr-markdown,
.gradio-container .gr-markdown p,
.gradio-container .gr-markdown li {
  color: var(--text-muted) !important;
}

.gradio-container .gr-markdown strong,
.gradio-container .gr-markdown th,
.gradio-container .gr-markdown [data-testid="block-label"] {
  color: var(--text-strong) !important;
}

.gradio-container table {
  color: var(--text-main) !important;
}

.gradio-container table th {
  color: var(--text-strong) !important;
}

.gradio-container table td {
  color: var(--text-main) !important;
}

.footer-note {
  margin-top: 12px;
  color: var(--text-muted);
  font-size: 0.9rem;
}

@media (max-width: 920px) {
  .metric-grid {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 600px) {
  .hero-shell { padding: 20px; border-radius: 20px; }
  .hero-title { font-size: 2.2rem; line-height: 1.05; }
  .hero-copy { font-size: 0.95rem; line-height: 1.5; }
  .metric-grid { gap: 8px; margin-top: 16px; }
  .metric-card { padding: 12px; border-radius: 12px; }
  .metric-value { font-size: 1.15rem; margin-top: 4px; }
  .metric-copy { display: none; }
  .dashboard-card, .panel { padding: 14px; border-radius: 18px; }
}
"""

HERO_HTML = """
<section class="hero-shell">
  <div class="hero-kicker">ATM Security Monitoring · YOLOv8 Detection</div>
  <h1 class="hero-title">Explore face-cover detection.</h1>
  <p class="hero-copy">
    This Space runs a custom YOLOv8 model trained for ATM monitoring. It separates covered and
    uncovered faces and supports three workflows: single photo review, short clip analysis, and
    live webcam inference streamed from the browser.
  </p>
  <div class="metric-grid">
    <div class="metric-card">
      <div class="metric-label">Modes</div>
      <div class="metric-value">Photo · Video · Live</div>
      <div class="metric-copy">Move from one-off review to continuous webcam monitoring without leaving the page.</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Classes</div>
      <div class="metric-value">2 Detection Labels</div>
      <div class="metric-copy">The model detects <strong>Face_Covered</strong> and <strong>Face_Uncovered</strong>.</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Runtime</div>
      <div class="metric-value">CPU Friendly</div>
      <div class="metric-copy">Images are fastest. For video and live mode, keep clips short and camera framing tight.</div>
    </div>
  </div>
</section>
"""

LEGEND_HTML = """
<div class="legend-grid">
  <div class="legend-item">
    <div class="legend-swatch" style="background:#9b392a;"></div>
    <div class="legend-copy">
      <strong>Face_Covered</strong>
      <span>Flags face coverings such as masks, scarves, helmet-like obstructions, or other concealment patterns.</span>
    </div>
  </div>
  <div class="legend-item">
    <div class="legend-swatch" style="background:#c58d2a;"></div>
    <div class="legend-copy">
      <strong>Face_Uncovered</strong>
      <span>Detects visible faces. Neither label establishes intent, identity, or whether a crime is occurring.</span>
    </div>
  </div>
</div>
"""


@lru_cache(maxsize=1)
def resolve_model_path() -> Path:
    if MODEL_PATH_ENV:
        explicit_model_path = Path(MODEL_PATH_ENV).expanduser()
        if explicit_model_path.is_file():
            return explicit_model_path
        raise FileNotFoundError(f"MODEL_PATH points to a missing file: '{explicit_model_path}'.")

    if HF_MODEL_REPO_ID:
        from huggingface_hub import hf_hub_download

        hub_cache_dir = APP_CACHE_DIR / "hf-model-cache"
        return Path(
            hf_hub_download(
                repo_id=HF_MODEL_REPO_ID,
                filename=HF_MODEL_FILENAME,
                cache_dir=str(hub_cache_dir),
            )
        )

    for candidate in DEFAULT_LOCAL_MODEL_CANDIDATES:
        if candidate.is_file():
            return candidate

    searched_paths = ", ".join(str(path) for path in DEFAULT_LOCAL_MODEL_CANDIDATES)
    raise FileNotFoundError(
        "Model weights were not found. Set MODEL_PATH, place 'weights/best.pt' in the Space repo, "
        f"or configure HF_MODEL_REPO_ID. Checked: {searched_paths}."
    )


@lru_cache(maxsize=1)
def load_model() -> YOLO:
    return YOLO(str(resolve_model_path()))


def build_detection_summary(result: Results, heading: str | None = None) -> str:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return f"{heading}\n\nNo detections found." if heading else "No detections found."

    counts: Counter[str] = Counter()
    best_confidence: dict[str, float] = {}
    names = result.names

    for class_id, confidence in zip(boxes.cls.tolist(), boxes.conf.tolist()):
        label = names.get(int(class_id), str(int(class_id)))
        counts[label] += 1
        best_confidence[label] = max(best_confidence.get(label, 0.0), float(confidence))

    total = sum(counts.values())
    summary_lines = []
    if heading:
        summary_lines.extend([heading, ""])
    summary_lines.extend(
        [
            f"Detected **{total}** object(s).",
            "",
            "| Class | Count | Best confidence |",
            "| --- | ---: | ---: |",
        ]
    )

    for label in sorted(counts):
        summary_lines.append(f"| {label} | {counts[label]} | {best_confidence[label]:.2f} |")

    return "\n".join(summary_lines)


def render_annotated_image(result: Results) -> PILImage.Image:
    annotated_bgr: np.ndarray = result.plot()
    annotated_rgb = annotated_bgr[:, :, ::-1]
    return PILImage.fromarray(annotated_rgb)


def run_image_inference(
    image: PILImage.Image | None,
    conf: float,
    iou: float,
    idle_message: str,
    active_heading: str,
) -> ImagePrediction:
    if image is None:
        return None, idle_message

    try:
        with MODEL_LOCK:
            result = load_model().predict(image, conf=conf, iou=iou, verbose=False)[0]
            annotated = render_annotated_image(result)
        return annotated, build_detection_summary(result, active_heading)
    except Exception as exc:
        return None, f"Error while running inference: {exc}"


def predict_photo(image: PILImage.Image | None, conf: float, iou: float) -> ImagePrediction:
    return run_image_inference(
        image=image,
        conf=conf,
        iou=iou,
        idle_message="Upload a photo or capture one from the webcam to begin.",
        active_heading="Photo review complete.",
    )


def predict_live_frame(image: PILImage.Image | None, conf: float, iou: float) -> ImagePrediction:
    return run_image_inference(
        image=image,
        conf=conf,
        iou=iou,
        idle_message="Start the webcam stream to begin live inference.",
        active_heading="Live inference is active.",
    )


def predict_video(video_path: str | None, conf: float, iou: float) -> VideoPrediction:
    if not video_path:
        return None, "Upload a short video clip first."

    output_root = None
    try:
        capture = cv2.VideoCapture(video_path)
        try:
            fps = capture.get(cv2.CAP_PROP_FPS)
            frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
            if not capture.isOpened() or fps <= 0 or frames <= 0:
                raise ValueError("The video could not be read. Try a valid MP4 clip.")
            if frames / fps > MAX_VIDEO_SECONDS:
                raise ValueError(f"Choose a clip no longer than {MAX_VIDEO_SECONDS} seconds.")
        finally:
            capture.release()
        video_cache = APP_CACHE_DIR / "videos"
        video_cache.mkdir(exist_ok=True)
        for old in video_cache.glob("atm-space-*"):
            if old.is_dir() and time.time() - old.stat().st_mtime > 3600:
                shutil.rmtree(old, ignore_errors=True)
        output_root = Path(tempfile.mkdtemp(prefix="atm-space-", dir=video_cache))
        output_dir = output_root / "prediction"

        with MODEL_LOCK:
            results = load_model().predict(
                source=video_path,
                conf=conf,
                iou=iou,
                project=str(output_root),
                name="prediction",
                save=True,
                exist_ok=True,
                verbose=False,
                stream=True,
            )
            for _ in results:
                pass

        candidates = sorted(
            path for path in output_dir.iterdir() if path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}
        )
        if not candidates:
            raise RuntimeError("Ultralytics did not produce an annotated output video.")

        return (
            str(candidates[0]),
            "Video review complete.\n\nAnnotated output generated. Short MP4 clips work best on CPU Spaces.",
        )
    except Exception as exc:
        if output_root is not None:
            shutil.rmtree(output_root, ignore_errors=True)
        return None, f"Error while running video inference: {exc}"


with gr.Blocks(title=TITLE, fill_width=True, delete_cache=(3600, 3600)) as demo:
    gr.HTML(HERO_HTML)
    gr.Markdown(
        "**Research demo:** face coverings do not establish theft or intent. Uploaded media and webcam frames are sent to this server for processing; use only media you have permission to share."
    )

    with gr.Row():
        with gr.Column(scale=2, elem_classes="dashboard-card"):
            gr.Markdown(
                """
                ### Detection Controls
                Tune the thresholds once and reuse them across all three modes.
                Lower confidence catches more detections but can increase false positives.
                """
            )
            with gr.Accordion("Threshold Settings", open=True):
                conf = gr.Slider(
                    minimum=0.10,
                    maximum=0.90,
                    value=0.25,
                    step=0.05,
                    label="Confidence threshold",
                )
                iou = gr.Slider(
                    minimum=0.10,
                    maximum=0.90,
                    value=0.45,
                    step=0.05,
                    label="IoU threshold",
                )
            gr.HTML(
                """
                <div class="guide-strip">
                  Use <strong>Photo</strong> for one frame, <strong>Short Video</strong> for clip review,
                  and <strong>Live Inference</strong> for continuous webcam processing in the browser.
                </div>
                """
            )

        with gr.Column(scale=1, elem_classes="panel"):
            gr.Markdown("### Detection Legend")
            gr.HTML(LEGEND_HTML)
            gr.Markdown(
                """
                <div class="footer-note">
                  For the cleanest results, keep the camera pointed toward the face region and avoid
                  rapid scene changes during live mode.
                </div>
                """
            )

    with gr.Tabs():
        with gr.Tab("Photo"):
            with gr.Row():
                with gr.Column(elem_classes="dashboard-card"):
                    gr.Markdown("### Input")
                    photo_input = gr.Image(
                        type="pil",
                        sources=["upload", "webcam"],
                        label="Photo input",
                        height=420,
                    )
                    photo_run = gr.Button("Run photo inference", variant="primary")
                with gr.Column(elem_classes="dashboard-card"):
                    gr.Markdown("### Result")
                    photo_output = gr.Image(label="Annotated photo", height=420)
                    photo_summary = gr.Markdown("Upload a photo or capture one from the webcam to begin.")
                    gr.ClearButton([photo_input, photo_output, photo_summary], value="Clear photo and result")

            photo_run.click(
                fn=predict_photo,
                inputs=[photo_input, conf, iou],
                outputs=[photo_output, photo_summary],
                show_progress="minimal",
                concurrency_limit=1,
                concurrency_id="yolo",
            )

        with gr.Tab("Short Video"):
            with gr.Row():
                with gr.Column(elem_classes="dashboard-card"):
                    gr.Markdown("### Input")
                    video_input = gr.Video(
                        sources=["upload", "webcam"],
                        label="Short video input",
                        height=420,
                    )
                    gr.Markdown(
                        """
                        <div class="helper">
                          Use clips up to 30 seconds and 25 MB. MP4 clips with a stable camera
                          angle work best.
                        </div>
                        """
                    )
                    video_run = gr.Button("Run video inference", variant="primary")
                with gr.Column(elem_classes="dashboard-card"):
                    gr.Markdown("### Result")
                    video_output = gr.Video(label="Annotated video", height=420, autoplay=True)
                    video_summary = gr.Markdown("Upload a short video clip to begin.")

            video_run.click(
                fn=predict_video,
                inputs=[video_input, conf, iou],
                outputs=[video_output, video_summary],
                show_progress="minimal",
                concurrency_limit=1,
                concurrency_id="yolo",
            )

        with gr.Tab("Live Inference"):
            with gr.Row():
                with gr.Column(elem_classes="dashboard-card"):
                    gr.Markdown("### Live Camera Feed")
                    live_input = gr.Image(
                        type="pil",
                        sources=["webcam"],
                        streaming=True,
                        label="Browser webcam stream",
                        height=420,
                    )
                    gr.Markdown(
                        """
                        <div class="helper">
                          Allow camera access in the browser, then start the webcam stream. Frames are
                          processed continuously while the stream is active.
                        </div>
                        """
                    )
                with gr.Column(elem_classes="dashboard-card"):
                    gr.Markdown("### Live Detection")
                    live_output = gr.Image(label="Annotated live frame", height=420)
                    live_summary = gr.Markdown("Start the webcam stream to begin live inference.")

            live_input.stream(
                fn=predict_live_frame,
                inputs=[live_input, conf, iou],
                outputs=[live_output, live_summary],
                show_progress="hidden",
                queue=True,
                concurrency_limit=1,
                concurrency_id="yolo",
                stream_every=0.35,
            )

demo.queue(default_concurrency_limit=1, max_size=16)


if __name__ == "__main__":
    demo.launch(theme=THEME, css=CSS, max_file_size="25mb")
