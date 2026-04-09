from __future__ import annotations

import os
import tempfile
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Final, TypeAlias

ROOT: Final[Path] = Path(__file__).resolve().parent
APP_CACHE_DIR = ROOT / ".runtime"
APP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(APP_CACHE_DIR))

import gradio as gr
import numpy as np
from PIL import Image as PILImage
from ultralytics import YOLO
from ultralytics.engine.results import Results

ImagePrediction: TypeAlias = tuple[PILImage.Image | None, str]
VideoPrediction: TypeAlias = tuple[str | None, str]

DEFAULT_MODEL_PATH: Final[Path] = ROOT / "weights" / "best.pt"
MODEL_PATH_ENV = os.getenv("MODEL_PATH")
MODEL_PATH: Final[Path] = Path(MODEL_PATH_ENV) if MODEL_PATH_ENV else DEFAULT_MODEL_PATH
TITLE: Final[str] = "ATM Theft Detection"

THEME = gr.themes.Soft(
    primary_hue="amber",
    secondary_hue="rose",
    neutral_hue="stone",
)

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

:root {
  --page-bg:
    radial-gradient(circle at top left, rgba(255, 196, 125, 0.24), transparent 30%),
    radial-gradient(circle at top right, rgba(164, 52, 58, 0.18), transparent 26%),
    linear-gradient(180deg, #f8f1e7 0%, #f2e4d3 54%, #ead5c1 100%);
  --card-bg: rgba(255, 251, 246, 0.82);
  --card-border: rgba(123, 67, 42, 0.14);
  --card-shadow: 0 22px 50px rgba(98, 53, 32, 0.10);
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
#root,
.gradio-container {
  background: var(--page-bg);
  color: var(--text-main);
  font-family: 'IBM Plex Sans', ui-sans-serif, sans-serif;
  text-rendering: optimizeLegibility;
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
.gradio-container span,
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

.hero-shell {
  position: relative;
  overflow: hidden;
  padding: 28px 30px;
  border: 1px solid rgba(147, 73, 43, 0.18);
  border-radius: 28px;
  background:
    linear-gradient(135deg, rgba(255, 244, 226, 0.94), rgba(255, 248, 240, 0.88)),
    linear-gradient(120deg, rgba(197, 141, 42, 0.08), rgba(155, 57, 42, 0.08));
  box-shadow: var(--card-shadow);
}

.hero-shell::after {
  content: "";
  position: absolute;
  inset: auto -30px -40px auto;
  width: 220px;
  height: 220px;
  border-radius: 999px;
  background: radial-gradient(circle, rgba(155, 57, 42, 0.18), transparent 68%);
  pointer-events: none;
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
  background: rgba(255, 252, 248, 0.72);
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
  backdrop-filter: blur(14px);
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
  background: rgba(255, 247, 239, 0.8);
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
  background: linear-gradient(90deg, rgba(197, 141, 42, 0.15), rgba(155, 57, 42, 0.08));
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
  box-shadow: 0 14px 28px rgba(122, 38, 29, 0.24) !important;
  color: #fff7f0 !important;
  font-weight: 600 !important;
}

button.secondary {
  border-color: rgba(123, 67, 42, 0.18) !important;
  color: var(--text-strong) !important;
  background: rgba(255, 250, 246, 0.92) !important;
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
"""

HERO_HTML = """
<section class="hero-shell">
  <div class="hero-kicker">ATM Security Monitoring · YOLOv8 Detection</div>
  <h1 class="hero-title">Read faces, flag risk, keep the feed under watch.</h1>
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
      <div class="metric-value">2 Security Labels</div>
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
      <span>Represents visible, non-concealed faces and non-threatening observation states in the current model design.</span>
    </div>
  </div>
</div>
"""


@lru_cache(maxsize=1)
def load_model() -> YOLO:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model weights not found at '{MODEL_PATH}'. Expected 'weights/best.pt' in the repo."
        )
    return YOLO(str(MODEL_PATH))


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

    try:
        output_root = Path(tempfile.mkdtemp(prefix="atm-space-"))
        output_dir = output_root / "prediction"

        load_model().predict(
            source=video_path,
            conf=conf,
            iou=iou,
            project=str(output_root),
            name="prediction",
            save=True,
            exist_ok=True,
            verbose=False,
        )

        candidates = sorted(path for path in output_dir.iterdir() if path.is_file())
        if not candidates:
            raise RuntimeError("Ultralytics did not produce an annotated output video.")

        return (
            str(candidates[0]),
            "Video review complete.\n\nAnnotated output generated. Short MP4 clips work best on CPU Spaces.",
        )
    except Exception as exc:
        return None, f"Error while running video inference: {exc}"


with gr.Blocks(title=TITLE, fill_width=True) as demo:
    gr.HTML(HERO_HTML)

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
                    gr.ClearButton(
                        components=[photo_input],
                        value="Clear photo",
                    )
                with gr.Column(elem_classes="dashboard-card"):
                    gr.Markdown("### Result")
                    photo_output = gr.Image(label="Annotated photo", height=420)
                    photo_summary = gr.Markdown("Upload a photo or capture one from the webcam to begin.")

            photo_run.click(
                fn=predict_photo,
                inputs=[photo_input, conf, iou],
                outputs=[photo_output, photo_summary],
                show_progress="minimal",
                concurrency_limit=1,
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
                          Keep clips short for the best turnaround on CPU. MP4 clips with a stable camera
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
                queue=False,
                concurrency_limit=1,
                stream_every=0.35,
            )

demo.queue(default_concurrency_limit=1)


if __name__ == "__main__":
    demo.launch(theme=THEME, css=CSS)
