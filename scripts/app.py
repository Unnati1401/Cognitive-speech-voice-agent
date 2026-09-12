#!/usr/bin/env python
"""Phase 5: a small Gradio web UI over the agent.

Record, upload, OR click a bundled sample -> see the transcript, the marker
table, and the grounded screening report. This just calls run_session()
(Phase 4) and lays the result out in the browser.

    uv run python scripts/app.py         # then open the printed local URL

Needs a trained scorer (Phase 2) and an LLM key in .env (auto-loaded).
Generate the clickable samples once with: uv run python scripts/make_samples.py
"""
from __future__ import annotations

import time
from pathlib import Path

import gradio as gr
import pandas as pd

# HF ZeroGPU needs a function decorated with @spaces.GPU. Provide a no-op fallback
# so the app still runs locally where the `spaces` package isn't installed.
try:
    import spaces
except Exception:  # noqa: BLE001
    class _NoSpaces:
        @staticmethod
        def GPU(*args, **kwargs):
            if args and callable(args[0]) and not kwargs:
                return args[0]                      # used as @spaces.GPU

            def _wrap(fn):
                return fn
            return _wrap                            # used as @spaces.GPU(...)

    spaces = _NoSpaces()

from cognitive_speech_agent.config import REPO_ROOT, load_config
from cognitive_speech_agent import orchestrator as O

CFG = load_config()
DISCLAIMER = CFG.get("safety", {}).get("disclaimer", "Research demo. Not a medical device.")

SAMPLES_DIR = REPO_ROOT / "assets" / "samples"
SAMPLE_FILES = sorted(str(p) for p in SAMPLES_DIR.glob("*.wav")) if SAMPLES_DIR.exists() else []


def _bootstrap_model():
    """Retrain the demo scorer at startup so it matches THIS environment's
    scikit-learn. A model pickled elsewhere fails with version-mismatch errors
    (e.g. 'LogisticRegression has no attribute multi_class')."""
    import copy

    models_dir = REPO_ROOT / "models"
    if (models_dir / "scorer.json").exists() or (models_dir / "scorer.joblib").exists():
        return  # a real/shipped model is present -> never overwrite it

    feats = REPO_ROOT / "assets" / "demo_features.csv"
    if not feats.exists():
        return
    try:
        from cognitive_speech_agent import scoring

        cfg = copy.deepcopy(CFG)
        cfg.setdefault("scoring", {})["model"] = "logistic"   # light + deterministic
        scoring.train(str(feats), cfg)
        print(f"Demo scorer retrained from {feats}")
    except Exception as e:  # noqa: BLE001
        print(f"Model bootstrap skipped: {e}")


_bootstrap_model()

# --- simple global rate limit (protects the OpenAI bill on a public demo) ---
MAX_ANALYSES_PER_HOUR = 40
_CALL_TIMES: list[float] = []


def _rate_limited() -> bool:
    now = time.time()
    _CALL_TIMES[:] = [t for t in _CALL_TIMES if now - t < 3600]
    if len(_CALL_TIMES) >= MAX_ANALYSES_PER_HOUR:
        return True
    _CALL_TIMES.append(now)
    return False


def _markers_table(markers: dict) -> pd.DataFrame:
    rows = [(k, round(float(v), 3)) for k, v in sorted(markers.items())]
    return pd.DataFrame(rows, columns=["marker", "value"])


@spaces.GPU(duration=120)
def analyze(audio_path, model_size, diarize):
    """Run the agent on one clip and return (status, transcript, markers, report)."""
    if not audio_path:
        return "Please pick a sample, record, or upload an audio clip first.", "", pd.DataFrame(), ""

    if _rate_limited():
        return ("Demo is at its hourly limit — please try again later.", "", pd.DataFrame(), "")

    cfg = load_config()
    cfg["transcription"]["model_size"] = model_size
    try:
        sess = O.run_session(audio_path, cfg, diarize=diarize)
    except Exception as e:  # noqa: BLE001 - surface errors in the UI, don't crash it
        return f"Error: {type(e).__name__}: {e}", "", pd.DataFrame(), ""

    status = f"State: {sess.state.value} — {sess.message}"
    transcript = sess.transcript.text if sess.transcript else ""
    table = _markers_table(sess.markers) if sess.markers else pd.DataFrame()
    report = sess.report or "_No report generated — see status above._"
    return status, transcript, table, report


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Cognitive Speech Screening (research demo)") as demo:
        gr.Markdown("# Cognitive Speech Screening — research demo")
        gr.Markdown(f"> {DISCLAIMER}")
        with gr.Row():
            with gr.Column():
                audio = gr.Audio(sources=["upload", "microphone"], type="filepath",
                                 label="Picture-description clip")
                if SAMPLE_FILES:
                    gr.Examples(
                        examples=[[f] for f in SAMPLE_FILES],
                        inputs=[audio],
                        label="Or click a sample clip (no recording needed)",
                    )
                model = gr.Dropdown(["tiny", "base", "small", "medium", "large-v3"],
                                    value="small",
                                    label="Whisper model (smaller = faster on Mac)")
                diarize = gr.Checkbox(value=False, label="Two speakers (clinician + patient)")
                go = gr.Button("Analyze", variant="primary")
                status = gr.Textbox(label="Status", interactive=False)
            with gr.Column():
                report = gr.Markdown(label="Screening report")
        with gr.Accordion("Details (transcript + markers)", open=False):
            transcript = gr.Textbox(label="Transcript", lines=4, interactive=False)
            markers = gr.Dataframe(label="Markers", interactive=False)

        go.click(analyze, inputs=[audio, model, diarize],
                 outputs=[status, transcript, markers, report])
    return demo


if __name__ == "__main__":
    build_ui().launch()