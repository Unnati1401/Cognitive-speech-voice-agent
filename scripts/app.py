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

from pathlib import Path

import gradio as gr
import pandas as pd

from cognitive_speech_agent.config import REPO_ROOT, load_config
from cognitive_speech_agent import orchestrator as O

CFG = load_config()
DISCLAIMER = CFG.get("safety", {}).get("disclaimer", "Research demo. Not a medical device.")

SAMPLES_DIR = REPO_ROOT / "assets" / "samples"
SAMPLE_FILES = sorted(str(p) for p in SAMPLES_DIR.glob("*.wav")) if SAMPLES_DIR.exists() else []


def _markers_table(markers: dict) -> pd.DataFrame:
    rows = [(k, round(float(v), 3)) for k, v in sorted(markers.items())]
    return pd.DataFrame(rows, columns=["marker", "value"])


def analyze(audio_path, model_size, diarize):
    """Run the agent on one clip and return (status, transcript, markers, report)."""
    if not audio_path:
        return "Please pick a sample, record, or upload an audio clip first.", "", pd.DataFrame(), ""

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
        gr.Markdown("# Cognitive Speech Screening - research demo")
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