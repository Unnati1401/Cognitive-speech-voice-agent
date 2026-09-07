"""Hugging Face Spaces entry point.

Spaces runs this file (app.py at the repo root). It makes the src/ package and
the scripts/ UI importable, then launches the Gradio app defined in scripts/app.py.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))       # the cognitive_speech_agent package
sys.path.insert(0, str(ROOT / "scripts"))   # the Gradio UI (scripts/app.py)

from app import build_ui  # noqa: E402  (scripts/app.py)

demo = build_ui()

if __name__ == "__main__":
    demo.launch()