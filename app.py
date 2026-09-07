"""Hugging Face Spaces entry point.

Spaces runs this file (app.py at the repo root). It makes the src/ package and
the scripts/ UI importable, then launches the Gradio app defined in scripts/app.py.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))       # the cognitive_speech_agent package
sys.path.insert(0, str(ROOT / "scripts"))   # the Gradio UI (scripts/app.py)

# --- defensive workaround for the gradio_client bool-schema crash -----------
# Harmless with the pinned 4.44.1 stack; kept as a belt-and-suspenders guard.
try:
    import gradio_client.utils as _gcu

    _orig_json = _gcu._json_schema_to_python_type

    def _safe_json(schema, defs=None):
        if isinstance(schema, bool):
            return "Any"
        return _orig_json(schema, defs)

    _gcu._json_schema_to_python_type = _safe_json

    _orig_get_type = _gcu.get_type

    def _safe_get_type(schema):
        if isinstance(schema, bool):
            return "Any"
        return _orig_get_type(schema)

    _gcu.get_type = _safe_get_type
except Exception:  # noqa: BLE001 - never let the patch itself break startup
    pass
# ---------------------------------------------------------------------------

from app import build_ui  # noqa: E402  (scripts/app.py)

demo = build_ui()

if __name__ == "__main__":
    demo.launch()