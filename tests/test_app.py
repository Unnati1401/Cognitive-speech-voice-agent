"""Phase 5 test: the UI builds and analyze() handles empty input.

Skips cleanly if gradio isn't installed (e.g. a minimal CI env).
"""
import sys
from pathlib import Path

import pytest

pytest.importorskip("gradio")

# make scripts/app.py importable regardless of where pytest is invoked
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import app  # noqa: E402


def test_analyze_handles_missing_audio():
    status, transcript, table, report = app.analyze(None, "small", False)
    assert "upload" in status.lower() or "record" in status.lower()
    assert transcript == "" and report == ""


def test_ui_builds():
    assert app.build_ui() is not None