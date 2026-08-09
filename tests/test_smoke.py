"""Smoke tests for Phase 0: package imports and config loads."""
from cognitive_speech_agent import __version__
from cognitive_speech_agent.config import load_config


def test_version():
    assert __version__ == "0.1.0"


def test_config_loads():
    cfg = load_config()
    assert cfg["task"]["type"] in {"picture_description", "open_prompt"}
    assert cfg["scoring"]["split"] == "speaker_independent"  # the non-negotiable
    assert cfg["report"]["grounded_only"] is True
