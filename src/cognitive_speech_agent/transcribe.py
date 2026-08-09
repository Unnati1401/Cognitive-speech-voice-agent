"""Tool 2 - Transcription: patient audio -> text WITH word-level timestamps.

Phase 1 work. Word timings are required for pause/timing markers.
Uses faster-whisper (large-v3, float16 on CUDA).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Word:
    text: str
    start: float          # seconds
    end: float            # seconds


@dataclass
class Transcript:
    text: str
    words: list[Word] = field(default_factory=list)
    duration: float = 0.0


def transcribe(audio_path: str, model_size: str = "large-v3") -> Transcript:
    """Transcribe patient audio and return text + word timings.

    TODO (Phase 1):
        - run faster-whisper with word_timestamps=True
        - flatten segments into Word objects
        - populate Transcript.duration
    """
    raise NotImplementedError("Implemented in Phase 1.")
