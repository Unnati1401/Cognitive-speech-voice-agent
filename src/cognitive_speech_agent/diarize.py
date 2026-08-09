"""Tool 1 - Diarization: separate the patient's voice from the clinician's.

Phase 1 work. Uses pyannote.audio (speaker-diarization-3.1).
Requires a HuggingFace token with the model terms accepted (see README).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SpeakerSegment:
    start: float          # seconds
    end: float            # seconds
    speaker: str          # e.g. "SPEAKER_00"


def diarize(audio_path: str, hf_token: str | None = None) -> list[SpeakerSegment]:
    """Return speaker-labeled time segments for an audio file.

    TODO (Phase 1):
        - load pyannote pipeline
        - run on 16 kHz mono audio
        - return list[SpeakerSegment]
    """
    raise NotImplementedError("Implemented in Phase 1.")


def select_target_speaker(
    segments: list[SpeakerSegment], strategy: str = "most_speech"
) -> str:
    """Pick which diarized speaker is the patient.

    Default heuristic: the patient talks most during an open-ended task.
    TODO (Phase 1): sum durations per speaker, return the max.
    """
    raise NotImplementedError("Implemented in Phase 1.")
