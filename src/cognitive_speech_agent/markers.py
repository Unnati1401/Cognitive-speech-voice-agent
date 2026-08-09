"""Tool 3 - Marker extraction: the interpretable cognitive-linguistic features.

Phase 1 work. Four families feed both the report and (optionally) the scorer:
    timing      - pauses, speech rate, silence ratio, fillers, false starts
    vocabulary  - type-token ratio, moving-average TTR, vague-word ratio
    grammar     - mean length of utterance, syntactic complexity/completeness
    coherence   - topic drift, repetition, information-unit coverage
"""
from __future__ import annotations

from .transcribe import Transcript


def timing_markers(transcript: Transcript, audio_path: str) -> dict[str, float]:
    """Pause count/duration, speech rate, silence ratio, fillers, false starts.
    Sourced from Whisper word timings + openSMILE (eGeMAPS). TODO (Phase 1)."""
    raise NotImplementedError("Implemented in Phase 1.")


def vocabulary_markers(transcript: Transcript) -> dict[str, float]:
    """TTR, moving-average TTR, vague-word ratio, word-frequency profile.
    TODO (Phase 1)."""
    raise NotImplementedError("Implemented in Phase 1.")


def grammar_markers(transcript: Transcript) -> dict[str, float]:
    """Mean length of utterance, syntactic depth, grammatical completeness (spaCy).
    TODO (Phase 1)."""
    raise NotImplementedError("Implemented in Phase 1.")


def coherence_markers(transcript: Transcript, task_type: str) -> dict[str, float]:
    """Sentence-to-sentence topic drift, repetition, information-unit coverage.
    TODO (Phase 1)."""
    raise NotImplementedError("Implemented in Phase 1.")


def extract_markers(
    transcript: Transcript, audio_path: str, task_type: str
) -> dict[str, float]:
    """Run all enabled marker families and return one flat feature dict."""
    raise NotImplementedError("Implemented in Phase 1.")
