"""Tool 1 - Diarization: separate the patient's voice from the clinician's.

Uses pyannote.audio (speaker-diarization-3.1). Requires a HuggingFace token with
the model terms accepted (export HF_TOKEN=...). Heavy deps imported lazily.

For a single-speaker recording (e.g. a patient describing a picture alone, or your
own test clip) you can skip diarization entirely; see scripts/run_markers.py.
"""
from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass

from .transcribe import Transcript, Word


@dataclass
class SpeakerSegment:
    start: float          # seconds
    end: float            # seconds
    speaker: str          # e.g. "SPEAKER_00"

    def contains(self, t: float) -> bool:
        return self.start <= t <= self.end


def _load_pipeline(token: str):
    """Load the pyannote pipeline, tolerating both old and new arg names.

    Newer pyannote.audio uses token=..., older versions use use_auth_token=...
    """
    from pyannote.audio import Pipeline  # lazy

    name = "pyannote/speaker-diarization-3.1"
    try:
        return Pipeline.from_pretrained(name, token=token)          # new API
    except TypeError:
        return Pipeline.from_pretrained(name, use_auth_token=token)  # old API


def diarize(audio_path: str, hf_token: str | None = None) -> list[SpeakerSegment]:
    """Return speaker-labeled time segments for an audio file."""
    token = hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        raise RuntimeError(
            "No HuggingFace token found. Accept terms for "
            "pyannote/speaker-diarization-3.1 and set HF_TOKEN in your environment."
        )

    pipeline = _load_pipeline(token)

    # Move to GPU if available (pyannote runs in pure PyTorch, supports cuda/mps).
    try:
        import torch

        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            pipeline.to(torch.device("mps"))
    except Exception:  # noqa: BLE001
        pass

    diarization = pipeline(audio_path)
    return [
        SpeakerSegment(start=float(turn.start), end=float(turn.end), speaker=str(spk))
        for turn, _, spk in diarization.itertracks(yield_label=True)
    ]


def select_target_speaker(
    segments: list[SpeakerSegment], strategy: str = "most_speech"
) -> str:
    """Pick which diarized speaker is the patient.

    Default heuristic: in an open-ended task the patient talks the most.
    """
    if not segments:
        raise ValueError("No speaker segments to select from.")
    totals: dict[str, float] = defaultdict(float)
    for s in segments:
        totals[s.speaker] += s.end - s.start
    return max(totals, key=totals.get)


def filter_words_by_speaker(
    words: list[Word], segments: list[SpeakerSegment], speaker: str
) -> list[Word]:
    """Keep only words whose midpoint falls inside the target speaker's segments."""
    target = [s for s in segments if s.speaker == speaker]
    kept: list[Word] = []
    for w in words:
        mid = (w.start + w.end) / 2.0
        if any(s.contains(mid) for s in target):
            kept.append(w)
    return kept


def patient_transcript(
    transcript: Transcript, segments: list[SpeakerSegment], speaker: str
) -> Transcript:
    """Return a new Transcript containing only the target speaker's words."""
    kept = filter_words_by_speaker(transcript.words, segments, speaker)
    text = " ".join(w.text for w in kept)
    return Transcript(text=text, words=kept, duration=transcript.duration)