"""Tool 2 - Transcription: patient audio -> text WITH word-level timestamps.

Word timings are required for pause/timing markers.
Uses faster-whisper (CTranslate2). On Apple Silicon / Mac this runs on CPU
(set compute_type=int8); on an NVIDIA GPU use device=cuda, compute_type=float16.

Heavy deps are imported lazily inside functions so importing this module (and the
marker code that depends on it) never requires faster-whisper to be installed.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Word:
    text: str
    start: float          # seconds
    end: float            # seconds

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class Transcript:
    text: str
    words: list[Word] = field(default_factory=list)
    duration: float = 0.0     # total audio duration in seconds


def _pick_device(device: str) -> str:
    """faster-whisper (CTranslate2) supports cpu / cuda. Mac -> cpu."""
    if device != "auto":
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:  # noqa: BLE001
        pass
    return "cpu"


def transcribe(
    audio_path: str,
    model_size: str = "large-v3",
    compute_type: str = "int8",
    device: str = "auto",
    language: str = "en",
) -> Transcript:
    """Transcribe audio and return text + per-word timestamps.

    Returns a Transcript with words (each with start/end) and total duration.
    """
    from faster_whisper import WhisperModel  # lazy

    dev = _pick_device(device)
    # int8 is safe on CPU; float16 needs CUDA. Guard the mismatch.
    if dev == "cpu" and compute_type == "float16":
        compute_type = "int8"

    model = WhisperModel(model_size, device=dev, compute_type=compute_type)
    segments, info = model.transcribe(
        audio_path, language=language, word_timestamps=True, vad_filter=False
    )

    words: list[Word] = []
    text_parts: list[str] = []
    for seg in segments:
        text_parts.append(seg.text.strip())
        for w in seg.words or []:
            tok = w.word.strip()
            if tok:
                words.append(Word(text=tok, start=float(w.start), end=float(w.end)))

    duration = float(getattr(info, "duration", 0.0)) or (words[-1].end if words else 0.0)
    return Transcript(text=" ".join(p for p in text_parts if p), words=words, duration=duration)
