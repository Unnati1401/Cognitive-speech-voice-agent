"""The agent orchestrator: wires the five tools into a stateful session loop.

Phase 4 work. This is what makes it an *agent* rather than a script: it holds
session state, sequences the tools, and handles conditions (e.g. audio too
short -> ask to re-record) before assembling the final report.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class SessionState(str, Enum):
    AWAITING_AUDIO = "awaiting_audio"
    DIARIZING = "diarizing"
    TRANSCRIBING = "transcribing"
    EXTRACTING = "extracting"
    SCORING = "scoring"
    REPORTING = "reporting"
    DONE = "done"
    NEEDS_RERECORD = "needs_rerecord"


@dataclass
class Session:
    audio_path: str
    state: SessionState = SessionState.AWAITING_AUDIO
    artifacts: dict = field(default_factory=dict)   # transcript, markers, score, report


def run_session(audio_path: str, config: dict) -> Session:
    """Run the full pipeline end-to-end for one audio file.

    TODO (Phase 4):
        diarize -> select speaker -> check min duration -> transcribe ->
        extract markers -> score -> build fact sheet -> render + validate report.
    Can be a plain state machine or a LangGraph tool-calling loop.
    """
    raise NotImplementedError("Implemented in Phase 4.")
