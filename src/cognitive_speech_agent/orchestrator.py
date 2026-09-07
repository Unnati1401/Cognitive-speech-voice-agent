"""The agent orchestrator: wires the five tools into one stateful session.

This is what makes the project an *agent* rather than five scripts. `run_session`
walks a small state machine (transcribe -> [diarize] -> markers -> score -> report),
holds the artifacts produced at each step, and makes a decision: if the sample is
too short or empty it stops and asks for a re-record instead of pushing bad data
downstream.

All heavy steps are injectable (transcribe_fn / score_fn / complete_fn) so the
flow can be unit-tested offline without Whisper, a trained model, or an API key.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from . import diarize as D
from . import markers as M
from . import report as R
from . import scoring
from .transcribe import Transcript, transcribe


class SessionState(str, Enum):
    TRANSCRIBING = "transcribing"
    DIARIZING = "diarizing"
    EXTRACTING = "extracting"
    SCORING = "scoring"
    REPORTING = "reporting"
    DONE = "done"
    NEEDS_RERECORD = "needs_rerecord"


@dataclass
class Session:
    """Everything produced during one run, plus where the agent ended up."""
    audio_path: str
    state: SessionState = SessionState.TRANSCRIBING
    message: str = ""
    transcript: Transcript | None = None
    markers: dict = field(default_factory=dict)
    score: "scoring.ScoreResult | None" = None
    report: str = ""

    @property
    def ok(self) -> bool:
        return self.state == SessionState.DONE


# --- default step implementations (used unless injected) ---------------------

def _default_transcribe(config: dict):
    t = config["transcription"]
    # Force CPU: on HF ZeroGPU torch reports CUDA available but the CUDA math
    # libs (libcublas) aren't loadable for CTranslate2. This app is CPU-only.
    return lambda path: transcribe(
        path, model_size=t["model_size"], compute_type=t["compute_type"],
        language=t["language"], device="cpu",
    )


def _default_score(config: dict):
    models_dir = Path(config.get("paths", {}).get("models", "models"))
    model_p, norms_p = models_dir / "scorer.joblib", models_dir / "norms.json"
    if not model_p.exists():
        raise RuntimeError(
            "No trained scorer found (models/scorer.joblib). Run Phase 2 first.")
    return lambda feats: scoring.score(feats, str(model_p), str(norms_p))


# --- the agent loop ----------------------------------------------------------

def run_session(
    audio_path: str,
    config: dict,
    *,
    diarize: bool = False,
    transcribe_fn=None,
    score_fn=None,
    complete_fn=None,
) -> Session:
    """Run the full pipeline for one recording and return a Session.

    Stops early with state=NEEDS_RERECORD if the sample is empty or too short.
    """
    task = config["task"]["type"]
    min_dur = float(config["task"].get("min_duration_sec", 30))
    sess = Session(audio_path=audio_path, state=SessionState.TRANSCRIBING)

    # 1. transcribe
    tr = (transcribe_fn or _default_transcribe(config))(audio_path)

    # 2. optional diarization (two-speaker recordings)
    if diarize:
        sess.state = SessionState.DIARIZING
        segs = D.diarize(audio_path)
        tr = D.patient_transcript(tr, segs, D.select_target_speaker(segs))
    sess.transcript = tr

    # decision point: is there enough usable speech?
    if not tr.words or tr.duration < min_dur:
        sess.state = SessionState.NEEDS_RERECORD
        sess.message = (
            f"Sample too short or empty ({tr.duration:.0f}s, {len(tr.words)} words; "
            f"need >= {min_dur:.0f}s). Please re-record a longer description.")
        return sess

    # 3. markers
    sess.state = SessionState.EXTRACTING
    sess.markers = M.extract_markers(
        tr, task_type=task, families=config["markers"],
        embedder_name=config["markers"]["coherence_embedder"])

    # 4. score
    sess.state = SessionState.SCORING
    sess.score = (score_fn or _default_score(config))(sess.markers)

    # 5. grounded report
    sess.state = SessionState.REPORTING
    fs = R.build_fact_sheet(sess.markers, sess.score, tr, config)
    sess.report = R.render_report(fs, config, complete_fn=complete_fn)

    sess.state = SessionState.DONE
    sess.message = "Report generated."
    return sess