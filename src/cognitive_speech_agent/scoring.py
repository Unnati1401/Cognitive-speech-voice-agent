"""Tool 4 - Scoring: markers -> risk score, plus norm-based flags.

Phase 2 work. Hybrid approach (Path E): handcrafted markers (+ optional
embeddings) -> XGBoost / logistic regression, trained on ADReSS labels.

CRITICAL: split by SPEAKER, never by utterance, or metrics are fake.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ScoreResult:
    risk_probability: float                       # 0..1
    label: str                                    # "elevated markers" | "within typical range"
    flagged_markers: dict[str, float] = field(default_factory=dict)   # marker -> percentile
    percentiles: dict[str, float] = field(default_factory=dict)


def train(feature_table_path: str, config: dict) -> None:
    """Train the scorer with a speaker-independent split; save model to models/.
    TODO (Phase 2): CV, AUC/F1/confusion matrix, calibrate probabilities."""
    raise NotImplementedError("Implemented in Phase 2.")


def score(markers: dict[str, float], model_path: str, norms_path: str) -> ScoreResult:
    """Return risk probability + which markers fall outside healthy-control norms.
    TODO (Phase 2)."""
    raise NotImplementedError("Implemented in Phase 2.")
