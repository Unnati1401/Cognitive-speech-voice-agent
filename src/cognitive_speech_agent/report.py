"""Tool 5 - Report: grounded LLM writes the screening report from a fact sheet.

Phase 3 work. Design rule: the model computes the truth, the LLM only narrates.
The LLM receives a structured fact sheet and may use ONLY those values.
A post-generation check verifies every number in the report exists in the sheet.
"""
from __future__ import annotations

from dataclasses import dataclass

from .scoring import ScoreResult
from .transcribe import Transcript


@dataclass
class FactSheet:
    """Everything the report writer is allowed to use. No other numbers permitted."""
    task_type: str
    duration_sec: float
    score: ScoreResult
    markers: dict[str, float]
    transcript_excerpts: list[str]      # short quotes to illustrate flagged markers
    disclaimer: str


def build_fact_sheet(
    markers: dict[str, float], score: ScoreResult, transcript: Transcript, config: dict
) -> FactSheet:
    """Assemble the grounded fact sheet. TODO (Phase 3)."""
    raise NotImplementedError("Implemented in Phase 3.")


def render_report(fact_sheet: FactSheet, config: dict) -> str:
    """Grounded LLM generation into the report template. TODO (Phase 3)."""
    raise NotImplementedError("Implemented in Phase 3.")


def validate_numbers(report_text: str, fact_sheet: FactSheet) -> list[str]:
    """Return any numbers in the report NOT present in the fact sheet (should be empty).
    TODO (Phase 3): the anti-hallucination guardrail."""
    raise NotImplementedError("Implemented in Phase 3.")
