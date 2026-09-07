"""Tool 5 - Report: grounded LLM writing from a fact sheet.

Design rule: the model computes the truth, the LLM only narrates it.
    1. build_fact_sheet()  - assemble every value the writer is allowed to use
    2. render_report()     - LLM writes ONLY prose; numbers are printed by us
    3. validate_numbers()  - guardrail: catch any number the LLM invented

The final report's figures (risk score, percentiles, flagged table) are rendered
deterministically from the fact sheet, so they are correct by construction. The
LLM fills three prose fields: summary, observations, recommendation.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .scoring import ScoreResult
from .transcribe import Transcript

# Human-readable names for the markers we most often surface.
READABLE = {
    "timing_pause_rate_per_min": "rate of pauses",
    "timing_long_pause_count": "number of long pauses",
    "timing_filled_pause_ratio": 'filled pauses ("um"/"uh")',
    "timing_speech_rate_wpm": "speaking rate",
    "vocab_type_token_ratio": "vocabulary diversity",
    "vocab_moving_avg_ttr": "vocabulary diversity (length-adjusted)",
    "vocab_vague_term_ratio": 'vague-word use ("thing"/"stuff")',
    "vocab_honore_r": "lexical richness",
    "gram_mean_length_utterance": "sentence length",
    "gram_propositional_density": "idea density",
    "gram_pronoun_noun_ratio": "pronoun-to-noun ratio",
    "coh_global_coherence_mean": "topic coherence",
    "coh_info_unit_coverage": "picture details mentioned",
}


def _readable(name: str) -> str:
    return READABLE.get(name, name.split("_", 1)[-1].replace("_", " "))


@dataclass
class FactSheet:
    """Everything the report writer is allowed to use. No other numbers permitted."""
    task_type: str
    duration_sec: float
    risk_probability: float
    risk_label: str
    risk_band: str                                   # low | moderate | elevated
    flagged: list[dict] = field(default_factory=list)         # {name, readable, percentile, direction}
    key_markers: dict[str, float] = field(default_factory=dict)
    transcript_excerpts: list[str] = field(default_factory=list)
    disclaimer: str = ""

    def to_json(self) -> str:
        return json.dumps({
            "task_type": self.task_type,
            "duration_sec": round(self.duration_sec, 1),
            "risk_probability": round(self.risk_probability, 2),
            "risk_label": self.risk_label,
            "risk_band": self.risk_band,
            "flagged_markers": self.flagged,
            "key_markers": {k: round(v, 3) for k, v in self.key_markers.items()},
            "transcript_excerpts": self.transcript_excerpts,
        }, indent=2)


# --- 1. build the fact sheet -------------------------------------------------

def build_fact_sheet(
    markers: dict[str, float], score: ScoreResult, transcript: Transcript, config: dict
) -> FactSheet:
    """Turn raw markers + score into the grounded, writer-facing fact sheet."""
    p = float(score.risk_probability)
    band = "low" if p < 0.34 else ("moderate" if p < 0.67 else "elevated")

    flagged = []
    for name, pct in sorted(score.flagged_markers.items(), key=lambda kv: abs(kv[1] - 50), reverse=True):
        flagged.append({
            "name": name,
            "readable": _readable(name),
            "percentile": round(float(pct), 1),
            "direction": "above typical" if pct >= 50 else "below typical",
        })

    key = {k: markers[k] for k in (
        "timing_speech_rate_wpm", "timing_pause_rate_per_min",
        "vocab_moving_avg_ttr", "coh_info_unit_coverage") if k in markers}

    # short, de-identified excerpt (first ~30 words) for illustration only
    words = transcript.text.split()
    excerpt = " ".join(words[:30]) + ("…" if len(words) > 30 else "")

    disclaimer = (config or {}).get("safety", {}).get("disclaimer", "").strip() or (
        "Research screening demonstration only. Not a diagnosis and not a medical "
        "device. A licensed clinician must review all output.")

    return FactSheet(
        task_type=markers.get("_task", (config or {}).get("task", {}).get("type", "picture_description")),
        duration_sec=float(markers.get("timing_duration_sec", transcript.duration)),
        risk_probability=p, risk_label=score.label, risk_band=band,
        flagged=flagged, key_markers=key,
        transcript_excerpts=[excerpt] if excerpt.strip() else [],
        disclaimer=disclaimer,
    )


# --- 2. render the report (LLM writes prose only) ----------------------------

SYSTEM_PROMPT = (
    "You are a careful clinical-writing assistant helping draft a cognitive SCREENING "
    "note from speech markers. Strict rules:\n"
    "- Use ONLY the values in the provided fact sheet. Do NOT introduce any number, "
    "statistic, percentile, or score that is not in it.\n"
    "- This is a screening aid, NOT a diagnosis. Never state or imply the person has "
    "dementia or any disease. Use hedged language ('may warrant', 'is consistent with').\n"
    "- Neutral, professional, non-alarming tone. No treatment advice.\n"
    "- Return STRICT JSON with exactly these keys: summary, observations, recommendation. "
    "Each value is 1-3 sentences of prose with NO numbers in it."
)


def _user_prompt(fs: FactSheet) -> str:
    return (
        "Here is the fact sheet (the only information you may use):\n\n"
        f"{fs.to_json()}\n\n"
        "Write the three prose fields. In 'observations', refer to the flagged markers "
        "by their readable names and whether they are above/below typical, but WITHOUT "
        "quoting the numeric percentiles (the report prints those separately). Return JSON only."
    )


def _parse_sections(text: str) -> dict:
    """Extract the JSON object the model returned (robust to stray text)."""
    try:
        return json.loads(text)
    except Exception:  # noqa: BLE001
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:  # noqa: BLE001
                pass
    return {"summary": text.strip(), "observations": "", "recommendation": ""}


def _assemble(fs: FactSheet, prose: dict) -> str:
    """Deterministically lay out the report; numbers come from the fact sheet only."""
    lines = [
        "# Cognitive Speech Screening — Research Report",
        "",
        f"**Task:** {fs.task_type}  **Sample length:** {fs.duration_sec:.1f}s",
        "",
        "## Summary",
        prose.get("summary", "").strip(),
        "",
        "## What was observed",
        prose.get("observations", "").strip(),
    ]
    if fs.flagged:
        lines += ["", "### Markers outside the typical range",
                  "| Marker | Percentile vs. controls | Direction |",
                  "| --- | --- | --- |"]
        lines += [f"| {f['readable']} | {f['percentile']:.0f} | {f['direction']} |"
                  for f in fs.flagged]
    lines += [
        "",
        "## Screening indication",
        f"Overall screening signal: **{fs.risk_label}** "
        f"(model risk score {fs.risk_probability:.2f}, {fs.risk_band} band).",
        "",
        prose.get("recommendation", "").strip(),
        "",
        "---",
        f"_{fs.disclaimer}_",
    ]
    return "\n".join(lines)


def render_report(fact_sheet: FactSheet, config: dict, complete_fn=None) -> str:
    """LLM writes the prose; we assemble and validate. Pass complete_fn to test offline."""
    rep_cfg = (config or {}).get("report", {})
    if complete_fn is None:
        from .llm import complete as complete_fn  # noqa: F811

    raw = complete_fn(
        SYSTEM_PROMPT, _user_prompt(fact_sheet),
        provider=rep_cfg.get("provider", "openai"),
        model=rep_cfg.get("model", "gpt-4o"),
        temperature=0.2,
    )
    prose = _parse_sections(raw)

    if rep_cfg.get("validate_numbers", True):
        prose_text = " ".join(str(prose.get(k, "")) for k in ("summary", "observations", "recommendation"))
        stray = validate_numbers(prose_text, fact_sheet)
        if stray:
            # strip invented numbers rather than risk a wrong figure in the note
            for k in ("summary", "observations", "recommendation"):
                prose[k] = re.sub(r"\b\d+(?:\.\d+)?%?\b", "", str(prose.get(k, ""))).strip()

    return _assemble(fact_sheet, prose)


# --- 3. guardrail ------------------------------------------------------------

def validate_numbers(report_text: str, fact_sheet: FactSheet) -> list[str]:
    """Return numbers in the text that are NOT traceable to the fact sheet.

    Heuristic anti-hallucination check: every numeric token the LLM wrote should
    correspond (within rounding) to a value we gave it. Ideally the prose has no
    numbers at all (they are printed deterministically), so this should be empty.
    """
    allowed: set[float] = set()

    def add(v):
        try:
            f = float(v)
        except (TypeError, ValueError):
            return
        for x in (f, round(f), round(f, 1), round(f * 100)):
            allowed.add(float(x))

    add(fact_sheet.duration_sec)
    add(fact_sheet.risk_probability)
    add(len(fact_sheet.flagged))
    for f in fact_sheet.flagged:
        add(f.get("percentile"))
    for v in fact_sheet.key_markers.values():
        add(v)
    excerpt_nums = set(re.findall(r"\d+(?:\.\d+)?", " ".join(fact_sheet.transcript_excerpts)))

    stray = []
    for tok in re.findall(r"\d+(?:\.\d+)?", report_text):
        val = float(tok)
        if tok in excerpt_nums:
            continue
        if any(abs(val - a) <= 0.5 or (a and abs(val - a) / max(abs(a), 1e-9) <= 0.01)
               for a in allowed):
            continue
        stray.append(tok)
    return stray
