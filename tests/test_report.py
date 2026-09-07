"""Phase 3 tests: fact-sheet assembly, grounded rendering, number guardrail.

No API key needed - render_report is called with an injected fake completion.
"""
from cognitive_speech_agent import report as R
from cognitive_speech_agent.scoring import ScoreResult
from cognitive_speech_agent.transcribe import Transcript


def _fact_sheet() -> R.FactSheet:
    tr = Transcript(text="the boy is taking a cookie from the jar", words=[], duration=42.0)
    score = ScoreResult(
        risk_probability=0.78, label="elevated markers",
        flagged_markers={"timing_long_pause_count": 96.0, "vocab_type_token_ratio": 4.0},
        percentiles={"timing_long_pause_count": 96.0, "vocab_type_token_ratio": 4.0},
    )
    markers = {"timing_duration_sec": 42.0, "timing_speech_rate_wpm": 88.0,
               "vocab_moving_avg_ttr": 0.61, "coh_info_unit_coverage": 0.4}
    return R.build_fact_sheet(markers, score, tr, {"task": {"type": "picture_description"}})


def test_fact_sheet_fields():
    fs = _fact_sheet()
    assert fs.risk_band == "elevated"            # 0.78 -> elevated
    assert len(fs.flagged) == 2
    assert fs.flagged[0]["direction"] == "above typical"


def test_validate_numbers_catches_hallucination():
    fs = _fact_sheet()
    # 0.78 and 42 are grounded; 87 is invented
    assert R.validate_numbers("the score was 0.78 over 42 seconds", fs) == []
    assert "87" in R.validate_numbers("accuracy was 87 percent", fs)


def test_render_report_is_grounded():
    fs = _fact_sheet()

    def fake_complete(system, user, **kw):
        # a well-behaved model returns prose with no numbers
        return ('{"summary": "Speech showed some hesitancy.",'
                ' "observations": "Longer pauses and reduced vocabulary diversity were noted.",'
                ' "recommendation": "Follow-up screening may be warranted."}')

    md = R.render_report(fs, {"report": {"provider": "openai"}}, complete_fn=fake_complete)
    assert "0.78" in md                          # deterministic score printed by us
    assert "elevated markers" in md
    assert "Not a diagnosis" in md or "not a medical device" in md.lower()
    assert "| Marker |" in md                    # flagged table rendered