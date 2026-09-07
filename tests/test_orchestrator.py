"""Phase 4 tests: the agent state machine, offline (no Whisper / model / API).

We inject fake transcribe/score/complete functions so only the orchestration
logic is exercised.
"""
from cognitive_speech_agent import orchestrator as O
from cognitive_speech_agent.scoring import ScoreResult
from cognitive_speech_agent.transcribe import Transcript, Word


CFG = {
    "task": {"type": "picture_description", "min_duration_sec": 30},
    # keep spaCy/embeddings off so the pure-Python markers run without heavy deps
    "markers": {"timing": True, "vocabulary": True, "grammar": False,
                "coherence": False, "coherence_embedder": "unused"},
    "report": {"provider": "openai"},
    "safety": {"disclaimer": "Research screening demonstration only. Not a diagnosis "
                             "and not a medical device."},
}


def _good_transcript(_path):
    words = [Word(t, i * 0.4, i * 0.4 + 0.3)
             for i, t in enumerate("the boy takes a cookie from the jar".split())]
    return Transcript(text="the boy takes a cookie from the jar",
                      words=words, duration=60.0)


def _fake_score(_markers):
    return ScoreResult(risk_probability=0.72, label="elevated markers",
                       flagged_markers={"timing_num_words": 96.0},
                       percentiles={"timing_num_words": 96.0})


def _fake_complete(system, user, **kw):
    return ('{"summary": "Speech was somewhat hesitant.",'
            ' "observations": "Fewer content words than typical.",'
            ' "recommendation": "Follow-up screening may be warranted."}')


def test_full_session_reaches_done():
    sess = O.run_session("x.wav", CFG, transcribe_fn=_good_transcript,
                         score_fn=_fake_score, complete_fn=_fake_complete)
    assert sess.state is O.SessionState.DONE
    assert sess.ok
    assert "0.72" in sess.report
    assert "not a medical device" in sess.report.lower()


def test_short_sample_asks_for_rerecord():
    short = lambda _p: Transcript(text="um", words=[Word("um", 0.0, 0.3)], duration=5.0)
    sess = O.run_session("x.wav", CFG, transcribe_fn=short,
                         score_fn=_fake_score, complete_fn=_fake_complete)
    assert sess.state is O.SessionState.NEEDS_RERECORD
    assert not sess.ok
    assert sess.report == ""            # never reached the report step