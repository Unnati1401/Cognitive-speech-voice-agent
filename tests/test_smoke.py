"""Smoke + Phase 1 unit tests.

These cover config loading and the pure-Python marker math (timing, vocabulary,
info-units). They do NOT require spaCy / sentence-transformers / whisper, so they
run fast in CI. Grammar/coherence-embedding tests belong in an integration suite.
"""
from cognitive_speech_agent import __version__
from cognitive_speech_agent.config import load_config
from cognitive_speech_agent import markers as M
from cognitive_speech_agent.transcribe import Transcript, Word


def _synthetic_transcript() -> Transcript:
    seq = ["the", "boy", "is", "taking", "a", "cookie", "um", "from", "the", "jar"]
    words, t = [], 0.0
    for i, tok in enumerate(seq):
        words.append(Word(tok, t, t + 0.30))
        t += 0.30 + (0.8 if i == 3 else 0.1)     # one long pause after "taking"
    return Transcript(text=" ".join(seq), words=words, duration=t)


def test_version():
    assert __version__ == "0.1.0"


def test_config_loads():
    cfg = load_config()
    assert cfg["task"]["type"] in {"picture_description", "open_prompt"}
    assert cfg["scoring"]["split"] == "speaker_independent"   # the non-negotiable
    assert cfg["report"]["grounded_only"] is True


def test_timing_markers():
    tim = M.timing_markers(_synthetic_transcript())
    assert tim["num_words"] == 10
    assert tim["filled_pause_count"] == 1.0
    assert tim["long_pause_count"] == 1.0
    assert 0.0 < tim["pause_ratio"] < 1.0


def test_vocabulary_markers():
    voc = M.vocabulary_markers(_synthetic_transcript())
    assert voc["num_tokens"] == 10
    assert voc["num_types"] == 9                              # "the" repeats
    assert abs(voc["type_token_ratio"] - 0.9) < 1e-6


def test_cookie_theft_info_units():
    coh = M.coherence_markers(_synthetic_transcript(), task_type="picture_description")
    assert coh["info_unit_count"] >= 3                        # boy, cookie, jar