"""Tool 3 - Marker extraction: the interpretable cognitive-linguistic features.

Four families feed both the report and (optionally) the scorer:
    timing      - pauses, speech rate, fillers            (needs word timings)
    vocabulary  - lexical diversity/richness, vague words (plain text; no spaCy)
    grammar     - MLU, tree depth, propositional density  (spaCy)
    coherence   - topic drift, repetition, info-units     (sentence embeddings)

Design notes
------------
* Timing and vocabulary are pure-Python (numpy only) so they can be unit-tested
  without spaCy / sentence-transformers installed.
* spaCy and sentence-transformers are imported lazily and cached.
* Every feature is documented; higher-level code turns these numbers into the
  report ("the model computes the truth, the LLM only narrates it").
"""
from __future__ import annotations

import math
import re
from functools import lru_cache

from .transcribe import Transcript

# --- thresholds / lexicons ---------------------------------------------------

PAUSE_THRESHOLD_SEC = 0.15     # gap above this between words counts as a pause
LONG_PAUSE_SEC = 0.50          # gap above this counts as a "long" pause
MATTR_WINDOW = 50              # tokens per window for moving-average TTR

FILLERS = {"um", "uh", "erm", "er", "hmm", "mm", "eh", "ah", "uhm"}

VAGUE_TERMS = {
    "thing", "things", "stuff", "something", "anything", "everything",
    "someone", "somebody", "somewhere", "somehow", "whatever", "whatsit",
    "thingy", "one", "ones",
}

# Standard Cookie Theft information units (Boston Diagnostic Aphasia Examination).
# Each unit maps to keyword variants; a unit counts as "present" if any appears.
COOKIE_THEFT_UNITS: dict[str, list[str]] = {
    "boy": ["boy"],
    "girl": ["girl", "sister"],
    "woman": ["woman", "mother", "mom", "lady"],
    "kitchen": ["kitchen"],
    "cookie": ["cookie", "cookies", "biscuit"],
    "jar": ["jar"],
    "stool": ["stool"],
    "falling_stool": ["fall", "falling", "fell", "tipping", "toppling", "wobbl"],
    "sink": ["sink"],
    "water_overflow": ["overflow", "spill", "spilling", "running over", "flooding"],
    "dishes": ["dish", "dishes", "plate", "plates"],
    "cupboard": ["cupboard", "cabinet"],
    "window": ["window"],
    "curtain": ["curtain", "curtains"],
    "counter": ["counter", "worktop"],
    "drying": ["drying", "dry", "washing", "wiping", "wiping"],
}


# --- helpers -----------------------------------------------------------------

def _tokens(text: str) -> list[str]:
    """Lowercase alphabetic word tokens (apostrophes kept)."""
    return re.findall(r"[a-zA-Z']+", text.lower())


@lru_cache(maxsize=1)
def _nlp():
    import spacy  # lazy

    return spacy.load("en_core_web_sm")


@lru_cache(maxsize=1)
def _embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    from sentence_transformers import SentenceTransformer  # lazy

    return SentenceTransformer(model_name)


def _safe(v: float) -> float:
    return float(v) if (v is not None and math.isfinite(v)) else 0.0


# --- 1. timing ---------------------------------------------------------------

def timing_markers(transcript: Transcript) -> dict[str, float]:
    """Pauses, speech rate, and fillers from word timings.

    Note: pauses are gaps between *consecutive kept words*. For a monologue this
    is exact; after diarization, inter-turn gaps are already excluded because the
    clinician's words were removed upstream.
    """
    words = transcript.words
    duration = transcript.duration or (words[-1].end if words else 0.0)
    n = len(words)
    out: dict[str, float] = {
        "duration_sec": _safe(duration),
        "num_words": float(n),
    }
    if n == 0 or duration <= 0:
        return out

    # pauses
    gaps = [words[i + 1].start - words[i].end for i in range(n - 1)]
    pauses = [g for g in gaps if g > PAUSE_THRESHOLD_SEC]
    long_pauses = [g for g in gaps if g > LONG_PAUSE_SEC]
    total_pause = sum(pauses)
    speech_time = sum(w.duration for w in words)

    # fillers
    toks = [w.text.lower().strip(".,?!") for w in words]
    n_fillers = sum(1 for t in toks if t in FILLERS)

    out.update({
        "speech_rate_wpm": _safe(n / (duration / 60.0)),
        "articulation_rate_wps": _safe(n / speech_time) if speech_time > 0 else 0.0,
        "num_pauses": float(len(pauses)),
        "pause_rate_per_min": _safe(len(pauses) / (duration / 60.0)),
        "total_pause_sec": _safe(total_pause),
        "mean_pause_sec": _safe(total_pause / len(pauses)) if pauses else 0.0,
        "pause_ratio": _safe(total_pause / duration),      # fraction of time silent
        "long_pause_count": float(len(long_pauses)),
        "filled_pause_count": float(n_fillers),
        "filled_pause_ratio": _safe(n_fillers / n),
    })
    return out


# --- 2. vocabulary -----------------------------------------------------------

def vocabulary_markers(transcript: Transcript) -> dict[str, float]:
    """Lexical diversity and richness (no spaCy needed)."""
    toks = _tokens(transcript.text)
    n = len(toks)
    out: dict[str, float] = {"num_tokens": float(n)}
    if n == 0:
        return out

    types = set(toks)
    v = len(types)
    freq: dict[str, int] = {}
    for t in toks:
        freq[t] = freq.get(t, 0) + 1
    hapax = sum(1 for c in freq.values() if c == 1)   # words used exactly once

    # moving-average TTR (robust to length)
    if n >= MATTR_WINDOW:
        ttrs = []
        for i in range(n - MATTR_WINDOW + 1):
            window = toks[i : i + MATTR_WINDOW]
            ttrs.append(len(set(window)) / MATTR_WINDOW)
        mattr = sum(ttrs) / len(ttrs)
    else:
        mattr = v / n

    # Brunet's W (lower = richer): N ** (V ** -0.165)
    brunet_w = n ** (v ** -0.165) if v > 0 else 0.0
    # Honore's R (higher = richer): 100 * logN / (1 - V1/V)
    denom = 1.0 - (hapax / v) if v > 0 else 0.0
    honore_r = (100.0 * math.log(n)) / denom if denom > 0 else 0.0

    n_vague = sum(freq.get(w, 0) for w in VAGUE_TERMS)

    out.update({
        "num_types": float(v),
        "type_token_ratio": _safe(v / n),
        "moving_avg_ttr": _safe(mattr),
        "hapax_ratio": _safe(hapax / n),
        "brunet_w": _safe(brunet_w),
        "honore_r": _safe(honore_r),
        "vague_term_ratio": _safe(n_vague / n),
    })
    return out


# --- 3. grammar --------------------------------------------------------------

def _tree_depth(token) -> int:
    depth = 0
    while token.head != token:
        depth += 1
        token = token.head
        if depth > 100:
            break
    return depth


def grammar_markers(transcript: Transcript) -> dict[str, float]:
    """Syntactic complexity and propositional density (spaCy)."""
    text = transcript.text.strip()
    if not text:
        return {}
    doc = _nlp()(text)
    sents = list(doc.sents)
    content_toks = [t for t in doc if t.is_alpha]
    n = len(content_toks)
    out: dict[str, float] = {"num_sentences": float(len(sents))}
    if n == 0 or not sents:
        return out

    mlu = n / len(sents)                                   # mean length of utterance
    depths = [max((_tree_depth(t) for t in s), default=0) for s in sents]
    mean_depth = sum(depths) / len(depths)

    pos = [t.pos_ for t in content_toks]
    n_pron = pos.count("PRON")
    n_noun = pos.count("NOUN") + pos.count("PROPN")
    n_verb = pos.count("VERB")
    # subordination: subordinating conjunctions + clausal dependents
    n_subord = sum(1 for t in doc if t.dep_ in {"mark", "advcl", "ccomp", "xcomp", "acl"})
    # propositional (idea) density: verbs, adj, adv, prepositions, conjunctions / words
    prop = sum(1 for p in pos if p in {"VERB", "ADJ", "ADV", "ADP", "CCONJ", "SCONJ"})

    out.update({
        "mean_length_utterance": _safe(mlu),
        "mean_tree_depth": _safe(mean_depth),
        "subordination_ratio": _safe(n_subord / len(sents)),
        "propositional_density": _safe(prop / n),
        "pronoun_ratio": _safe(n_pron / n),
        "noun_ratio": _safe(n_noun / n),
        "verb_ratio": _safe(n_verb / n),
        "pronoun_noun_ratio": _safe(n_pron / n_noun) if n_noun > 0 else 0.0,
    })
    return out


# --- 4. coherence ------------------------------------------------------------

def _sentences(text: str) -> list[str]:
    try:
        return [s.text.strip() for s in _nlp()(text).sents if s.text.strip()]
    except Exception:  # noqa: BLE001 - fall back to naive split if spaCy unavailable
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def coherence_markers(
    transcript: Transcript,
    task_type: str = "picture_description",
    embedder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> dict[str, float]:
    """Topic drift (sentence-embedding similarity), repetition, and info-units."""
    import numpy as np

    text = transcript.text.strip()
    out: dict[str, float] = {}
    sents = _sentences(text)

    if len(sents) >= 2:
        emb = _embedder(embedder_name).encode(sents, normalize_embeddings=True)
        emb = np.asarray(emb)
        adj = [float(np.dot(emb[i], emb[i + 1])) for i in range(len(emb) - 1)]
        centroid = emb.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-9)
        global_sim = [float(np.dot(e, centroid)) for e in emb]
        out.update({
            "local_coherence_mean": _safe(sum(adj) / len(adj)),
            "local_coherence_min": _safe(min(adj)),
            "global_coherence_mean": _safe(sum(global_sim) / len(global_sim)),
        })

    # repetition: fraction of repeated bigrams
    toks = _tokens(text)
    if len(toks) >= 2:
        bigrams = list(zip(toks[:-1], toks[1:]))
        uniq = len(set(bigrams))
        out["bigram_repetition_ratio"] = _safe(1.0 - uniq / len(bigrams))

    # task-specific: Cookie Theft information-unit coverage
    if task_type == "picture_description":
        low = text.lower()
        present = sum(
            1 for kws in COOKIE_THEFT_UNITS.values() if any(k in low for k in kws)
        )
        out["info_unit_count"] = float(present)
        out["info_unit_coverage"] = _safe(present / len(COOKIE_THEFT_UNITS))

    return out


# --- orchestration -----------------------------------------------------------

def extract_markers(
    transcript: Transcript,
    task_type: str = "picture_description",
    families: dict[str, bool] | None = None,
    embedder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> dict[str, float]:
    """Run enabled marker families and return one flat, prefixed feature dict."""
    fam = families or {"timing": True, "vocabulary": True, "grammar": True, "coherence": True}
    out: dict[str, float] = {}
    if fam.get("timing", True):
        out.update({f"timing_{k}": v for k, v in timing_markers(transcript).items()})
    if fam.get("vocabulary", True):
        out.update({f"vocab_{k}": v for k, v in vocabulary_markers(transcript).items()})
    if fam.get("grammar", True):
        out.update({f"gram_{k}": v for k, v in grammar_markers(transcript).items()})
    if fam.get("coherence", True):
        out.update({
            f"coh_{k}": v
            for k, v in coherence_markers(transcript, task_type, embedder_name).items()
        })
    return out
