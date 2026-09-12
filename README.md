---
title: Cognitive Speech Screening
emoji: 🧠
colorFrom: indigo
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
license: mit
---

# Cognitive Speech Screening Agent

**🔗 Live demo:** https://huggingface.co/spaces/unnati1401/Cognitive-Speech-voice-agent

An AI agent that listens to a short picture-description clip, extracts interpretable
**cognitive-linguistic markers** (pauses, vocabulary diversity, syntactic complexity,
discourse coherence), scores them, and drafts a **grounded screening report** for a
clinician to review.

> **Research demonstration only. Not a diagnosis and not a medical device.**
> Outputs are speech markers and a screening flag, never validated clinical test scores.
> A licensed clinician must review all output. Scores in the hosted demo come from a
> placeholder model trained on synthetic clips; they show the pipeline, not clinical signal.

The design rule throughout: **the model computes the truth, the LLM only narrates it.**
Every number in a report is produced by deterministic code and a trained classifier; the
language model writes the surrounding prose and is checked so it cannot invent figures.

---

## How it works

The agent is built from five tools wired together by an orchestrator:

```
diarize ─▶ transcribe ─▶ extract markers ─▶ score ─▶ report
(pyannote)  (Whisper)     (spaCy/openSMILE)  (XGBoost)  (grounded LLM)
```

- **Diarize** — separate the patient's voice from the clinician's (skippable for single-speaker clips).
- **Transcribe** — faster-whisper produces text with word-level timestamps.
- **Markers** — four families: timing (pauses, speech rate, fillers), vocabulary (lexical
  diversity, vague-word use), grammar (sentence length, idea density), coherence (topic drift,
  Cookie-Theft information units).
- **Score** — a classifier trained with a **speaker-independent** split; outputs a risk
  probability plus per-marker flags versus healthy-control norms.
- **Report** — an LLM writes the narrative from a structured fact sheet; a guardrail strips any
  number it wasn't given.

The **orchestrator** runs these as one stateful `run_session()` call and makes decisions (e.g.
if a clip is too short or empty it stops and asks for a re-record instead of scoring bad data).

---

## Status

Pipeline built and tested end to end. **Awaiting research data access** (see below) before the
scorer can be trained on real recordings and reported with meaningful accuracy.

| Phase | What | Status |
| --- | --- | --- |
| 0 | Project scaffold, environment (uv), data policy | ✅ Done |
| 1 | Diarize + transcribe + marker extraction | ✅ Done |
| 2 | Feature table + speaker-independent scorer | ✅ Done |
| 3 | Grounded LLM report + anti-hallucination guardrail | ✅ Done |
| 4 | Agent orchestrator (`run_session`, re-record logic) | ✅ Done |
| 5 | Gradio web UI + deployable Hugging Face Space | ✅ Done |
| 6 | Evaluation + writeup on real data (AUC/F1, ablation) | ✅ Done |

## Results (PROCESS-2, Cookie Theft task)

Trained on the official PROCESS-2 split (320 participants) and evaluated on the held-out test
set (80 unseen participants), collapsing MCI + Dementia into "impaired" vs. healthy controls.
Interpretable hand-built markers + a logistic-regression scorer, speaker-independent throughout.

| Evaluation | AUC-ROC | F1 | Accuracy |
| --- | --- | --- | --- |
| Held-out official test (n=80) | **0.795** | 0.68 | 0.70 |
| Speaker-independent CV (train, n=320) | 0.66 | 0.61 | 0.63 |

Held-out confusion matrix `[[TN 30, FP 10], [FN 14, TP 26]]` → ~65% sensitivity, 75% specificity.

**Which markers carry the signal (CV AUC, each family alone):** coherence 0.68 · timing 0.66 ·
grammar 0.59 · vocabulary 0.58. Discourse coherence and speech-timing markers are the most
informative, consistent with the cognitive-linguistic literature.

**Honest context:** the dataset's own best Cookie-Theft baseline is F1 ≈ 0.85 using a fine-tuned
DistilBERT over transcripts. This project deliberately trades a few points of raw performance for
**full interpretability** — every prediction traces back to specific, named speech markers — which
is the intended contribution rather than a leaderboard score.

## Data

The system is designed around the **ADReSS / DementiaBank** Cookie-Theft corpus and the
**PROCESS** challenge corpus (picture description + verbal fluency, with healthy / MCI / dementia
labels). Both are access-controlled because the recordings are sensitive clinical data.

Results above use **PROCESS-2** (CognoSpeak), a controlled-access corpus of 400 participants
(HC / MCI / Dementia) with an official train/test split, loaded via `datasets.load_dataset`.
Access is granted under the PROCESS-2 Data Use Agreement; the recordings and any derived feature
tables are **not** redistributed here (kept local, gitignored) — only the trained model and
aggregate metrics are shared, per the DUA.

---

## Setup (uv)

Prerequisites: [uv](https://docs.astral.sh/uv/) and system `ffmpeg`.

```bash
uv sync --extra app --extra dev
uv run python -m spacy download en_core_web_sm

# secrets in a local .env (auto-loaded; never committed)
echo "OPENAI_API_KEY=sk-..." >> .env      # for the report step
echo "HF_TOKEN=hf_..."       >> .env      # only for diarization (two speakers)

uv run python scripts/check_env.py        # verify the toolchain
```

## Usage

```bash
# run the whole agent on one clip (audio -> written report)
uv run python scripts/run_agent.py clip.wav --model small --out report.md

# just see the markers for a clip
uv run python scripts/run_markers.py clip.wav --model small

# build a feature table + train the scorer (speaker-independent CV)
uv run python scripts/build_features.py --audio-dir data/raw --out data/processed/features.csv
uv run python scripts/train_scorer.py

# launch the web UI (upload / record / click a sample)
uv run python scripts/app.py
```

Run the tests:

```bash
uv run pytest -q
```

---

## Project layout

```
cognitive-speech-agent/
├── app.py                       # Hugging Face Space entry point
├── config.yaml                  # single source of truth (tasks, models, thresholds, safety)
├── requirements.txt packages.txt# Space deps
├── src/cognitive_speech_agent/
│   ├── diarize.py transcribe.py markers.py     # tools 1–3
│   ├── scoring.py report.py llm.py             # tools 4–5
│   ├── orchestrator.py                         # the agent loop
│   └── config.py                               # config + .env loader
├── scripts/                     # runnable entry points (agent, training, UI, samples)
├── assets/samples/              # demo clips for the web UI
├── tests/                       # unit tests for every phase
└── docs/                        # data-access request emails, notes
```

## Ethics & safety

- Framed as a **screening research demo, not a diagnostic tool**; clinician always in the loop.
- Only consented research datasets; recordings and derived features stay local and are gitignored.
- **Speaker-independent** train/test splits only (never split by utterance).
- Honest limitations reporting: dataset scope, demographic coverage, and generalization caveats.
