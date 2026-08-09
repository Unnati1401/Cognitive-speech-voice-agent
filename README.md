# Cognitive Speech Screening Agent

A voice-based AI agent that listens to a short, open-ended speech sample (a picture
description), extracts interpretable **cognitive-linguistic markers**, scores them, and drafts a
**screening report** for a clinician to review.

> **Research demonstration only. Not a diagnosis and not a medical device.**
> Outputs are speech markers and a screening flag, not validated clinical test scores.
> A licensed clinician must review all output. See `DATA_POLICY.md`.

## Pipeline (five tools + an agent loop)

```
diarize -> transcribe -> extract markers -> score -> report
(pyannote) (Whisper)     (spaCy/openSMILE)  (XGBoost) (grounded LLM)
```

The orchestrator (`orchestrator.py`) sequences these as a stateful agent, with the design rule:
**the model computes the truth, the LLM only narrates it.**

## Setup (uv + local CUDA GPU)

Prerequisites: [uv](https://docs.astral.sh/uv/), and system `ffmpeg`
(`sudo apt install ffmpeg` / `brew install ffmpeg`).

```bash
# 1. install deps (creates .venv, resolves CUDA torch from the pytorch index)
uv sync --extra app --extra dev

# 2. spaCy English model
uv run python -m spacy download en_core_web_sm

# 3. HuggingFace token for pyannote diarization
#    - accept model terms at hf.co/pyannote/speaker-diarization-3.1
#    - then:
echo "HF_TOKEN=hf_xxx" >> .env

# 4. LLM key for the report writer (Anthropic by default; see config.yaml)
echo "ANTHROPIC_API_KEY=sk-ant-xxx" >> .env

# 5. verify everything
uv run python scripts/check_env.py

# 6. create local data folders (gitignored)
mkdir -p data/raw data/interim data/processed
```

## Configuration

All knobs live in `config.yaml` (task type, Whisper size, marker toggles, scoring approach,
report provider, safety disclaimer). Code reads it via `config.load_config()`.

## Project layout

```
cognitive-speech-agent/
  config.yaml                 # single source of truth
  DATA_POLICY.md              # ethics + data-handling rules (read first)
  scripts/check_env.py        # Phase 0 environment verification
  docs/dementiabank_access_email.md
  src/cognitive_speech_agent/
    diarize.py     transcribe.py   markers.py
    scoring.py     report.py       orchestrator.py   config.py
  data/                       # gitignored; no patient data in git
  tests/
```

## Data

Primary: **ADReSS** (balanced Pitt-corpus subset, 156 participants) via the DementiaBank
consortium; request access with `docs/dementiabank_access_email.md`.
Fallback while access is pending: the open **PROCESS / PROCESS-2** cognitive-impairment speech
corpus. Set paths in `config.yaml`.

## Roadmap

- **Phase 0 (this scaffold):** environment, access request, data policy. ✅ when `check_env.py` passes.
- **Phase 1:** implement diarize/transcribe/markers.
- **Phase 2:** train the scorer (speaker-independent split).
- **Phase 3:** grounded report writer + number-validation guardrail.
- **Phase 4:** agent orchestration loop.
- **Phase 5:** Gradio demo.
- **Phase 6:** evaluation, ablation, writeup.

Full detail in the build spec.
