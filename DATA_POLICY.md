# Data & Ethics Policy

This project is a **research screening demonstration, not a medical device**, and handles
sensitive clinical speech data. These rules are non-negotiable and apply from day one.

## Framing
- Outputs are **cognitive-linguistic markers and a screening flag**, never a diagnosis or a
  validated clinical test score.
- A **licensed clinician is always in the loop**; the report is a draft for professional review.
- Every report and the project README carry the disclaimer in `config.yaml -> safety.disclaimer`.

## Data handling
- Use **only consented research datasets** (DementiaBank/ADReSS under consortium rules, or the
  open PROCESS corpus). **Never** use real patient recordings collected outside these.
- All audio, transcripts, and derived feature tables stay **local**; they are gitignored and
  never uploaded to cloud storage, notebook services, or LLM prompts.
- **No redistribution** of any recordings or transcripts. **No re-identification** attempts.
- The LLM report writer receives only aggregate markers and **short de-identified excerpts**,
  never raw audio or full transcripts.

## Modeling integrity
- **Speaker-independent splits only.** Never split train/test by utterance.
- Report honest metrics (AUC/F1/confusion matrix) with a limitations section covering dataset
  scope and demographic coverage.

## Secrets
- API keys and HuggingFace tokens live in a local `.env` (gitignored), never in code or config.
