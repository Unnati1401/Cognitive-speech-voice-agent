# data/

**No audio, transcripts, or derived features are ever committed to git** (see `.gitignore`).

Layout:

```
data/
  raw/         # source audio from DementiaBank / ADReSS or PROCESS (gitignored)
  interim/     # diarized + transcribed intermediates
  processed/   # feature tables (CSV/parquet) used for modeling
```

Create the folders locally:

```bash
mkdir -p data/raw data/interim data/processed
```

Keep all patient data on this machine only. Do not upload to any cloud, notebook
service, or LLM prompt. The report writer receives only aggregate markers and
short de-identified excerpts, never raw recordings.
