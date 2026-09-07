#!/usr/bin/env python
"""Phase 3 end-to-end: audio -> markers -> score -> grounded LLM report.

Requires a trained scorer (run Phase 2 first) and an LLM key in .env
(OPENAI_API_KEY or ANTHROPIC_API_KEY, per config.yaml report.provider).

    uv run python scripts/run_report.py sample.wav --model small
    uv run python scripts/run_report.py sample.wav --out report.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cognitive_speech_agent.config import load_config
from cognitive_speech_agent import diarize as D
from cognitive_speech_agent import markers as M
from cognitive_speech_agent import report as R
from cognitive_speech_agent import scoring
from cognitive_speech_agent.transcribe import transcribe


def main(argv: list[str] | None = None) -> int:
    cfg = load_config()
    p = argparse.ArgumentParser(description="Generate a grounded screening report (Phase 3).")
    p.add_argument("audio")
    p.add_argument("--diarize", action="store_true")
    p.add_argument("--task", default=cfg["task"]["type"])
    p.add_argument("--model", default=cfg["transcription"]["model_size"])
    p.add_argument("--out", default=None, help="write the report to this .md file")
    args = p.parse_args(argv)

    models_dir = Path(cfg.get("paths", {}).get("models", "models"))
    model_p, norms_p = models_dir / "scorer.joblib", models_dir / "norms.json"
    if not model_p.exists():
        print("No trained scorer found. Run Phase 2 (train_scorer.py) first.", file=sys.stderr)
        return 1

    tr = transcribe(args.audio, model_size=args.model,
                    compute_type=cfg["transcription"]["compute_type"],
                    language=cfg["transcription"]["language"])
    if args.diarize:
        segs = D.diarize(args.audio)
        tr = D.patient_transcript(tr, segs, D.select_target_speaker(segs))

    feats = M.extract_markers(tr, task_type=args.task, families=cfg["markers"],
                              embedder_name=cfg["markers"]["coherence_embedder"])
    result = scoring.score(feats, str(model_p), str(norms_p))
    fs = R.build_fact_sheet(feats, result, tr, cfg)
    report_md = R.render_report(fs, cfg)

    if args.out:
        Path(args.out).write_text(report_md, encoding="utf-8")
        print(f"Wrote report -> {args.out}", file=sys.stderr)
    else:
        print("\n" + report_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
