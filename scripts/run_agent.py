#!/usr/bin/env python
"""Phase 4: run the whole agent in one call (the single entry point).

Unlike run_report.py (which chains steps manually), this calls the orchestrator's
run_session(), which owns the state machine and the re-record decision.

    uv run python scripts/run_agent.py sample.wav --model small --out report.md

Needs a trained scorer (Phase 2) and an LLM key in .env (for the report step).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cognitive_speech_agent.config import load_config
from cognitive_speech_agent import orchestrator as O


def main(argv: list[str] | None = None) -> int:
    cfg = load_config()
    p = argparse.ArgumentParser(description="Run the cognitive-speech agent (Phase 4).")
    p.add_argument("audio")
    p.add_argument("--diarize", action="store_true", help="two-speaker recording")
    p.add_argument("--model", default=None, help="override whisper size for this run")
    p.add_argument("--out", default=None, help="write the report to this .md file")
    args = p.parse_args(argv)

    if args.model:
        cfg["transcription"]["model_size"] = args.model

    sess = O.run_session(args.audio, cfg, diarize=args.diarize)

    print(f"[state: {sess.state.value}] {sess.message}", file=sys.stderr)
    if not sess.ok:
        return 2                                  # e.g. NEEDS_RERECORD

    if args.out:
        Path(args.out).write_text(sess.report, encoding="utf-8")
        print(f"Wrote report -> {args.out}", file=sys.stderr)
    else:
        print("\n" + sess.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())