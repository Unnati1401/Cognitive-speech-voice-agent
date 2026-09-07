#!/usr/bin/env python
"""Phase 1 end-to-end runner: audio file -> transcript -> markers.

Runs the marker pipeline on ONE recording so you can eyeball the numbers before
building the scorer. For a single-speaker clip (you describing the picture, or a
`say`-generated test clip) diarization is skipped. Pass --diarize for a real
two-person clinician+patient recording.

Examples
--------
    # quick smoke test with a Mac text-to-speech clip (no real data needed):
    say -o sample.wav --data-format=LEI16@16000 \
        "the little boy is taking a cookie from the jar while his sister waits"
    uv run python scripts/run_markers.py sample.wav --model small

    # real two-speaker visit recording:
    uv run python scripts/run_markers.py visit.wav --diarize

    # save the markers to JSON:
    uv run python scripts/run_markers.py sample.wav --json markers.json
"""
from __future__ import annotations

import argparse
import json
import sys

from cognitive_speech_agent.config import load_config
from cognitive_speech_agent import diarize as D
from cognitive_speech_agent import markers as M
from cognitive_speech_agent.transcribe import transcribe


def _print_group(title: str, prefix: str, feats: dict[str, float]) -> None:
    rows = {k[len(prefix):]: v for k, v in feats.items() if k.startswith(prefix)}
    if not rows:
        return
    print(f"\n{title}")
    for k, v in rows.items():
        print(f"  {k:<26} {v:>10.3f}")


def main(argv: list[str] | None = None) -> int:
    cfg = load_config()
    p = argparse.ArgumentParser(description="Run the Phase 1 marker pipeline on one audio file.")
    p.add_argument("audio", help="path to an audio file (wav/mp3/m4a/…)")
    p.add_argument("--diarize", action="store_true",
                   help="two-speaker recording: split speakers and keep the patient only")
    p.add_argument("--task", default=cfg["task"]["type"],
                   help="picture_description | open_prompt (default from config.yaml)")
    p.add_argument("--model", default=cfg["transcription"]["model_size"],
                   help="whisper size: tiny/base/small/medium/large-v3 (small is fast on Mac)")
    p.add_argument("--json", dest="json_out", default=None, help="optional path to save markers")
    args = p.parse_args(argv)

    # 1. transcribe (word-level timestamps)
    print(f"Transcribing {args.audio} with whisper '{args.model}' …", file=sys.stderr)
    tr = transcribe(
        args.audio,
        model_size=args.model,
        compute_type=cfg["transcription"]["compute_type"],
        language=cfg["transcription"]["language"],
    )

    # 2. (optional) diarize -> keep only the patient's words
    if args.diarize:
        print("Diarizing and selecting the main speaker …", file=sys.stderr)
        segments = D.diarize(args.audio)
        speaker = D.select_target_speaker(segments)
        tr = D.patient_transcript(tr, segments, speaker)
        print(f"Kept speaker '{speaker}' ({len(tr.words)} words).", file=sys.stderr)

    # 3. extract markers
    feats = M.extract_markers(
        tr, task_type=args.task,
        families=cfg["markers"],
        embedder_name=cfg["markers"]["coherence_embedder"],
    )

    # 4. report to the console
    print("\n" + "=" * 60)
    print(f"TRANSCRIPT ({tr.duration:.1f}s, {len(tr.words)} words)")
    print("=" * 60)
    print(tr.text or "(empty)")
    _print_group("TIMING",      "timing_", feats)
    _print_group("VOCABULARY",  "vocab_",  feats)
    _print_group("GRAMMAR",     "gram_",   feats)
    _print_group("COHERENCE",   "coh_",    feats)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(feats, f, indent=2)
        print(f"\nSaved {len(feats)} markers -> {args.json_out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
