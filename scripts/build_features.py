
#!/usr/bin/env python
"""Phase 2, step 1: batch the Phase 1 pipeline over many recordings -> one CSV.

Each row = one recording's markers + metadata (id, speaker_id, label, mmse).
This CSV is the input to train_scorer.py.

Two input modes
---------------
1. Manifest CSV (recommended for PROCESS/ADReSS):
     columns: audio_path, speaker_id, label[, mmse, task]
     uv run python scripts/build_features.py --manifest data/raw/manifest.csv \
         --out data/processed/features.csv

2. Folder-by-class (quick / your own clips): subfolders are the labels
     data/raw/HC/*.wav , data/raw/Dementia/*.wav , ...
     uv run python scripts/build_features.py --audio-dir data/raw \
         --out data/processed/features.csv

Add --diarize for two-speaker recordings.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from cognitive_speech_agent.config import load_config
from cognitive_speech_agent import diarize as D
from cognitive_speech_agent import markers as M
from cognitive_speech_agent.transcribe import transcribe

AUDIO_EXT = {".wav", ".mp3", ".m4a", ".flac", ".aiff", ".aif", ".ogg"}


def _iter_manifest(path: str):
    df = pd.read_csv(path)
    for _, r in df.iterrows():
        yield {
            "audio_path": str(r["audio_path"]),
            "speaker_id": str(r.get("speaker_id", Path(str(r["audio_path"])).stem)),
            "label": str(r["label"]),
            "mmse": r.get("mmse", ""),
            "task": str(r.get("task", "")) or None,
        }


def _iter_folder(root: str):
    for label_dir in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        for f in sorted(label_dir.rglob("*")):
            if f.suffix.lower() in AUDIO_EXT:
                yield {"audio_path": str(f), "speaker_id": f.stem,
                       "label": label_dir.name, "mmse": "", "task": None}


def _one_recording(item: dict, cfg: dict, model: str, do_diarize: bool) -> dict | None:
    tr = transcribe(item["audio_path"], model_size=model,
                    compute_type=cfg["transcription"]["compute_type"],
                    language=cfg["transcription"]["language"])
    if do_diarize:
        segs = D.diarize(item["audio_path"])
        tr = D.patient_transcript(tr, segs, D.select_target_speaker(segs))

    task = item["task"] or cfg["task"]["type"]
    feats = M.extract_markers(tr, task_type=task, families=cfg["markers"],
                              embedder_name=cfg["markers"]["coherence_embedder"])
    row = {"id": Path(item["audio_path"]).stem, "speaker_id": item["speaker_id"],
           "label": item["label"], "mmse": item["mmse"]}
    row.update(feats)
    return row


def main(argv: list[str] | None = None) -> int:
    cfg = load_config()
    p = argparse.ArgumentParser(description="Build the marker feature table (Phase 2).")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--manifest", help="CSV with audio_path, speaker_id, label[, mmse, task]")
    g.add_argument("--audio-dir", help="folder whose subfolders are class labels")
    p.add_argument("--out", default="data/processed/features.csv")
    p.add_argument("--model", default=cfg["transcription"]["model_size"],
                   help="whisper size (small is fast on Mac)")
    p.add_argument("--diarize", action="store_true", help="two-speaker recordings")
    args = p.parse_args(argv)

    items = list(_iter_manifest(args.manifest) if args.manifest
                 else _iter_folder(args.audio_dir))
    if not items:
        print("No recordings found.", file=sys.stderr)
        return 1

    rows, failures = [], 0
    for i, item in enumerate(items, 1):
        print(f"[{i}/{len(items)}] {item['audio_path']} ({item['label']})", file=sys.stderr)
        try:
            row = _one_recording(item, cfg, args.model, args.diarize)
            if row:
                rows.append(row)
        except Exception as e:  # noqa: BLE001 - keep going, log the failure
            failures += 1
            print(f"    !! skipped: {type(e).__name__}: {e}", file=sys.stderr)

    if not rows:
        print("All recordings failed.", file=sys.stderr)
        return 1

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(f"\nWrote {len(df)} rows x {df.shape[1]} cols -> {out} "
          f"({failures} skipped)", file=sys.stderr)
    print(f"Label counts:\n{df['label'].value_counts().to_string()}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
