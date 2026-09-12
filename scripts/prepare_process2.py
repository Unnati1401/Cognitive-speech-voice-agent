#!/usr/bin/env python
"""Phase 6, step 0: build a manifest from PROCESS-2 via the datasets library.

Uses `datasets.load_dataset("CognoSpeak/PROCESS-2")` (audio is cached under
~/.cache/huggingface, not in this repo). The dataset's `label` column is the
PARTICIPANT id, not the diagnosis, so we join diagnosis/MMSE/split from the
dataset's metadata.csv. Output columns match build_features.py's manifest mode:
    audio_path, speaker_id, label, mmse, task, split

    # needs HF access + HF_TOKEN in .env
    uv run python scripts/prepare_process2.py --limit 20 --out data/processed/trial.csv
    uv run python scripts/prepare_process2.py --out data/processed/process2_manifest.csv
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

from cognitive_speech_agent.config import load_config  # loads .env -> HF_TOKEN

REPO_ID = "CognoSpeak/PROCESS-2"
TASK_MAP = {"CTD": "picture_description", "SFT": "open_prompt", "PFT": "open_prompt"}
CACHE_WAVS = Path("data/raw/process2_cache")   # only used if a row has bytes but no path


def _pick_col(cols: dict, *names):
    for n in names:
        if n in cols:
            return cols[n]
    return None


def _id_variants(s: str) -> set[str]:
    """Candidate join keys so 'PROCESS-2_rec__001', '001' and '1' all match."""
    import re
    s = str(s).strip()
    keys = {s, s.lower()}
    digits = re.findall(r"\d+", s)
    if digits:
        keys.add(digits[-1])                 # '001'
        keys.add(str(int(digits[-1])))       # '1'
    return keys


def _metadata_lookup(token: str | None) -> dict:
    """participant_id -> {diagnosis, mmse, split}, auto-finding the metadata CSV."""
    from huggingface_hub import hf_hub_download, list_repo_files

    files = list_repo_files(REPO_ID, repo_type="dataset", token=token)
    csvs = [f for f in files if f.lower().endswith(".csv")]
    pick = next((f for f in csvs if "metadata" in f.lower()), csvs[0] if csvs else None)
    if pick is None:
        raise FileNotFoundError(
            f"No CSV metadata found in {REPO_ID}. Repo files: {files[:60]}")
    print(f"Using metadata file: {pick}", file=sys.stderr)

    csv = hf_hub_download(REPO_ID, pick, repo_type="dataset", token=token)
    meta = pd.read_csv(csv)
    cols = {c.lower(): c for c in meta.columns}
    c_id = _pick_col(cols, "dir_name", "ids", "id", "participant", "participant_id",
                     "speaker", "speaker_id", "file_name", "filename")
    c_dx = _pick_col(cols, "diagnosis", "label", "class", "group", "dx")
    c_mm = _pick_col(cols, "mmse")
    c_sp = _pick_col(cols, "split", "partition")
    if c_id is None or c_dx is None:
        raise ValueError(f"Unrecognized metadata columns: {list(meta.columns)}")

    out = {}
    for _, r in meta.iterrows():
        rec = {
            "diagnosis": str(r[c_dx]),
            "mmse": r.get(c_mm, "") if c_mm else "",
            "split": str(r.get(c_sp, "")) if c_sp else "",
        }
        for k in _id_variants(str(r[c_id])):
            out.setdefault(k, rec)
    return out


def _parse(path_or_name: str):
    """From '.../PROCESS-2_rec__001__CTD.wav' -> ('PROCESS-2_rec__001', 'CTD')."""
    stem = Path(path_or_name).stem                 # PROCESS-2_rec__001__CTD
    if "__" not in stem:
        return None, None
    code = stem.split("__")[-1]                     # CTD / SFT / PFT
    pid = stem[: -(len(code) + 2)]                  # strip '__CODE'
    return pid, code


def _resolve_path(audio: dict, fallback_name: str) -> str | None:
    """Return a local wav path; write bytes to a cache file if only bytes exist."""
    p = audio.get("path")
    if p and os.path.exists(p):
        return p
    data = audio.get("bytes")
    if data:
        CACHE_WAVS.mkdir(parents=True, exist_ok=True)
        out = CACHE_WAVS / fallback_name
        if not out.exists():
            out.write_bytes(data)
        return str(out)
    return p  # may be a non-local path; caller will validate


def main(argv: list[str] | None = None) -> int:
    load_config()
    p = argparse.ArgumentParser(description="Build the PROCESS-2 manifest via datasets (Phase 6).")
    p.add_argument("--out", default="data/processed/process2_manifest.csv")
    p.add_argument("--task", default="CTD", choices=["CTD", "SFT", "PFT"])
    p.add_argument("--all-tasks", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="cap rows (0 = all)")
    args = p.parse_args(argv)

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    from datasets import Audio, load_dataset

    print("Loading dataset (first run caches audio under ~/.cache/huggingface) …", file=sys.stderr)
    ds = load_dataset(REPO_ID, split="train", token=token)
    ds = ds.cast_column("audio", Audio(decode=False))   # get file paths, don't decode arrays
    print(f"Dataset columns: {ds.column_names}", file=sys.stderr)
    label_names = ds.features["label"].names if hasattr(ds.features["label"], "names") else None

    meta = _metadata_lookup(token)
    wanted = {args.task} if not args.all_tasks else {"CTD", "SFT", "PFT"}

    rows, skipped = [], 0
    for i, ex in enumerate(ds):
        if args.limit and len(rows) >= args.limit:
            break
        audio = ex["audio"]
        name = Path(audio.get("path") or "").name
        pid, code = _parse(name)
        if pid is None and label_names is not None:      # fall back to the class label
            pid = label_names[ex["label"]]
        if code is None or code not in wanted:
            skipped += 1
            continue
        m = next((meta[k] for k in _id_variants(pid) if k in meta), None)
        if m is None:
            skipped += 1
            continue
        path = _resolve_path(audio, f"{pid}__{code}.wav")
        if not path or not os.path.exists(path):
            skipped += 1
            continue
        rows.append({
            "audio_path": path, "speaker_id": pid, "label": m["diagnosis"],
            "mmse": m["mmse"], "task": TASK_MAP[code], "split": m["split"],
        })

    if not rows:
        sample_pid = _parse(Path(ds[0]["audio"].get("path") or "").name)[0]
        print("No matching rows. Debug:", file=sys.stderr)
        print(f"  example parsed participant id: {sample_pid!r}", file=sys.stderr)
        print(f"  example metadata keys: {list(meta.keys())[:5]}", file=sys.stderr)
        print("  -> if these don't match, paste them and I'll fix the join.", file=sys.stderr)
        return 1

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(f"\nWrote {len(df)} rows -> {out} ({skipped} skipped)", file=sys.stderr)
    print(f"Labels:\n{df['label'].value_counts().to_string()}", file=sys.stderr)
    print(f"Splits:\n{df['split'].value_counts().to_string()}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())