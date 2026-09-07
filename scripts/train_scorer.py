#!/usr/bin/env python
"""Phase 2, step 2: train the speaker-independent scorer on the feature table.

    uv run python scripts/train_scorer.py                       # uses data/processed/features.csv
    uv run python scripts/train_scorer.py --features my.csv --model logistic

Prints out-of-fold AUC / F1 / accuracy / confusion matrix + top markers, and
saves models/scorer.joblib and models/norms.json for Phase 3.
"""
from __future__ import annotations

import argparse

from cognitive_speech_agent.config import load_config
from cognitive_speech_agent import scoring


def main(argv: list[str] | None = None) -> int:
    cfg = load_config()
    p = argparse.ArgumentParser(description="Train the marker-based scorer (Phase 2).")
    p.add_argument("--features", default="data/processed/features.csv")
    p.add_argument("--model", choices=["xgboost", "logistic"], default=None,
                   help="override config.yaml scoring.model")
    args = p.parse_args(argv)

    if args.model:
        cfg.setdefault("scoring", {})["model"] = args.model

    scoring.train(args.features, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
