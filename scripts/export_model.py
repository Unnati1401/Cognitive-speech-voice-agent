#!/usr/bin/env python
"""Export the trained LOGISTIC scorer to a version-proof, DUA-safe JSON.

Reads models/scorer.joblib + models/norms.json (trained locally on PROCESS-2)
and writes models/scorer.json containing ONLY:
  - the logistic weights + the standardizer's mean/scale,
  - summary percentile bands per marker (for flagging), NOT raw participant values.

This JSON has no pickle and no scikit-learn dependency, so it loads on any
environment (fixes the Space's version-mismatch crashes), and it carries no
redistributable participant data (safe under the PROCESS-2 DUA).

    uv run python scripts/train_scorer.py --features data/processed/features.csv --model logistic
    uv run python scripts/export_model.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import joblib

from cognitive_speech_agent.config import load_config


def main() -> int:
    cfg = load_config()
    models = Path(cfg.get("paths", {}).get("models", "models"))
    bundle = joblib.load(models / "scorer.joblib")
    model = bundle["model"]
    feats = bundle["feature_names"]

    if not hasattr(model, "named_steps"):
        raise SystemExit("export_model expects the logistic pipeline; "
                         "retrain with:  train_scorer.py --model logistic")
    scaler = model.named_steps.get("standardscaler")
    clf = model.named_steps.get("logisticregression")
    if scaler is None or clf is None:
        raise SystemExit("Could not find StandardScaler + LogisticRegression in the pipeline.")

    controls = json.loads((models / "norms.json").read_text()).get("controls", {})
    qs = list(range(0, 101, 5))                      # 0,5,...,100 -> summary bands only
    grid = {}
    for f in feats:
        vals = controls.get(f, [])
        if vals:
            arr = np.asarray(vals, dtype=float)
            grid[f] = {"q": qs, "v": [float(np.percentile(arr, q)) for q in qs]}

    out = {
        "type": "logistic",
        "feature_names": feats,
        "medians": bundle["medians"],
        "scaler_mean": [float(x) for x in scaler.mean_],
        "scaler_scale": [float(x) for x in scaler.scale_],
        "coef": [float(x) for x in clf.coef_[0]],
        "intercept": float(clf.intercept_[0]),
        "threshold": float(bundle["threshold"]),
        "flag_low": float(bundle["flag_low"]),
        "flag_high": float(bundle["flag_high"]),
        "norm_grid": grid,
    }
    (models / "scorer.json").write_text(json.dumps(out, indent=2))
    print(f"Wrote {models / 'scorer.json'} ({len(feats)} markers, no pickle, no raw data).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())