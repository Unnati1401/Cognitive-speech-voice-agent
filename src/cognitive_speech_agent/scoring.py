"""Tool 4 - Scoring: markers -> risk score, plus norm-based flags.

Trains an interpretable classifier on the Phase 1 marker table using a
SPEAKER-INDEPENDENT split (GroupKFold by speaker), reports honest out-of-fold
metrics, fits a final model on all data, and stores healthy-control "norms" so
each marker can be flagged as typical / atypical at inference time.

Design rule: the model computes the truth; the LLM (Phase 3) only narrates it.

Artifacts written to models/:
    scorer.joblib  - {model, feature_names, medians, threshold, meta}
    norms.json     - {controls: {feature: [sorted control values]}, medians: {...}}
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# Columns that are metadata, never model features.
NON_FEATURE = {"id", "speaker_id", "label", "y", "mmse", "text", "audio_path", "task", "split"}


@dataclass
class ScoreResult:
    risk_probability: float                                   # 0..1
    label: str                                                # "elevated markers" | "within typical range"
    flagged_markers: dict[str, float] = field(default_factory=dict)   # marker -> percentile vs controls
    percentiles: dict[str, float] = field(default_factory=dict)       # every marker -> percentile


# --- helpers -----------------------------------------------------------------

def _scoring_cfg(config: dict) -> dict:
    s = (config or {}).get("scoring", {})
    return {
        "model": s.get("model", "xgboost"),
        "n_folds": int(s.get("n_folds", 5)),
        "label_column": s.get("label_column", "label"),
        "positive_labels": {str(x).lower() for x in
                            s.get("positive_labels", ["mci", "dementia", "impaired"])},
        "threshold": float(s.get("decision_threshold", 0.5)),
        "flag_low": float(s.get("flag_low_pct", 5)),
        "flag_high": float(s.get("flag_high_pct", 95)),
        "models_dir": (config or {}).get("paths", {}).get("models", "models"),
    }


def _build_model(kind: str, y: np.ndarray):
    """Interpretable, small-data-friendly classifiers."""
    if kind == "logistic":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced"),
        )
    # default: gradient-boosted trees
    from xgboost import XGBClassifier
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    spw = (n_neg / n_pos) if n_pos else 1.0
    return XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
        eval_metric="logloss", scale_pos_weight=spw, n_jobs=0,
    )


def _percentile(value: float, sorted_controls: list[float]) -> float:
    """Percentile of `value` within the healthy-control distribution (0..100)."""
    if value is None or not sorted_controls:
        return 50.0
    arr = np.asarray(sorted_controls, dtype=float)
    return float((arr <= value).mean() * 100.0)


def _feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns
            if c not in NON_FEATURE and pd.api.types.is_numeric_dtype(df[c])]


# --- training ----------------------------------------------------------------

def train(feature_table_path: str, config: dict) -> dict:
    """Train, evaluate speaker-independently, and persist the scorer + norms.

    Returns a metrics dict; also prints a human-readable summary.
    """
    from sklearn.metrics import (
        accuracy_score, confusion_matrix, f1_score, roc_auc_score,
    )
    from sklearn.model_selection import GroupKFold, cross_val_predict
    import joblib

    cfg = _scoring_cfg(config)
    df = pd.read_csv(feature_table_path)

    # labels -> binary (impaired vs control)
    y = df[cfg["label_column"]].astype(str).str.lower().isin(cfg["positive_labels"]).astype(int).to_numpy()

    # groups for the speaker-independent split (the non-negotiable)
    groups = df["speaker_id"] if "speaker_id" in df.columns else pd.Series(range(len(df)))

    feats = _feature_columns(df)
    X = df[feats].apply(pd.to_numeric, errors="coerce")
    medians = X.median(numeric_only=True)
    X = X.fillna(medians).to_numpy()

    n_groups = groups.nunique()
    n_splits = max(2, min(cfg["n_folds"], n_groups))
    model = _build_model(cfg["model"], y)

    # honest, out-of-fold predictions (never train and test on the same speaker)
    gkf = GroupKFold(n_splits=n_splits)
    oof = cross_val_predict(model, X, y, groups=groups, cv=gkf,
                            method="predict_proba")[:, 1]
    pred = (oof >= cfg["threshold"]).astype(int)

    metrics = {
        "n": int(len(y)), "n_positive": int(y.sum()), "n_features": len(feats),
        "cv_folds": n_splits, "model": cfg["model"],
        "auc": float(roc_auc_score(y, oof)) if len(set(y)) > 1 else float("nan"),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
    }

    # final model on all data (for deployment)
    model.fit(X, y)

    # norms = healthy-control distribution per feature
    ctrl_mask = y == 0
    controls = {f: sorted(pd.to_numeric(df[f], errors="coerce")
                          .fillna(medians[f])[ctrl_mask].tolist())
                for f in feats}

    models_dir = Path(cfg["models_dir"]); models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "feature_names": feats, "medians": medians.to_dict(),
         "threshold": cfg["threshold"], "flag_low": cfg["flag_low"],
         "flag_high": cfg["flag_high"], "meta": metrics},
        models_dir / "scorer.joblib",
    )
    with open(models_dir / "norms.json", "w", encoding="utf-8") as f:
        json.dump({"controls": controls, "medians": medians.to_dict()}, f)

    _print_summary(metrics, model, feats)
    return metrics


def _print_summary(metrics: dict, model, feats: list[str]) -> None:
    print("\n" + "=" * 56)
    print(f"SPEAKER-INDEPENDENT CV  ({metrics['cv_folds']} folds, "
          f"{metrics['model']}, n={metrics['n']}, +{metrics['n_positive']})")
    print("=" * 56)
    print(f"  AUC-ROC   {metrics['auc']:.3f}")
    print(f"  F1        {metrics['f1']:.3f}")
    print(f"  Accuracy  {metrics['accuracy']:.3f}")
    cm = metrics["confusion_matrix"]
    print(f"  Confusion [[TN {cm[0][0]}, FP {cm[0][1]}], [FN {cm[1][0]}, TP {cm[1][1]}]]")

    # top features by importance / |coef|
    imp = getattr(model, "feature_importances_", None)
    if imp is None and hasattr(model, "named_steps"):
        lr = model.named_steps.get("logisticregression")
        imp = np.abs(lr.coef_[0]) if lr is not None else None
    if imp is not None:
        top = sorted(zip(feats, imp), key=lambda x: x[1], reverse=True)[:10]
        print("\n  Top markers:")
        for name, w in top:
            print(f"    {name:<28} {w:.4f}")
    print(f"\n  Saved -> models/scorer.joblib, models/norms.json\n")


# --- inference ---------------------------------------------------------------

def score(markers: dict[str, float], model_path: str, norms_path: str) -> ScoreResult:
    """Return risk probability + which markers fall outside healthy-control norms."""
    import joblib

    bundle = joblib.load(model_path)
    with open(norms_path, "r", encoding="utf-8") as f:
        norms = json.load(f)

    feats = bundle["feature_names"]
    medians = bundle["medians"]
    controls = norms.get("controls", {})

    vec = np.array([[float(markers.get(f, medians.get(f, 0.0)) or medians.get(f, 0.0))
                     for f in feats]])
    proba = float(bundle["model"].predict_proba(vec)[0, 1])
    label = ("elevated markers" if proba >= bundle["threshold"]
             else "within typical range")

    percentiles = {f: _percentile(markers.get(f), controls.get(f, [])) for f in feats}
    lo, hi = bundle["flag_low"], bundle["flag_high"]
    flagged = {f: p for f, p in percentiles.items() if p <= lo or p >= hi}

    return ScoreResult(risk_probability=proba, label=label,
                       flagged_markers=flagged, percentiles=percentiles)