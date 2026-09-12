#!/usr/bin/env python
"""Phase 6: honest evaluation of the marker scorer on real data.

Produces, from a features CSV (built by build_features.py over PROCESS-2):
  1. Held-out metrics on the dataset's OFFICIAL train/test split (headline number).
  2. Speaker-independent cross-validation on the training portion (robustness).
  3. A marker-family ABLATION (timing / vocabulary / grammar / coherence).
  4. ROC curve + confusion matrix PNGs (if matplotlib is available).
  5. results/metrics.json and results/RESULTS.md.

    uv run python scripts/evaluate.py --features data/processed/features.csv
    uv run python scripts/evaluate.py --features data/processed/features.csv --model xgboost
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import GroupKFold, cross_val_predict

from cognitive_speech_agent.config import load_config
from cognitive_speech_agent.scoring import _build_model, _feature_columns, _scoring_cfg

FAMILIES = {"timing": "timing_", "vocabulary": "vocab_", "grammar": "gram_", "coherence": "coh_"}


def _binarize(df: pd.DataFrame, cfg: dict) -> np.ndarray:
    return (df[cfg["label_column"]].astype(str).str.lower()
            .isin(cfg["positive_labels"]).astype(int).to_numpy())


def _matrix(df: pd.DataFrame, feats: list[str], medians=None):
    X = df[feats].apply(pd.to_numeric, errors="coerce")
    medians = X.median() if medians is None else medians
    return X.fillna(medians).to_numpy(), medians


def _metrics(y, proba, thr) -> dict:
    pred = (proba >= thr).astype(int)
    return {
        "n": int(len(y)), "n_positive": int(y.sum()),
        "auc": float(roc_auc_score(y, proba)) if len(set(y)) > 1 else float("nan"),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
    }


def _cv_proba(X, y, groups, cfg) -> np.ndarray:
    n_splits = max(2, min(cfg["n_folds"], groups.nunique()))
    return cross_val_predict(_build_model(cfg["model"], y), X, y, groups=groups,
                             cv=GroupKFold(n_splits), method="predict_proba")[:, 1]


def _plots(out: Path, y, proba) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return []
    made = []
    # ROC
    fpr, tpr, _ = roc_curve(y, proba)
    auc = roc_auc_score(y, proba)
    plt.figure(figsize=(4.5, 4))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("ROC — held-out test"); plt.legend(loc="lower right"); plt.tight_layout()
    plt.savefig(out / "roc_curve.png", dpi=150); plt.close(); made.append("roc_curve.png")
    # confusion
    cm = confusion_matrix(y, (proba >= 0.5).astype(int))
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, cmap="Blues"); plt.title("Confusion — held-out test")
    plt.xticks([0, 1], ["control", "impaired"]); plt.yticks([0, 1], ["control", "impaired"])
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.tight_layout(); plt.savefig(out / "confusion_matrix.png", dpi=150)
    plt.close(); made.append("confusion_matrix.png")
    return made


def main(argv: list[str] | None = None) -> int:
    base = load_config()
    p = argparse.ArgumentParser(description="Evaluate the marker scorer (Phase 6).")
    p.add_argument("--features", default="data/processed/features.csv")
    p.add_argument("--model", choices=["xgboost", "logistic"], default=None)
    p.add_argument("--out-dir", default="results")
    args = p.parse_args(argv)
    if args.model:
        base.setdefault("scoring", {})["model"] = args.model
    cfg = _scoring_cfg(base)

    df = pd.read_csv(args.features)
    feats = _feature_columns(df)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    results: dict = {"model": cfg["model"], "n_features": len(feats), "n_rows": len(df)}

    has_split = "split" in df.columns and df["split"].astype(str).str.lower().isin(["train", "test"]).any()

    # --- 1. official held-out split (headline) ---
    plots: list[str] = []
    if has_split:
        s = df["split"].astype(str).str.lower()
        tr, te = df[s == "train"], df[s == "test"]
        Xtr, med = _matrix(tr, feats)
        Xte, _ = _matrix(te, feats, med)
        ytr, yte = _binarize(tr, cfg), _binarize(te, cfg)
        model = _build_model(cfg["model"], ytr); model.fit(Xtr, ytr)
        proba = model.predict_proba(Xte)[:, 1]
        results["official_test"] = _metrics(yte, proba, cfg["threshold"])
        plots = _plots(out, yte, proba)
        cv_df = tr
    else:
        cv_df = df

    # --- 2. speaker-independent CV (robustness) ---
    X, _ = _matrix(cv_df, feats)
    y = _binarize(cv_df, cfg)
    groups = cv_df["speaker_id"] if "speaker_id" in cv_df else pd.Series(range(len(cv_df)))
    results["cv"] = _metrics(y, _cv_proba(X, y, groups, cfg), cfg["threshold"])

    # --- 3. marker-family ablation (CV AUC using each family alone) ---
    ablation = {}
    for name, prefix in FAMILIES.items():
        cols = [c for c in feats if c.startswith(prefix)]
        if not cols:
            continue
        Xf, _ = _matrix(cv_df, cols)
        try:
            proba_f = _cv_proba(Xf, y, groups, cfg)
            ablation[name] = {"n_features": len(cols),
                              "auc": float(roc_auc_score(y, proba_f)) if len(set(y)) > 1 else float("nan")}
        except Exception as e:  # noqa: BLE001
            ablation[name] = {"n_features": len(cols), "auc": None, "error": str(e)}
    results["ablation"] = ablation

    # --- 4/5. write artifacts ---
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    _write_report(out, results, plots)
    print(json.dumps(results, indent=2))
    print(f"\nWrote {out}/metrics.json, {out}/RESULTS.md" + (f", + {plots}" if plots else ""))
    return 0


def _write_report(out: Path, r: dict, plots: list[str]) -> None:
    lines = ["# PROCESS-2 evaluation results", "",
             f"Model: **{r['model']}** · {r['n_features']} markers · {r['n_rows']} recordings", ""]
    if "official_test" in r:
        t = r["official_test"]
        lines += ["## Held-out official test split", "",
                  f"- AUC-ROC: **{t['auc']:.3f}**",
                  f"- F1: {t['f1']:.3f}  ·  Accuracy: {t['accuracy']:.3f}",
                  f"- n = {t['n']} ( +{t['n_positive']} impaired )",
                  f"- Confusion [[TN,FP],[FN,TP]]: {t['confusion_matrix']}", ""]
    c = r["cv"]
    lines += ["## Speaker-independent cross-validation (training portion)", "",
              f"- AUC-ROC: **{c['auc']:.3f}**  ·  F1: {c['f1']:.3f}  ·  Acc: {c['accuracy']:.3f}", ""]
    if r.get("ablation"):
        lines += ["## Marker-family ablation (CV AUC, each family alone)", "",
                  "| Family | # markers | AUC |", "| --- | --- | --- |"]
        for name, a in sorted(r["ablation"].items(), key=lambda kv: (kv[1].get("auc") or 0), reverse=True):
            auc = a.get("auc")
            lines.append(f"| {name} | {a['n_features']} | {auc:.3f} |" if auc is not None else
                         f"| {name} | {a['n_features']} | n/a |")
        lines.append("")
    for pth in plots:
        lines.append(f"![{pth}]({pth})")
    lines += ["", "_Screening research demo. Not a diagnosis. Speaker-independent splits throughout._"]
    (out / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())