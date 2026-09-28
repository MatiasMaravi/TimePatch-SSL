"""Independent audit of Experiment 2 predictions and Table 2 metrics."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score,
    roc_auc_score, confusion_matrix, roc_curve
)


def audit_predictions(y_true, p_stress, threshold=0.5):
    y = np.asarray(y_true).astype(int)
    p = np.asarray(p_stress, dtype=float)
    pred = (p >= threshold).astype(int)

    result = {
        "n": int(len(y)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }
    try:
        result["auroc"] = float(roc_auc_score(y, p))
        result["auroc_inverted"] = float(roc_auc_score(y, 1.0 - p))
        fpr, tpr, thresholds = roc_curve(y, p)
        result["roc"] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": thresholds.tolist(),
        }
    except ValueError:
        result["auroc"] = None
        result["auroc_inverted"] = None
        result["roc"] = None
    return result


def per_participant_audit(y_true, p_stress, groups, threshold=0.5):
    y = np.asarray(y_true)
    p = np.asarray(p_stress)
    g = np.asarray(groups)
    rows = []
    for sid in np.unique(g):
        idx = np.flatnonzero(g == sid)
        row = {"participant": str(sid), "n_windows": int(len(idx))}
        metrics = audit_predictions(y[idx], p[idx], threshold)
        metrics.pop("roc", None)
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def audit_prediction_file(path, threshold=0.5):
    """Read JSON containing y_true, p_stress and optionally participant."""
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    result = audit_predictions(obj["y_true"], obj["p_stress"], threshold)
    if "participant" in obj:
        result["per_participant"] = per_participant_audit(
            obj["y_true"], obj["p_stress"], obj["participant"], threshold
        ).to_dict(orient="records")
    return result


def save_audit(result, output):
    Path(output).write_text(json.dumps(result, indent=2), encoding="utf-8")
