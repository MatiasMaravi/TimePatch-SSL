"""Per-class metrics for reviewer-requested uncertainty/transparency analyses."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_fscore_support, confusion_matrix,
    classification_report
)


def per_class_metrics(y_true, p_stress=None, y_pred=None, threshold=0.5):
    y = np.asarray(y_true).astype(int)
    if y_pred is None:
        if p_stress is None:
            raise ValueError("Provide p_stress or y_pred.")
        pred = (np.asarray(p_stress, dtype=float) >= threshold).astype(int)
    else:
        pred = np.asarray(y_pred).astype(int)

    precision, recall, f1, support = precision_recall_fscore_support(
        y, pred, labels=[0, 1], zero_division=0
    )
    labels = ["non_stress", "stress"]
    return pd.DataFrame({
        "class": labels,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support.astype(int),
    })


def full_report(y_true, p_stress=None, y_pred=None, threshold=0.5):
    y = np.asarray(y_true).astype(int)
    if y_pred is None:
        pred = (np.asarray(p_stress) >= threshold).astype(int)
    else:
        pred = np.asarray(y_pred).astype(int)
    return {
        "threshold": float(threshold),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y, pred, labels=[0, 1],
            target_names=["non_stress", "stress"],
            zero_division=0, output_dict=True
        ),
    }


def per_participant_per_class(y_true, p_stress, groups, threshold=0.5):
    y = np.asarray(y_true)
    p = np.asarray(p_stress)
    g = np.asarray(groups)
    frames = []
    for sid in np.unique(g):
        idx = np.flatnonzero(g == sid)
        df = per_class_metrics(y[idx], p_stress=p[idx], threshold=threshold)
        df.insert(0, "participant", str(sid))
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
