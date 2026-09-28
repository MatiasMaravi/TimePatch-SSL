"""Report participant, window, and class counts after label filtering.

Designed for the reproducibility table requested for 30-s and 60-s settings.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np


def dataset_counts(y, groups):
    y = np.asarray(y).astype(int)
    groups = np.asarray(groups)
    return {
        "participants": int(len(np.unique(groups))),
        "windows": int(len(y)),
        "non_stress_windows": int(np.sum(y == 0)),
        "stress_windows": int(np.sum(y == 1)),
        "non_stress_fraction": float(np.mean(y == 0)) if len(y) else 0.0,
        "stress_fraction": float(np.mean(y == 1)) if len(y) else 0.0,
    }


def partition_counts(y, groups, indices):
    idx = np.asarray(indices, dtype=int)
    return dataset_counts(np.asarray(y)[idx], np.asarray(groups)[idx])


def cross_dataset_counts(y_source, g_source, fit_idx, val_idx, y_target, g_target):
    return {
        "source_training": partition_counts(y_source, g_source, fit_idx),
        "source_validation": partition_counts(y_source, g_source, val_idx),
        "target_evaluation": dataset_counts(y_target, g_target),
    }


def counts_for_windows(configurations):
    """Accept {30: (y,groups), 60: (y,groups)} and return JSON-ready counts."""
    return {
        str(window): dataset_counts(y, groups)
        for window, (y, groups) in configurations.items()
    }


def save_counts(counts, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(counts, indent=2), encoding="utf-8")
