"""Recompute the participant-level paired statistics reported in Table 4."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.statistics import wilcoxon_compare

METRICS = ("f1", "balanced_accuracy", "auroc")
BONFERRONI_ALPHA = 0.05 / 6


def align_by_participant(ssl_records, supervised_records, metric):
    ssl = {str(r["participant"]): float(r[metric]) for r in ssl_records}
    sup = {str(r["participant"]): float(r[metric]) for r in supervised_records}
    ids = sorted(set(ssl).intersection(sup))
    if not ids:
        raise ValueError("No common participant IDs were found.")
    return ids, np.array([ssl[i] for i in ids]), np.array([sup[i] for i in ids])


def compare_records(ssl_records, supervised_records, metric):
    ids, a, b = align_by_participant(ssl_records, supervised_records, metric)
    result = wilcoxon_compare(a, b)
    result.update({
        "metric": metric,
        "n_participants": len(ids),
        "participants": ids,
        "ssl_mean": float(np.mean(a)),
        "supervised_mean": float(np.mean(b)),
        "delta_ssl_minus_supervised": float(np.mean(a - b)),
        "bonferroni_alpha": BONFERRONI_ALPHA,
        "significant": bool(result["p_value"] < BONFERRONI_ALPHA),
    })
    return result


def build_table4(dataset_records):
    """dataset_records: {dataset: {"ssl": [...], "supervised": [...]}}."""
    rows = []
    for dataset, records in dataset_records.items():
        for metric in METRICS:
            row = {"dataset": dataset}
            row.update(compare_records(records["ssl"], records["supervised"], metric))
            rows.append(row)
    return pd.DataFrame(rows)


def save_table4(df, output_csv):
    path = Path(output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
