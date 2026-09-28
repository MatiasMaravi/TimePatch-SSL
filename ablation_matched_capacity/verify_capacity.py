"""Verification script to audit and report parameter parity across the 3 architectures:
Standard vs. Hierarchical vs. Standard-Matched.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# --- Autodetectar la raíz del proyecto y TimeMAE en sys.path ---
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TIMEMAE_PATH = Path("C:/Users/USUARIO/Desktop/PFC/PFC2")
if TIMEMAE_PATH.exists() and str(TIMEMAE_PATH) not in sys.path:
    sys.path.insert(0, str(TIMEMAE_PATH))
# -------------------------------------------------------------

import pandas as pd

from src.models import count_trainable_parameters
from ablation_matched_capacity.models import (
    build_model,
    calculate_matched_depth,
)


def get_model_param_summary(model):
    total = int(sum(p.numel() for p in model.parameters()))
    trainable = int(count_trainable_parameters(model))
    encoder = int(sum(
        p.numel() for name, p in model.named_parameters()
        if not name.startswith("cls_head") and not name.startswith("ssl_head")
    ))
    return {
        "total": total,
        "trainable": trainable,
        "encoder": encoder,
    }


def audit_configuration(name, in_ch, patch_len, emb_dim, depth):
    m_std = build_model("standard", in_ch=in_ch, patch_len=patch_len, emb_dim=emb_dim, depth=depth)
    m_hie = build_model("hierarchical", in_ch=in_ch, patch_len=patch_len, emb_dim=emb_dim, depth=depth)
    m_mat = build_model("standard_matched", in_ch=in_ch, patch_len=patch_len, emb_dim=emb_dim, depth=depth)

    p_std = get_model_param_summary(m_std)
    p_hie = get_model_param_summary(m_hie)
    p_mat = get_model_param_summary(m_mat)

    rel_diff = ((p_mat["trainable"] - p_hie["trainable"]) / p_hie["trainable"]) * 100.0

    return {
        "configuration": name,
        "emb_dim": emb_dim,
        "base_depth": depth,
        "matched_depth": m_mat.matched_depth,
        "patch_len": patch_len,
        "standard_trainable": p_std["trainable"],
        "hierarchical_trainable": p_hie["trainable"],
        "standard_matched_trainable": p_mat["trainable"],
        "diff_matched_vs_hierarchical": p_mat["trainable"] - p_hie["trainable"],
        "rel_diff_pct": round(rel_diff, 2),
        "parity_ratio": round(p_mat["trainable"] / p_hie["trainable"], 4),
    }


def run_full_audit(csv_path="configs/selected_hyperparameters.csv"):
    rows = []
    # 1. Base default configuration
    rows.append(audit_configuration("Base Default", in_ch=3, patch_len=8, emb_dim=128, depth=4))

    # 2. Selected hyperparameter configurations if file exists
    path = Path(csv_path)
    if path.exists():
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            cfg_name = f"{r['dataset']} {r['window_seconds']}s"
            rows.append(audit_configuration(
                name=cfg_name,
                in_ch=3,
                patch_len=int(r["patch_len"]),
                emb_dim=int(r["emb_dim"]),
                depth=int(r["depth"]),
            ))

    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit capacity parity across architectures.")
    parser.add_argument(
        "--save-csv",
        type=str,
        default="results/tables/ablation_matched_capacity/ablation_matched_capacity_parameters.csv",
    )
    args = parser.parse_args()

    results_df = run_full_audit()
    print("\n" + "=" * 110)
    print("CAPACITY-MATCHED ABLATION AUDIT (Reviewer Point 5)")
    print("=" * 110)
    print(results_df.to_string(index=False))
    print("=" * 110 + "\n")

    if args.save_csv:
        out_path = Path(args.save_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(out_path, index=False)
        print(f"Results saved to: {out_path.resolve()}")

        # Mantener espejo en results/tables/ para retrocompatibilidad
        legacy_path = PROJECT_ROOT / "results" / "tables" / "ablation_matched_capacity_parameters.csv"
        if legacy_path.resolve() != out_path.resolve():
            legacy_path.parent.mkdir(parents=True, exist_ok=True)
            results_df.to_csv(legacy_path, index=False)
        print()
