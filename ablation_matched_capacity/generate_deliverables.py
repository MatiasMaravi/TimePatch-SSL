"""Generate all Reviewer 3 deliverables for the Capacity-Matched Ablation Study:
1. Extended Table 7 (WESAD & Stress-Predict, 30s & 60s) with exact parameter counts and parity ratios.
2. Parameter count summary table across all configurations.
3. Paired statistical comparisons (Hierarchical vs. Standard-Matched, Hierarchical vs. Standard).
4. Comparative bar charts (Stress-Predict 30s & 60s, and combined WESAD + Stress-Predict).
"""
from __future__ import annotations

import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

for candidate_timemae in [
    PROJECT_ROOT / "notebooks" / "TimeMAE",
    Path("C:/Users/USUARIO/Desktop/TimePatch-SSL/notebooks/TimeMAE"),
    Path("C:/Users/USUARIO/Desktop/PFC/PFC2"),
]:
    if candidate_timemae.exists() and str(candidate_timemae) not in sys.path:
        sys.path.insert(0, str(candidate_timemae))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from ablation_matched_capacity.verify_capacity import run_full_audit


VARIANT_LABELS = {
    "standard": "Standard",
    "hierarchical": "Hierarchical",
    "standard_matched": "Standard-Matched",
}


def build_extended_table7(tables_dir: Path, param_df: pd.DataFrame) -> pd.DataFrame:
    """Consolidate WESAD and Stress-Predict (30s and 60s) ablation results into Extended Table 7."""
    configs = [
        ("WESAD", "wesad", 30, "WESAD 30s"),
        ("WESAD", "wesad", 60, "WESAD 60s"),
        ("Stress-Predict", "stress_predict", 30, "Stress-Predict / Stress-ID 30s"),
        ("Stress-Predict", "stress_predict", 60, "Stress-Predict / Stress-ID 60s"),
    ]
    rows = []
    for ds_label, ds_key, win, param_cfg_name in configs:
        res_csv = tables_dir / f"ablation_matched_capacity_results_{ds_key}_{win}s.csv"
        if not res_csv.exists():
            continue
        df = pd.read_csv(res_csv)
        p_row = param_df[param_df["configuration"] == param_cfg_name].iloc[0]
        param_map = {
            "standard": int(p_row["standard_trainable"]),
            "hierarchical": int(p_row["hierarchical_trainable"]),
            "standard_matched": int(p_row["standard_matched_trainable"]),
        }
        depth_map = {
            "standard": int(p_row["base_depth"]),
            "hierarchical": int(p_row["base_depth"]),
            "standard_matched": int(p_row["matched_depth"]),
        }
        hie_params = param_map["hierarchical"]

        for _, r in df.iterrows():
            v = r["variant"]
            v_params = param_map.get(v, int(r.get("trainable_params", 0)))
            rel_vs_hie = ((v_params - hie_params) / hie_params) * 100.0
            rows.append({
                "Dataset": ds_label,
                "Window (s)": win,
                "Model": VARIANT_LABELS.get(v, v),
                "Global Depth": depth_map.get(v, ""),
                "Trainable Params": v_params,
                "Rel. Diff vs Hierarchical (%)": round(rel_vs_hie, 2),
                "Participants (n)": int(r["n_participants"]),
                "Accuracy (Mean ± Std)": f"{r['accuracy_mean']:.4f} ± {r['accuracy_std']:.4f}",
                "F1-Score (Mean ± Std)": f"{r['f1_mean']:.4f} ± {r['f1_std']:.4f}",
                "Balanced Acc (Mean ± Std)": f"{r['balanced_accuracy_mean']:.4f} ± {r['balanced_accuracy_std']:.4f}",
                "AUROC (Mean ± Std)": f"{r['auroc_mean']:.4f} ± {r['auroc_std']:.4f}",
                "accuracy_mean": r["accuracy_mean"],
                "accuracy_std": r["accuracy_std"],
                "f1_mean": r["f1_mean"],
                "f1_std": r["f1_std"],
                "balanced_accuracy_mean": r["balanced_accuracy_mean"],
                "balanced_accuracy_std": r["balanced_accuracy_std"],
                "auroc_mean": r["auroc_mean"],
                "auroc_std": r["auroc_std"],
            })
    return pd.DataFrame(rows)


def compute_paired_comparisons(tables_dir: Path) -> pd.DataFrame:
    """Compute participant-level paired differences and Wilcoxon signed-rank tests."""
    configs = [
        ("WESAD", "wesad", 30),
        ("WESAD", "wesad", 60),
        ("Stress-Predict", "stress_predict", 30),
        ("Stress-Predict", "stress_predict", 60),
    ]
    comp_rows = []
    for ds_label, ds_key, win in configs:
        part_csv = tables_dir / f"ablation_matched_capacity_participants_{ds_key}_{win}s.csv"
        if not part_csv.exists():
            continue
        df = pd.read_csv(part_csv)
        for metric in ("accuracy", "f1", "balanced_accuracy", "auroc"):
            piv = df.pivot(index="participant", columns="variant", values=metric)
            if not {"standard", "hierarchical", "standard_matched"}.issubset(piv.columns):
                continue
            piv = piv.dropna(subset=["standard", "hierarchical", "standard_matched"])
            n = len(piv)

            for pair_name, col_a, col_b in [
                ("Hierarchical vs. Standard-Matched", "hierarchical", "standard_matched"),
                ("Hierarchical vs. Standard", "hierarchical", "standard"),
                ("Standard-Matched vs. Standard", "standard_matched", "standard"),
            ]:
                diff = piv[col_a] - piv[col_b]
                mean_diff = float(np.mean(diff))
                std_diff = float(np.std(diff, ddof=1)) if n > 1 else 0.0
                try:
                    if np.allclose(diff, 0):
                        stat, pval = 0.0, 1.0
                    else:
                        res = wilcoxon(piv[col_a], piv[col_b], zero_method="wilcox", alternative="two-sided")
                        stat, pval = float(res.statistic), float(res.pvalue)
                except Exception:
                    stat, pval = np.nan, np.nan

                comp_rows.append({
                    "Dataset": ds_label,
                    "Window (s)": win,
                    "Comparison": pair_name,
                    "Metric": metric,
                    "n": n,
                    "Mean_A": round(float(piv[col_a].mean()), 4),
                    "Mean_B": round(float(piv[col_b].mean()), 4),
                    "Mean_Diff (A - B)": round(mean_diff, 4),
                    "Std_Diff": round(std_diff, 4),
                    "Wilcoxon_stat": round(stat, 4) if not np.isnan(stat) else np.nan,
                    "p_value": round(pval, 4) if not np.isnan(pval) else np.nan,
                })
    return pd.DataFrame(comp_rows)


def plot_ablation_comparison(ext_df: pd.DataFrame, figures_dir: Path):
    """Create publication-quality bar plots comparing Standard, Hierarchical, and Standard-Matched."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("accuracy", "Accuracy"),
        ("f1", "F1-Score"),
        ("balanced_accuracy", "Balanced Accuracy"),
        ("auroc", "AUROC"),
    ]
    colors = {
        "Standard": "#4C72B0",
        "Hierarchical": "#55A868",
        "Standard-Matched": "#C44E52",
    }

    # 1. Stress-Predict Dedicated Plot (30s and 60s)
    sp_df = ext_df[ext_df["Dataset"] == "Stress-Predict"]
    if not sp_df.empty:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
        for ax_idx, win in enumerate([30, 60]):
            ax = axes[ax_idx]
            sub = sp_df[sp_df["Window (s)"] == win]
            x = np.arange(len(metrics))
            width = 0.25

            for m_idx, model_name in enumerate(["Standard", "Hierarchical", "Standard-Matched"]):
                m_row = sub[sub["Model"] == model_name]
                if m_row.empty:
                    continue
                m_row = m_row.iloc[0]
                means = np.array([m_row[f"{k}_mean"] for k, _ in metrics], dtype=float)
                stds = np.array([m_row[f"{k}_std"] for k, _ in metrics], dtype=float)
                lower_err = stds
                upper_err = np.minimum(stds, np.maximum(0.0, 1.0 - means))
                params_m = m_row["Trainable Params"] / 1e6
                label = f"{model_name} ({params_m:.2f}M)"

                bars = ax.bar(
                    x + (m_idx - 1) * width,
                    means,
                    width,
                    yerr=[lower_err, upper_err],
                    capsize=4,
                    label=label,
                    color=colors[model_name],
                    alpha=0.9,
                    edgecolor="black",
                    linewidth=0.7,
                    error_kw={"elinewidth": 1.1, "alpha": 0.75},
                )
                for bar, val in zip(bars, means):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        bar.get_height() + 0.012,
                        f"{val:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=8.5,
                        fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.88),
                    )

            ax.set_title(f"Stress-Predict — {win}s Window (n={int(sub['Participants (n)'].iloc[0])})", fontsize=12, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels([label for _, label in metrics], fontsize=10)
            ax.set_ylim(0.55, 1.05)
            ax.grid(axis="y", linestyle="--", alpha=0.5)
            ax.legend(loc="lower right", frameon=True, fontsize=9.5)
            if ax_idx == 0:
                ax.set_ylabel("Score (Mean ± SD across participants)", fontsize=11)

        fig.suptitle("Capacity-Matched Ablation on Stress-Predict (Experiment 1 Supervised LOGO Protocol)", fontsize=13.5, fontweight="bold", y=0.99)
        plt.tight_layout()
        sp_png = figures_dir / "ablation_matched_capacity_stress_predict.png"
        sp_pdf = figures_dir / "ablation_matched_capacity_stress_predict.pdf"
        fig.savefig(sp_png, dpi=300, bbox_inches="tight")
        fig.savefig(sp_pdf, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved Stress-Predict comparison plot to: {sp_png}")

    # 2. Combined 2x2 Plot (WESAD & Stress-Predict, 30s & 60s)
    if len(ext_df["Dataset"].unique()) >= 2:
        fig, axes = plt.subplots(2, 2, figsize=(14, 9.5), sharey=True)
        panels = [
            (0, 0, "WESAD", 30),
            (0, 1, "WESAD", 60),
            (1, 0, "Stress-Predict", 30),
            (1, 1, "Stress-Predict", 60),
        ]
        for r_idx, c_idx, ds_name, win in panels:
            ax = axes[r_idx, c_idx]
            sub = ext_df[(ext_df["Dataset"] == ds_name) & (ext_df["Window (s)"] == win)]
            if sub.empty:
                continue
            x = np.arange(len(metrics))
            width = 0.25
            for m_idx, model_name in enumerate(["Standard", "Hierarchical", "Standard-Matched"]):
                m_row = sub[sub["Model"] == model_name]
                if m_row.empty:
                    continue
                m_row = m_row.iloc[0]
                means = np.array([m_row[f"{k}_mean"] for k, _ in metrics], dtype=float)
                stds = np.array([m_row[f"{k}_std"] for k, _ in metrics], dtype=float)
                lower_err = stds
                upper_err = np.minimum(stds, np.maximum(0.0, 1.0 - means))
                params_m = m_row["Trainable Params"] / 1e6
                label = f"{model_name} ({params_m:.2f}M)"

                bars = ax.bar(
                    x + (m_idx - 1) * width,
                    means,
                    width,
                    yerr=[lower_err, upper_err],
                    capsize=3.5,
                    label=label,
                    color=colors[model_name],
                    alpha=0.9,
                    edgecolor="black",
                    linewidth=0.7,
                    error_kw={"elinewidth": 1.1, "alpha": 0.75},
                )
                for bar, val in zip(bars, means):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        bar.get_height() + 0.012,
                        f"{val:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.88),
                    )

            ax.set_title(f"{ds_name} — {win}s Window (n={int(sub['Participants (n)'].iloc[0])})", fontsize=11.5, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels([label for _, label in metrics], fontsize=10)
            ax.set_ylim(0.45, 1.05)
            ax.grid(axis="y", linestyle="--", alpha=0.5)
            ax.legend(loc="lower right", frameon=True, fontsize=9)
            if c_idx == 0:
                ax.set_ylabel("Score (Mean ± SD)", fontsize=10.5)

        fig.suptitle("Table 7 Extension: Capacity-Matched Ablation Across WESAD and Stress-Predict", fontsize=14, fontweight="bold", y=0.995)
        plt.tight_layout()
        all_png = figures_dir / "ablation_matched_capacity_all_datasets.png"
        all_pdf = figures_dir / "ablation_matched_capacity_all_datasets.pdf"
        fig.savefig(all_png, dpi=300, bbox_inches="tight")
        fig.savefig(all_pdf, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved combined 4-panel comparison plot to: {all_png}")


def main():
    tables_dir = PROJECT_ROOT / "results" / "tables" / "ablation_matched_capacity"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    param_df = run_full_audit()
    param_csv = tables_dir / "ablation_matched_capacity_parameters.csv"
    param_df.to_csv(param_csv, index=False)

    ext_df = build_extended_table7(tables_dir, param_df)
    if not ext_df.empty:
        display_cols = [
            "Dataset",
            "Window (s)",
            "Model",
            "Global Depth",
            "Trainable Params",
            "Rel. Diff vs Hierarchical (%)",
            "Participants (n)",
            "Accuracy (Mean ± Std)",
            "F1-Score (Mean ± Std)",
            "Balanced Acc (Mean ± Std)",
            "AUROC (Mean ± Std)",
        ]
        table7_csv = tables_dir / "table7_extended_ablation_summary.csv"
        ext_df[display_cols].to_csv(table7_csv, index=False)
        print(f"\nExtended Table 7 saved to: {table7_csv}")
        print(ext_df[display_cols].to_string(index=False))

        plot_ablation_comparison(ext_df, figures_dir)

    comp_df = compute_paired_comparisons(tables_dir)
    if not comp_df.empty:
        comp_csv = tables_dir / "ablation_matched_capacity_paired_comparisons.csv"
        comp_df.to_csv(comp_csv, index=False)
        print(f"\nPaired statistical comparisons saved to: {comp_csv}")
        print(comp_df.to_string(index=False))


if __name__ == "__main__":
    main()
