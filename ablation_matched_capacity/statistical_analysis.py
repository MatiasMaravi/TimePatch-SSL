"""Phase 6: Rigorous Paired Statistical Analysis of the Capacity-Matched Ablation Study.

Computes subject-level paired comparisons across WESAD (n=15) and Stress-Predict (n=33)
for 30s and 60s windows across all 4 metrics (Accuracy, F1, Balanced Accuracy, AUROC):
- Contrasts evaluated:
  1. Hierarchical vs. Standard-Matched (Delta = Hierarchical - Standard-Matched)
  2. Hierarchical vs. Standard (Delta = Hierarchical - Standard)
  3. Standard-Matched vs. Standard (Delta = Standard-Matched - Standard)
- Metrics reported per contrast:
  - Paired mean difference (Delta_mean), sample SD of difference (Delta_std, ddof=1)
  - Parametric 95% Confidence Interval of Delta (Student's t-distribution)
  - Non-parametric 95% Bootstrap Confidence Interval of Delta (10,000 resamples, seed=42)
  - Median difference and Interquartile Range (IQR = Q75 - Q25)
  - Participant win/tie/loss counts (n_pos, n_zero, n_neg)
  - Effect sizes:
    - Rank-Biserial Correlation (r_rb, consistent with src/statistics.py & Table 4)
    - Paired Cohen's d_z (Delta_mean / Delta_std)
  - Hypothesis tests:
    - Two-sided Wilcoxon signed-rank test (W statistic, raw p-value)
    - Two-sided paired Student's t-test (t statistic, raw p-value)
  - Multiple comparison corrections:
    - Bonferroni correction (within family of 4 metrics per comparison & 8 metrics across windows)
    - Holm-Bonferroni step-down adjusted p-values
- Generates publication-quality paired difference figures:
  - results/figures/ablation_paired_differences_stress_predict.{png,pdf}
  - results/figures/ablation_paired_differences_summary.{png,pdf}
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.statistics import rank_biserial


METRICS = [
    ("accuracy", "Accuracy"),
    ("f1", "F1-Score"),
    ("balanced_accuracy", "Balanced Accuracy"),
    ("auroc", "AUROC"),
]

CONTRASTS = [
    ("Hierarchical vs. Standard-Matched", "hierarchical", "standard_matched"),
    ("Hierarchical vs. Standard", "hierarchical", "standard"),
    ("Standard-Matched vs. Standard", "standard_matched", "standard"),
]


def bootstrap_ci_95(diff: np.ndarray, n_boot: int = 10000, seed: int = 42) -> tuple[float, float]:
    """Compute two-sided 95% percentile bootstrap confidence interval for the mean difference."""
    rng = np.random.default_rng(seed)
    n = len(diff)
    if n < 2:
        return (np.nan, np.nan)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = np.mean(diff[idx], axis=1)
    low, high = np.percentile(boot_means, [2.5, 97.5])
    return float(low), float(high)


def holm_bonferroni(p_values: list[float]) -> list[float]:
    """Compute Holm-Bonferroni adjusted p-values for a family of tests."""
    p_arr = np.asarray(p_values, dtype=float)
    m = len(p_arr)
    if m == 0:
        return []
    order = np.argsort(p_arr)
    adj = np.empty(m, dtype=float)
    running_max = 0.0
    for rank_idx, orig_idx in enumerate(order):
        p_val = p_arr[orig_idx]
        if np.isnan(p_val):
            adj[orig_idx] = np.nan
            continue
        scaled = p_val * (m - rank_idx)
        running_max = max(running_max, scaled)
        adj[orig_idx] = min(1.0, running_max)
    return adj.tolist()


def interpret_effect_size(r_rb: float, d_z: float) -> str:
    """Classify magnitude of effect size using standard thresholds for |r_rb|."""
    abs_r = abs(r_rb)
    if np.isnan(abs_r):
        return "N/A"
    if abs_r < 0.10:
        return "Negligible (|r_rb| < 0.10)"
    if abs_r < 0.30:
        return "Small (0.10 <= |r_rb| < 0.30)"
    if abs_r < 0.50:
        return "Medium (0.30 <= |r_rb| < 0.50)"
    return "Large (|r_rb| >= 0.50)"


def run_paired_statistical_analysis(tables_dir: Path) -> pd.DataFrame:
    """Compute comprehensive paired statistics across all datasets, windows, contrasts, and metrics."""
    configs = [
        ("Stress-Predict", 30, "ablation_matched_capacity_participants_stress_predict_30s.csv"),
        ("Stress-Predict", 60, "ablation_matched_capacity_participants_stress_predict_60s.csv"),
        ("WESAD", 30, "ablation_matched_capacity_participants_wesad_30s.csv"),
        ("WESAD", 60, "ablation_matched_capacity_participants_wesad_60s.csv"),
    ]

    rows = []
    for ds_name, win, fname in configs:
        fpath = tables_dir / fname
        if not fpath.exists():
            continue
        df = pd.read_csv(fpath)

        for contrast_label, col_a, col_b in CONTRASTS:
            family_rows = []
            for m_key, m_label in METRICS:
                piv = df.pivot(index="participant", columns="variant", values=m_key).dropna(subset=[col_a, col_b])
                piv = piv.sort_index()
                a = piv[col_a].to_numpy(dtype=float)
                b = piv[col_b].to_numpy(dtype=float)
                diff = a - b
                n = len(diff)

                mean_a = float(np.mean(a))
                mean_b = float(np.mean(b))
                mean_diff = float(np.mean(diff))
                std_diff = float(np.std(diff, ddof=1)) if n > 1 else 0.0
                se_diff = std_diff / np.sqrt(n) if n > 1 else 0.0

                # Parametric 95% CI (Student's t)
                t_crit = float(stats.t.ppf(0.975, df=n - 1)) if n > 1 else 0.0
                ci95_low = mean_diff - t_crit * se_diff
                ci95_high = mean_diff + t_crit * se_diff

                # Bootstrap 95% CI
                boot_low, boot_high = bootstrap_ci_95(diff, n_boot=10000, seed=42)

                # Non-parametric summary
                median_diff = float(np.median(diff))
                q25, q75 = np.percentile(diff, [25, 75])
                iqr_diff = float(q75 - q25)

                # Wins / Ties / Losses
                eps = 1e-9
                n_pos = int(np.sum(diff > eps))
                n_neg = int(np.sum(diff < -eps))
                n_tie = int(n - n_pos - n_neg)

                # Effect sizes
                r_rb = rank_biserial(a, b)
                cohen_dz = float(mean_diff / std_diff) if std_diff > 1e-12 else 0.0

                # Paired Wilcoxon signed-rank test
                nonzero = diff[np.abs(diff) > eps]
                if len(nonzero) > 0:
                    res_w = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
                    w_stat = float(res_w.statistic)
                    p_wilcoxon = float(res_w.pvalue)
                else:
                    w_stat, p_wilcoxon = 0.0, 1.0

                # Paired Student's t-test
                if std_diff > 1e-12 and n > 1:
                    res_t = stats.ttest_rel(a, b, alternative="two-sided")
                    t_stat = float(res_t.statistic)
                    p_ttest = float(res_t.pvalue)
                else:
                    t_stat, p_ttest = 0.0, 1.0

                family_rows.append({
                    "Dataset": ds_name,
                    "Window (s)": win,
                    "Comparison": contrast_label,
                    "Model_A": col_a,
                    "Model_B": col_b,
                    "Metric": m_key,
                    "Metric_Label": m_label,
                    "n": n,
                    "Mean_A": round(mean_a, 4),
                    "Mean_B": round(mean_b, 4),
                    "Mean_Diff (A - B)": round(mean_diff, 4),
                    "Std_Diff": round(std_diff, 4),
                    "CI95_Low": round(ci95_low, 4),
                    "CI95_High": round(ci95_high, 4),
                    "Boot_CI95_Low": round(boot_low, 4),
                    "Boot_CI95_High": round(boot_high, 4),
                    "Median_Diff": round(median_diff, 4),
                    "IQR_Diff": round(iqr_diff, 4),
                    "Wins_A": n_pos,
                    "Ties": n_tie,
                    "Losses_A": n_neg,
                    "Cohen_dz": round(cohen_dz, 4),
                    "Rank_Biserial_r": round(r_rb, 4),
                    "Effect_Magnitude": interpret_effect_size(r_rb, cohen_dz),
                    "Wilcoxon_stat": round(w_stat, 4),
                    "p_value": round(p_wilcoxon, 4),
                    "p_value_raw": p_wilcoxon,
                    "t_stat": round(t_stat, 4),
                    "p_ttest": round(p_ttest, 4),
                })

            # Within-window family multiple-testing correction (4 metrics)
            raw_ps = [r["p_value_raw"] for r in family_rows]
            holm_4 = holm_bonferroni(raw_ps)
            for r, p_h4 in zip(family_rows, holm_4):
                r["p_bonferroni_4"] = round(min(1.0, r["p_value_raw"] * 4), 4)
                r["p_holm_4"] = round(p_h4, 4)
            rows.extend(family_rows)

    full_df = pd.DataFrame(rows)
    if full_df.empty:
        return full_df

    # Also compute dataset-level family correction across both windows (8 tests per dataset x contrast: 2 windows x 4 metrics)
    # and Table-4 style correction (6 tests: 2 windows x 3 metrics [F1, BAcc, AUROC], alpha = 0.05 / 6 = 0.0083)
    full_df["p_bonferroni_8"] = np.nan
    full_df["p_holm_8"] = np.nan
    full_df["Bonferroni_Alpha_8"] = round(0.05 / 8.0, 4)
    full_df["Bonferroni_Alpha_6 (Table4)"] = round(0.05 / 6.0, 4)

    for (ds_name, contrast_label), grp in full_df.groupby(["Dataset", "Comparison"]):
        idx = grp.index
        raw_ps = grp["p_value_raw"].tolist()
        holm_8 = holm_bonferroni(raw_ps)
        full_df.loc[idx, "p_bonferroni_8"] = [round(min(1.0, p * len(raw_ps)), 4) for p in raw_ps]
        full_df.loc[idx, "p_holm_8"] = [round(ph, 4) for ph in holm_8]

    full_df["Significant_Uncorrected (p<0.05)"] = full_df["p_value_raw"] < 0.05
    full_df["Significant_Holm8 (p_adj<0.05)"] = full_df["p_holm_8"] < 0.05
    full_df["Significant_Table4_Alpha (p<0.0083)"] = full_df["p_value_raw"] < (0.05 / 6.0)

    drop_cols = ["p_value_raw"]
    return full_df.drop(columns=drop_cols)


def plot_participant_paired_differences(tables_dir: Path, stats_df: pd.DataFrame, figures_dir: Path):
    """Generate publication-quality paired difference visualizations for Stress-Predict & WESAD."""
    figures_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # FIGURE 1: Per-Participant Paired Differences on Stress-Predict (n=33)
    # Shows subject-by-subject Delta = Hierarchical - Standard-Matched and
    # Delta = Hierarchical - Standard for F1-Score and Balanced Accuracy (30s & 60s)
    # =========================================================================
    sp30_path = tables_dir / "ablation_matched_capacity_participants_stress_predict_30s.csv"
    sp60_path = tables_dir / "ablation_matched_capacity_participants_stress_predict_60s.csv"

    if sp30_path.exists() and sp60_path.exists():
        sp30 = pd.read_csv(sp30_path)
        sp60 = pd.read_csv(sp60_path)

        fig, axes = plt.subplots(2, 2, figsize=(15, 9.5), sharey="row")
        panel_specs = [
            (0, 0, sp30, 30, "f1", "F1-Score"),
            (0, 1, sp60, 60, "f1", "F1-Score"),
            (1, 0, sp30, 30, "balanced_accuracy", "Balanced Accuracy"),
            (1, 1, sp60, 60, "balanced_accuracy", "Balanced Accuracy"),
        ]

        for r_idx, c_idx, df_win, win, m_key, m_label in panel_specs:
            ax = axes[r_idx, c_idx]
            piv = df_win.pivot(index="participant", columns="variant", values=m_key).sort_index()
            participants = [f"S{int(p):02d}" for p in piv.index]
            x = np.arange(len(participants))

            diff_matched = (piv["hierarchical"] - piv["standard_matched"]).to_numpy(dtype=float)
            diff_standard = (piv["hierarchical"] - piv["standard"]).to_numpy(dtype=float)

            width = 0.38
            ax.axhline(0.0, color="black", linewidth=1.1, linestyle="-", zorder=2)

            # Bars for Hierarchical - Standard-Matched (Architectural effect at matched capacity)
            ax.bar(
                x - width / 2,
                diff_matched,
                width,
                label="Δ (Hierarchical − Standard-Matched) [Architecture Effect]",
                color="#2B7BBA",
                alpha=0.88,
                edgecolor="black",
                linewidth=0.5,
                zorder=3,
            )
            # Bars for Hierarchical - Standard (Total difference)
            ax.bar(
                x + width / 2,
                diff_standard,
                width,
                label="Δ (Hierarchical − Standard) [Total Difference]",
                color="#E6862A",
                alpha=0.82,
                edgecolor="black",
                linewidth=0.5,
                zorder=3,
            )

            # Mean lines & stats box
            mean_m = float(np.mean(diff_matched))
            mean_s = float(np.mean(diff_standard))
            ax.axhline(mean_m, color="#1B4F72", linestyle="--", linewidth=1.2, alpha=0.9)
            ax.axhline(mean_s, color="#A04000", linestyle=":", linewidth=1.2, alpha=0.9)

            row_m = stats_df[
                (stats_df["Dataset"] == "Stress-Predict")
                & (stats_df["Window (s)"] == win)
                & (stats_df["Comparison"] == "Hierarchical vs. Standard-Matched")
                & (stats_df["Metric"] == m_key)
            ].iloc[0]
            row_s = stats_df[
                (stats_df["Dataset"] == "Stress-Predict")
                & (stats_df["Window (s)"] == win)
                & (stats_df["Comparison"] == "Hierarchical vs. Standard")
                & (stats_df["Metric"] == m_key)
            ].iloc[0]

            annot_text = (
                f"Hier − Matched: Δ={mean_m:+.4f} [{row_m['CI95_Low']:+.4f}, {row_m['CI95_High']:+.4f}], "
                f"r_rb={row_m['Rank_Biserial_r']:+.3f}, p={row_m['p_value']:.3f} "
                f"(W/T/L: {row_m['Wins_A']}/{row_m['Ties']}/{row_m['Losses_A']})\n"
                f"Hier − Standard: Δ={mean_s:+.4f} [{row_s['CI95_Low']:+.4f}, {row_s['CI95_High']:+.4f}], "
                f"r_rb={row_s['Rank_Biserial_r']:+.3f}, p={row_s['p_value']:.3f} "
                f"(W/T/L: {row_s['Wins_A']}/{row_s['Ties']}/{row_s['Losses_A']})"
            )
            ax.text(
                0.015,
                0.96,
                annot_text,
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=8.2,
                family="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#F8F9F9", edgecolor="#BDC3C7", alpha=0.94),
                zorder=5,
            )

            ax.set_title(f"Stress-Predict ({win}s Window, n={len(participants)}) — {m_label}", fontsize=11.5, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(participants, rotation=60, fontsize=8)
            ax.grid(axis="y", linestyle="--", alpha=0.45, zorder=1)
            if c_idx == 0:
                ax.set_ylabel(f"Paired Difference in {m_label} (Δ)", fontsize=10.5)
            if r_idx == 0 and c_idx == 1:
                ax.legend(loc="lower right", frameon=True, fontsize=8.8)

        axes[0, 0].set_ylim(-0.22, 0.24)
        axes[1, 0].set_ylim(-0.26, 0.28)
        fig.suptitle(
            "Participant-Level Paired Differences on Stress-Predict (Experiment 1 Supervised LOGO Protocol)",
            fontsize=13.5,
            fontweight="bold",
            y=0.995,
        )
        plt.tight_layout()
        out_png = figures_dir / "ablation_paired_differences_stress_predict.png"
        out_pdf = figures_dir / "ablation_paired_differences_stress_predict.pdf"
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        fig.savefig(out_pdf, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved participant-level paired differences plot to: {out_png}")

    # =========================================================================
    # FIGURE 2: Forest / Box-and-Strip Plot of Paired Differences Across All Metrics
    # Comparing (Hierarchical - Standard-Matched) and (Hierarchical - Standard)
    # on Stress-Predict (30s & 60s) and WESAD (30s & 60s)
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.5), sharex=False)
    configs = [
        (0, 0, "Stress-Predict", 30, "ablation_matched_capacity_participants_stress_predict_30s.csv"),
        (0, 1, "Stress-Predict", 60, "ablation_matched_capacity_participants_stress_predict_60s.csv"),
        (1, 0, "WESAD", 30, "ablation_matched_capacity_participants_wesad_30s.csv"),
        (1, 1, "WESAD", 60, "ablation_matched_capacity_participants_wesad_60s.csv"),
    ]

    rng = np.random.default_rng(42)
    for r_idx, c_idx, ds_name, win, fname in configs:
        ax = axes[r_idx, c_idx]
        fpath = tables_dir / fname
        if not fpath.exists():
            continue
        df_p = pd.read_csv(fpath)
        n_subj = len(df_p["participant"].unique())

        ax.axvline(0.0, color="black", linestyle="-", linewidth=1.1, zorder=2)

        y_positions = np.arange(len(METRICS))
        offset = 0.18

        for m_idx, (m_key, m_label) in enumerate(METRICS):
            piv = df_p.pivot(index="participant", columns="variant", values=m_key).dropna()
            d_matched = (piv["hierarchical"] - piv["standard_matched"]).to_numpy(dtype=float)
            d_std = (piv["hierarchical"] - piv["standard"]).to_numpy(dtype=float)

            row_m = stats_df[
                (stats_df["Dataset"] == ds_name)
                & (stats_df["Window (s)"] == win)
                & (stats_df["Comparison"] == "Hierarchical vs. Standard-Matched")
                & (stats_df["Metric"] == m_key)
            ].iloc[0]
            row_s = stats_df[
                (stats_df["Dataset"] == ds_name)
                & (stats_df["Window (s)"] == win)
                & (stats_df["Comparison"] == "Hierarchical vs. Standard")
                & (stats_df["Metric"] == m_key)
            ].iloc[0]

            # Plot individual subject dots (jittered) + Mean & 95% CI error bar
            y_m = m_idx - offset
            y_s = m_idx + offset

            jitter_m = rng.uniform(-0.06, 0.06, size=len(d_matched))
            jitter_s = rng.uniform(-0.06, 0.06, size=len(d_std))

            ax.scatter(d_matched, y_m + jitter_m, color="#2B7BBA", alpha=0.35, s=22, zorder=3)
            ax.scatter(d_std, y_s + jitter_s, color="#E6862A", alpha=0.35, s=22, zorder=3)

            # Mean and 95% CI horizontal error bar
            ax.errorbar(
                row_m["Mean_Diff (A - B)"],
                y_m,
                xerr=[[row_m["Mean_Diff (A - B)"] - row_m["CI95_Low"]], [row_m["CI95_High"] - row_m["Mean_Diff (A - B)"]]],
                fmt="o",
                color="#1B4F72",
                ecolor="#1B4F72",
                elinewidth=2.2,
                capsize=4.5,
                markersize=6.5,
                label="Hierarchical − Standard-Matched (95% CI)" if m_idx == 0 else None,
                zorder=5,
            )
            ax.errorbar(
                row_s["Mean_Diff (A - B)"],
                y_s,
                xerr=[[row_s["Mean_Diff (A - B)"] - row_s["CI95_Low"]], [row_s["CI95_High"] - row_s["Mean_Diff (A - B)"]]],
                fmt="s",
                color="#A04000",
                ecolor="#A04000",
                elinewidth=2.2,
                capsize=4.5,
                markersize=6.0,
                label="Hierarchical − Standard (95% CI)" if m_idx == 0 else None,
                zorder=5,
            )

            # Text label on the right margin with Mean Diff, r_rb, and Wilcoxon p
            ax.text(
                row_m["CI95_High"] + 0.008,
                y_m,
                f"Δ={row_m['Mean_Diff (A - B)']:+.3f} (r={row_m['Rank_Biserial_r']:+.2f}, p={row_m['p_value']:.2f})",
                va="center",
                ha="left",
                fontsize=7.8,
                color="#1B4F72",
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.1", facecolor="white", edgecolor="none", alpha=0.85),
            )
            ax.text(
                row_s["CI95_High"] + 0.008,
                y_s,
                f"Δ={row_s['Mean_Diff (A - B)']:+.3f} (r={row_s['Rank_Biserial_r']:+.2f}, p={row_s['p_value']:.2f})",
                va="center",
                ha="left",
                fontsize=7.8,
                color="#A04000",
                bbox=dict(boxstyle="round,pad=0.1", facecolor="white", edgecolor="none", alpha=0.85),
            )

        ax.set_yticks(y_positions)
        ax.set_yticklabels([label for _, label in METRICS], fontsize=10, fontweight="bold")
        ax.invert_yaxis()
        ax.set_xlim(-0.24, 0.34)
        ax.grid(axis="x", linestyle="--", alpha=0.45, zorder=1)
        ax.set_title(f"{ds_name} — {win}s Window (n={n_subj} Participants)", fontsize=11.5, fontweight="bold")
        ax.set_xlabel("Paired Score Difference (Δ = Model A − Model B)", fontsize=10)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle(
        "Paired Statistical Comparison (Mean Δ ± 95% CI, Rank-Biserial r, and Wilcoxon p-value)",
        fontsize=13.5,
        fontweight="bold",
        y=0.995,
    )
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.962), ncol=2, frameon=True, fontsize=9.8)
    plt.tight_layout(rect=[0, 0, 1, 0.935])
    sum_png = figures_dir / "ablation_paired_differences_summary.png"
    sum_pdf = figures_dir / "ablation_paired_differences_summary.pdf"
    fig.savefig(sum_png, dpi=300, bbox_inches="tight")
    fig.savefig(sum_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved paired statistical forest/distribution plot to: {sum_png}")


def main():
    tables_dir = PROJECT_ROOT / "results" / "tables" / "ablation_matched_capacity"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    stats_df = run_paired_statistical_analysis(tables_dir)
    if stats_df.empty:
        print("No participant-level CSV files found.")
        return

    # Save comprehensive statistical table
    stat_csv = tables_dir / "ablation_matched_capacity_statistical_analysis.csv"
    stats_df.to_csv(stat_csv, index=False)
    print(f"Saved comprehensive statistical analysis table to: {stat_csv}")

    # Also update ablation_matched_capacity_paired_comparisons.csv and root copies
    paired_csv = tables_dir / "ablation_matched_capacity_paired_comparisons.csv"
    stats_df.to_csv(paired_csv, index=False)

    legacy_stat_csv = PROJECT_ROOT / "results" / "tables" / "ablation_matched_capacity_statistical_analysis.csv"
    legacy_paired_csv = PROJECT_ROOT / "results" / "tables" / "ablation_matched_capacity_paired_comparisons.csv"
    stats_df.to_csv(legacy_stat_csv, index=False)
    stats_df.to_csv(legacy_paired_csv, index=False)

    # Print Stress-Predict and WESAD primary comparisons
    primary = stats_df[
        stats_df["Comparison"].isin(["Hierarchical vs. Standard-Matched", "Hierarchical vs. Standard", "Standard-Matched vs. Standard"])
    ]
    display_cols = [
        "Dataset", "Window (s)", "Comparison", "Metric_Label", "n",
        "Mean_Diff (A - B)", "Std_Diff", "CI95_Low", "CI95_High",
        "Wins_A", "Ties", "Losses_A", "Cohen_dz", "Rank_Biserial_r",
        "Wilcoxon_stat", "p_value", "p_holm_8", "Significant_Holm8 (p_adj<0.05)"
    ]
    print("\n=========================================================================================================")
    print("PHASE 6: PAIRED STATISTICAL ANALYSIS OF CAPACITY-MATCHED ABLATION (Stress-Predict & WESAD)")
    print("=========================================================================================================")
    print(primary[display_cols].to_string(index=False))

    plot_participant_paired_differences(tables_dir, stats_df, figures_dir)


if __name__ == "__main__":
    main()
