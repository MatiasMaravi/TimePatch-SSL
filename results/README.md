# Results Directory Guide

This directory contains all experimental outputs, quantitative tables, participant-level metrics, audit verification logs, and publication-ready Supplementary Material figures for the **TimePatch-SSL** project.

It is structured to organize:
1. **Original Manuscript Benchmarks** (Experiments 1–4 across **WESAD** and **Stress-Predict**, Tables 2, 3, 4, and 6).
2. **Reviewer 3 Audit Suite (Points 1–4 & Point 7)** (Cross-dataset probability calibration, per-class metrics, confusion matrices, SSL learning curves, $K=5$ multi-split sensitivity, positive-class verification, and master traceability matrix).
3. **Reviewer 3 Capacity-Matched Ablation & Paired Statistical Analysis (Points 5 & 6)** (`Standard`, `Hierarchical`, and `Standard-Matched` across **WESAD** and **Stress-Predict** under Experiment 1 LOGO validation, Extended Table 7, paired Wilcoxon tests, effect sizes, and paired difference figures).

---

## Directory Organization

```text
results/
├── README.md                                  # This overview guide
│
├── metadata/                                  # Experimental configurations & hyperparameter selections
│   └── selected_hyperparameters_60s_from_colab.json
│
├── predictions/                               # Window- and participant-level prediction logs & statistical tests
│   ├── README.md
│   ├── prediction_schema.csv                  # Schema specification for window-level exports
│   ├── participant_metrics_60s_from_colab.csv # Participant-level breakdown for 60s from Colab
│   ├── table4_statistics_60s_from_colab.csv   # Wilcoxon signed-rank test results from Colab (Table 4)
│   └── audit_point1_stresspredict_to_wesad_30s_standard.csv # Window-level probabilities (Point 1 Audit)
│
├── tables/                                    # Quantitative summary tables (Manuscript, Audit & Ablation)
│   ├── README.md                              # Tables directory guide
│   ├── reported_metrics_summary.csv           # Unified benchmark summary of Experiments 1–4
│   ├── reviewer3_evidence_traceability_matrix.csv      # Point 7: Complete traceability matrix (Points 1–7)
│   ├── audit_point7_tables_consistency_summary.csv     # Point 7: Unified audit of Tables 2, 3, 4, 6 & 7 (50 rows)
│   ├── audit_point7_positive_class_verification.csv    # Point 7: Programmatic verification of y=1 (Stress)
│   ├── audit_point1_auroc_table.csv                    # Point 1: Cross-dataset p_stress vs (1-p_stress) AUROC audit (Table S1)
│   ├── audit_point2_per_class_metrics.csv              # Point 2: Per-class Precision/Recall/F1 & confusion matrices (Table S2)
│   ├── audit_point2_always_stress_comparison.csv       # Point 2: Comparison vs. trivial Always-Stress baseline (Table S3)
│   ├── audit_point3_convergence_summary.csv            # Point 3: Standard vs. Hierarchical convergence summary (Table S4)
│   ├── audit_point4_fairness_hyperparameters_budget.csv # Point 4: Fairness, Optuna budget & zero-leakage audit (Table S5)
│   ├── source_result_summaries/               # 18 raw execution logs from Google Colab
│   └── ablation_matched_capacity/             # Capacity-Matched Ablation Suite (Points 5 & 6)
│       ├── README.md                          # Full scientific summary of the ablation study
│       ├── table7_extended_ablation_summary.csv # Extended Table 7 across WESAD & Stress-Predict (Table S6)
│       ├── ablation_matched_capacity_parameters.csv # Parameter parity audit (-2.61%, +3.83%, +4.80% diff)
│       ├── ablation_matched_capacity_paired_comparisons.csv # Paired Wilcoxon tests
│       ├── ablation_matched_capacity_statistical_analysis.csv # Point 6: Full paired stats (CIs, r_rb, d_z, Holm) (Table S7)
│       ├── ablation_matched_capacity_results_stress_predict_60s.csv       # Stress-Predict 60s summary
│       ├── ablation_matched_capacity_participants_stress_predict_60s.csv  # Stress-Predict 60s (n=33)
│       ├── ablation_matched_capacity_results_stress_predict_30s.csv       # Stress-Predict 30s summary
│       ├── ablation_matched_capacity_participants_stress_predict_30s.csv  # Stress-Predict 30s (n=33)
│       ├── ablation_matched_capacity_results_wesad_60s.csv                # WESAD 60s summary
│       ├── ablation_matched_capacity_participants_wesad_60s.csv           # WESAD 60s (n=15)
│       ├── ablation_matched_capacity_results_wesad_30s.csv                # WESAD 30s summary
│       └── ablation_matched_capacity_participants_wesad_30s.csv           # WESAD 30s (n=15)
│
└── figures/                                   # Supplementary Material figures (Figures S1–S8)
    ├── README.md                              # Complete catalog & captions for Figures S1–S8
    ├── audit_point1_roc_curve.png             # Figure S1: Cross-dataset ROC curve (p_stress vs 1-p_stress)
    ├── audit_point2_confusion_matrices.png    # Figure S2: 12-panel confusion matrices across Exp 1, 2 & 4
    ├── audit_point3_loss_curves.png           # Figure S3: Train & validation cross-entropy loss curves
    ├── audit_point3_val_metrics_curves.png    # Figure S4: Validation F1 & Balanced Accuracy trajectories
    ├── ablation_matched_capacity_stress_predict.{png,pdf} # Figure S5: Stress-Predict 3-model ablation (30s & 60s)
    ├── ablation_matched_capacity_all_datasets.{png,pdf}   # Figure S6: 4-panel ablation across Stress-Predict & WESAD
    ├── ablation_paired_differences_stress_predict.{png,pdf} # Figure S7: Subject-level paired Δ distributions (n=33)
    └── ablation_paired_differences_summary.{png,pdf}      # Figure S8: Forest plot of paired Δ with t & bootstrap 95% CIs
```

---

## Quick Reference for Paper Review & Revision

| Objective | Relevant Subdirectory / File | Description |
| :--- | :--- | :--- |
| **Reviewer 3 Master Traceability Matrix** | [`tables/reviewer3_evidence_traceability_matrix.csv`](./tables/reviewer3_evidence_traceability_matrix.csv) | Maps every Reviewer 3 point (Points 1–7) to its reproduction script, tables, figures, and scientific conclusion. |
| **Master Consistency Audit (Tables 2, 3, 4, 6, 7)** | [`tables/audit_point7_tables_consistency_summary.csv`](./tables/audit_point7_tables_consistency_summary.csv) | 50-row unified consistency table across all manuscript and supplementary tables. |
| **Positive-Class Verification (`y=1` Stress)** | [`tables/audit_point7_positive_class_verification.csv`](./tables/audit_point7_positive_class_verification.csv) | Programmatic verification that all scripts and prediction files use `y=1` (Stress) and `softmax[:, 1]`. |
| **Extended Table 7 (WESAD + Stress-Predict)** | [`tables/ablation_matched_capacity/table7_extended_ablation_summary.csv`](./tables/ablation_matched_capacity/table7_extended_ablation_summary.csv) | Complete Table 7 extension comparing `Standard`, `Hierarchical`, and `Standard-Matched` across both datasets and window sizes. |
| **Point 6 Paired Statistical Analysis** | [`tables/ablation_matched_capacity/ablation_matched_capacity_statistical_analysis.csv`](./tables/ablation_matched_capacity/ablation_matched_capacity_statistical_analysis.csv) | Paired differences, Student-$t$ and 10,000-resample bootstrap 95% CIs, $r_{\text{rb}}$, Cohen's $d_z$, Wilcoxon $p$, and Holm corrections. |
| **Supplementary Figures & Captions Guide** | [`../docs/supplementary_material_figures_and_tables_guide.md`](../docs/supplementary_material_figures_and_tables_guide.md) | Publication-ready captions (English & Spanish) for Supplementary Figures S1–S8 and Tables S1–S8. |
