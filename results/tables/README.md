# Results Tables Organization Guide

This directory organizes all tabular data, quantitative summaries, statistical analyses, and audit verification tables of the **TimePatch-SSL** project into three categories:
1. **Manuscript Benchmark Results** (Original Experiments 1–4 from Colab, Tables 2, 3, 4, and 6)
2. **Reviewer 3 Audit & Reproducibility Tables (Points 1–4 & Point 7)** (Cross-dataset calibration, per-class metrics, confusion matrices, SSL learning curves, $K=5$ multi-split sensitivity, positive-class verification, and master consistency audit)
3. **Reviewer 3 Matched-Capacity Ablation & Paired Statistical Analysis (Points 5 & 6)** (`Standard`, `Hierarchical`, and `Standard-Matched` across **WESAD** and **Stress-Predict**)

---

## Directory Layout

```text
results/tables/
├── README.md                                           # This navigation guide
│
├── reported_metrics_summary.csv                        # Unified summary of Experiments 1–4 from manuscript
├── reviewer3_evidence_traceability_matrix.csv          # Point 7: Complete traceability matrix (Points 1–7)
├── audit_point7_tables_consistency_summary.csv         # Point 7: Master consistency summary (Tables 2, 3, 4, 6, 7)
├── audit_point7_positive_class_verification.csv        # Point 7: Positive-class (y=1 Stress) verification
│
├── audit_point1_threshold_sweep.csv                    # Point 1 / Table S1: Cross-dataset threshold sweep
├── audit_point1_participant_metrics.csv                # Point 1 / Table S2: Cross-dataset per-subject metrics (n=15)
├── audit_point2_per_class_metrics.csv                  # Point 2 / Table S3: Per-class Precision/Recall/F1 & specificity
├── audit_point2_confusion_matrices.csv                 # Point 2 / Table S4: Exact confusion matrix counts (TN, FP, FN, TP)
├── audit_point3_learning_curves_summary.csv            # Point 3 / Table S5: SSL pretraining & convergence dynamics
├── audit_point4_multisplit_runs.csv                    # Point 4: Individual K=5 GroupShuffleSplit runs
├── audit_point4_multisplit_summary.csv                 # Point 4 / Table S6: K=5 multi-split Mean ± SD & 95% CI
│
├── source_result_summaries/                            # Raw output logs from original Google Colab runs
│   ├── supervised_wesad_60s.txt                        # Exp 1: Supervised WESAD (60s)
│   ├── supervised_wesad_30s.txt                        # Exp 1: Supervised WESAD (30s)
│   ├── supervised_stresspredict_60s.txt                # Exp 1: Supervised Stress-Predict (60s)
│   ├── supervised_stresspredict_30s.txt                # Exp 1: Supervised Stress-Predict (30s)
│   ├── cross_dataset_*.txt                             # Exp 2: Cross-dataset transfer (both directions)
│   ├── ssl_*.txt                                       # Exp 3: Within-dataset SSL pretraining
│   └── combined_*.txt                                  # Exp 4: Combined-dataset evaluations
│
└── ablation_matched_capacity/                          # Capacity-Matched Ablation (WESAD & Stress-Predict)
    ├── README.md                                       # Detailed guide & summary tables for reviewer response
    ├── table7_extended_ablation_summary.csv            # Point 5 / Table S7: Extended Table 7 (WESAD & Stress-Predict)
    ├── ablation_matched_capacity_parameters.csv        # Point 5: Parameter parity audit (-2.61%, +3.83%, +4.80%)
    ├── ablation_matched_capacity_paired_comparisons.csv # Point 5: Paired Wilcoxon summary
    ├── ablation_matched_capacity_statistical_analysis.csv # Point 6 / Table S8: Complete paired statistical analysis
    ├── ablation_matched_capacity_results_stress_predict_60s.csv       # Stress-Predict 60s summary
    ├── ablation_matched_capacity_participants_stress_predict_60s.csv  # Stress-Predict 60s (n=33)
    ├── ablation_matched_capacity_results_stress_predict_30s.csv       # Stress-Predict 30s summary
    ├── ablation_matched_capacity_participants_stress_predict_30s.csv  # Stress-Predict 30s (n=33)
    ├── ablation_matched_capacity_results_wesad_60s.csv                # WESAD 60s summary
    ├── ablation_matched_capacity_participants_wesad_60s.csv           # WESAD 60s (n=15)
    ├── ablation_matched_capacity_results_wesad_30s.csv                # WESAD 30s summary
    └── ablation_matched_capacity_participants_wesad_30s.csv           # WESAD 30s (n=15)
```

---

## 1. Manuscript & Master Consistency Tables

- **[`reported_metrics_summary.csv`](./reported_metrics_summary.csv)**:
  Aggregates the 18 experimental configurations evaluated in the manuscript across both datasets (**WESAD** and **Stress-Predict**) and both window durations (**30s** and **60s**).
- **[`audit_point7_tables_consistency_summary.csv`](./audit_point7_tables_consistency_summary.csv)**:
  Unifies 50 rows across Manuscript **Tables 2, 3, 4, 6, and 7**, cross-checking reported manuscript metrics against the repository's audit and ablation regenerations.
- **[`audit_point7_positive_class_verification.csv`](./audit_point7_positive_class_verification.csv)**:
  Confirms across all 9 source/audit modules and prediction logs that `y = 1` (**Stress**) is consistently treated as the positive class (`p_stress = softmax(logits)[:, 1]`, default threshold `0.50`).
- **[`reviewer3_evidence_traceability_matrix.csv`](./reviewer3_evidence_traceability_matrix.csv)**:
  Links each of Reviewer 3's 7 requirements (`Points 1–7`) to the exact script, CSV table, Supplementary figure, and manuscript section.

---

## 2. Reviewer 3 Audit Tables (Points 1–4, Tables S1–S6)

- **[`audit_point1_threshold_sweep.csv`](./audit_point1_threshold_sweep.csv)** & **[`audit_point1_participant_metrics.csv`](./audit_point1_participant_metrics.csv)**:
  Explain the Table 2 cross-dataset discrepancy (`Stress-Predict → WESAD`, 30s `Standard`): high ranking discrimination ($\text{AUROC} = 0.8707$) coexists with low thresholded F1 ($0.0725$ at $\tau = 0.50$) due to target-domain probability shift ($p_{\text{stress}} \in [0.055, 0.575]$), whereas recalibrating to $\tau^* = 0.167$ recovers $\text{F1} = 0.8229$.
- **[`audit_point2_per_class_metrics.csv`](./audit_point2_per_class_metrics.csv)** & **[`audit_point2_confusion_matrices.csv`](./audit_point2_confusion_matrices.csv)**:
  Provide explicit per-class Precision, Recall (Sensitivity/Specificity), F1, and confusion matrix counts ($\text{TN}, \text{FP}, \text{FN}, \text{TP}$) across imbalanced experiments (Tables 2, 3, and 6).
- **[`audit_point3_learning_curves_summary.csv`](./audit_point3_learning_curves_summary.csv)**:
  Summarizes epoch-by-epoch NT-Xent pretraining loss and downstream fine-tuning dynamics (Experiment 3 / Table 3).
- **[`audit_point4_multisplit_runs.csv`](./audit_point4_multisplit_runs.csv)** & **[`audit_point4_multisplit_summary.csv`](./audit_point4_multisplit_summary.csv)**:
  Report $K=5$ subject-grouped `GroupShuffleSplit` evaluations for Experiment 4 (Table 6), replacing reliance on a single holdout split with $\text{Mean} \pm \text{SD}$ and 95% CIs.

---

## 3. Matched-Capacity Ablation & Paired Statistical Analysis (Points 5 & 6, Tables S7–S8)

Located in [`ablation_matched_capacity/`](./ablation_matched_capacity/README.md):
- **[`table7_extended_ablation_summary.csv`](./ablation_matched_capacity/table7_extended_ablation_summary.csv)** (**Table S7**):
  Extended Table 7 comparing `Standard`, `Hierarchical`, and `Standard-Matched` across **Stress-Predict** ($n=33$) and **WESAD** ($n=15$) for `30s` and `60s` windows, reporting exact relative parameter differences (**+3.83%** for 30s, **+4.80%** for Stress-Predict 60s, and **-2.61%** for WESAD 60s).
- **[`ablation_matched_capacity_statistical_analysis.csv`](./ablation_matched_capacity/ablation_matched_capacity_statistical_analysis.csv)** (**Table S8**):
  Complete subject-paired statistical evaluation (`Hierarchical vs. Standard-Matched`, `Hierarchical vs. Standard`, `Standard-Matched vs. Standard`) with parametric Student-$t$ 95% CIs, 10,000-resample BCa/percentile bootstrap 95% CIs, rank-biserial correlation ($r_{\text{rb}}$), Cohen's $d_z$, Wilcoxon signed-rank $p$-values, and Bonferroni/Holm-Bonferroni adjustments.
