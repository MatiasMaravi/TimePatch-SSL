# Results Directory Guide

This directory contains all experimental outputs, quantitative tables, participant-level metrics, and evaluation artifacts for the **TimePatch-SSL** project.

It is structured to separate original manuscript benchmarks from the capacity-matched ablation experiments across **WESAD** and **Stress-Predict**.

---

## Directory Organization

```text
results/
├── README.md                                  # This overview guide
│
├── metadata/                                  # Experimental configurations & hyperparameter selections
│   └── selected_hyperparameters_60s_from_colab.json
│
├── predictions/                               # Participant-level prediction logs & statistical tests
│   ├── README.md
│   ├── prediction_schema.csv                  # Schema specification for window-level exports
│   ├── participant_metrics_60s_from_colab.csv # Participant-level breakdown for 60s from Colab
│   └── table4_statistics_60s_from_colab.csv   # Wilcoxon signed-rank test results from Colab
│
├── tables/                                    # Quantitative summary tables (Manuscript & Ablation)
│   ├── README.md                              # Tables directory guide
│   ├── reported_metrics_summary.csv           # Unified benchmark summary of Experiments 1–4
│   ├── source_result_summaries/               # 18 raw execution logs from Google Colab
│   └── ablation_matched_capacity/             # Capacity-Matched Ablation Suite (WESAD & Stress-Predict)
│       ├── README.md                          # Full scientific summary of the ablation study
│       ├── table7_extended_ablation_summary.csv # Extended Table 7 across WESAD & Stress-Predict
│       ├── ablation_matched_capacity_parameters.csv # Parameter parity audit (-2.61% to +4.80% diff)
│       ├── ablation_matched_capacity_paired_comparisons.csv # Paired Wilcoxon tests
│       ├── ablation_matched_capacity_results_stress_predict_60s.csv       # Stress-Predict 60s summary
│       ├── ablation_matched_capacity_participants_stress_predict_60s.csv  # Stress-Predict 60s (n=33)
│       ├── ablation_matched_capacity_results_stress_predict_30s.csv       # Stress-Predict 30s summary
│       ├── ablation_matched_capacity_participants_stress_predict_30s.csv  # Stress-Predict 30s (n=33)
│       ├── ablation_matched_capacity_results_wesad_60s.csv                # WESAD 60s summary
│       ├── ablation_matched_capacity_participants_wesad_60s.csv           # WESAD 60s (n=15)
│       ├── ablation_matched_capacity_results_wesad_30s.csv                # WESAD 30s summary
│       └── ablation_matched_capacity_participants_wesad_30s.csv           # WESAD 30s (n=15)
│
└── figures/                                   # Manuscript figures & comparative ablation plots
    ├── README.md
    ├── ablation_matched_capacity_stress_predict.png # Comparative bar chart for Stress-Predict (30s & 60s)
    ├── ablation_matched_capacity_stress_predict.pdf
    ├── ablation_matched_capacity_all_datasets.png   # 4-panel comparative bar chart (WESAD & Stress-Predict)
    └── ablation_matched_capacity_all_datasets.pdf
```

---

## Quick Reference for Paper Review & Revision

| Objective | Relevant Subdirectory / File | Description |
| :--- | :--- | :--- |
| **Original Manuscript Results (Exp 1–4)** | [`tables/reported_metrics_summary.csv`](./tables/reported_metrics_summary.csv) | Consolidated table of all 18 runs across WESAD and Stress-Predict (30s & 60s). |
| **Statistical Tests (Table 4)** | [`predictions/table4_statistics_60s_from_colab.csv`](./predictions/table4_statistics_60s_from_colab.csv) | Wilcoxon signed-rank tests comparing Supervised vs. SSL across participants. |
| **Extended Table 7 (WESAD + Stress-Predict)** | [`tables/ablation_matched_capacity/table7_extended_ablation_summary.csv`](./tables/ablation_matched_capacity/table7_extended_ablation_summary.csv) | Complete Table 7 extension comparing `Standard`, `Hierarchical`, and `Standard-Matched` across both datasets and window sizes. |
| **Parameter Parity Audit** | [`tables/ablation_matched_capacity/ablation_matched_capacity_parameters.csv`](./tables/ablation_matched_capacity/ablation_matched_capacity_parameters.csv) | Trainable parameter counts reporting exact relative differences (**-2.61%** WESAD 60s, **+3.83%** 30s, **+4.80%** Stress-Predict 60s). |
| **Stress-Predict Ablation (60s & 30s)** | [`tables/ablation_matched_capacity/ablation_matched_capacity_results_stress_predict_60s.csv`](./tables/ablation_matched_capacity/ablation_matched_capacity_results_stress_predict_60s.csv) | 33-subject LOGO evaluation on Stress-Predict (60s & 30s) under Experiment 1 protocol. |
| **Comparative Ablation Figures** | [`figures/ablation_matched_capacity_stress_predict.png`](./figures/ablation_matched_capacity_stress_predict.png) | Bar plots comparing Accuracy, F1, Balanced Accuracy, and AUROC across all three models. |
