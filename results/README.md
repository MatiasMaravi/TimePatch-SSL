# Results Directory Guide

This directory contains all experimental outputs, quantitative tables, participant-level metrics, and evaluation artifacts for the **TimePatch-SSL** project.

It is structured to separate original manuscript benchmarks from the newly added Reviewer Point 5 ablation experiments.

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
│   └── ablation_matched_capacity/             # Reviewer Point 5: Matched-Capacity Ablation Suite
│       ├── README.md                          # Full scientific summary of the ablation study
│       ├── ablation_matched_capacity_parameters.csv # Parameter parity audit (<4% relative diff)
│       ├── ablation_matched_capacity_results_wesad_60s.csv       # Summary (Mean ± Std, 95% CI)
│       ├── ablation_matched_capacity_participants_wesad_60s.csv  # 15-subject metrics breakdown
│       ├── ablation_matched_capacity_results_wesad_30s.csv       # Summary (Mean ± Std, 95% CI)
│       └── ablation_matched_capacity_participants_wesad_30s.csv  # 15-subject metrics breakdown
│
└── figures/                                   # Manuscript figures, ROC curves, confusion matrices
    └── README.md
```

---

## Quick Reference for Paper Review & Revision

| Objective | Relevant Subdirectory / File | Description |
| :--- | :--- | :--- |
| **Original Manuscript Results (Exp 1–4)** | [`tables/reported_metrics_summary.csv`](./tables/reported_metrics_summary.csv) | Consolidated table of all 18 runs across WESAD and Stress-Predict (30s & 60s). |
| **Statistical Tests (Table 4)** | [`predictions/table4_statistics_60s_from_colab.csv`](./predictions/table4_statistics_60s_from_colab.csv) | Wilcoxon signed-rank tests comparing Supervised vs. SSL across participants. |
| **Colab Execution Logs** | [`tables/source_result_summaries/`](./tables/source_result_summaries/) | Text logs from Colab runs for supervised, cross-dataset, SSL, and combined protocols. |
| **Reviewer Point 5: Parity Audit** | [`tables/ablation_matched_capacity/ablation_matched_capacity_parameters.csv`](./tables/ablation_matched_capacity/ablation_matched_capacity_parameters.csv) | Trainable parameter counts proving < 4% parity between Hierarchical and Standard-Matched. |
| **Reviewer Point 5: 60s Ablation** | [`tables/ablation_matched_capacity/ablation_matched_capacity_results_wesad_60s.csv`](./tables/ablation_matched_capacity/ablation_matched_capacity_results_wesad_60s.csv) | 15-subject LOSO evaluation comparing Standard vs. Hierarchical vs. Standard-Matched on 60s windows. |
| **Reviewer Point 5: 30s Ablation** | [`tables/ablation_matched_capacity/ablation_matched_capacity_results_wesad_30s.csv`](./tables/ablation_matched_capacity/ablation_matched_capacity_results_wesad_30s.csv) | 15-subject LOSO evaluation comparing Standard vs. Hierarchical vs. Standard-Matched on 30s windows. |
| **Subject-Level Ablation Metrics** | [`tables/ablation_matched_capacity/ablation_matched_capacity_participants_*.csv`](./tables/ablation_matched_capacity/) | Detailed participant-by-participant performance for all 15 subjects ($S2 \dots S17$). |
