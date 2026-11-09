# TimePatch-SSL

**Contrastive Self-Supervised Learning with Hierarchical Time-Patch
Transformers for Multimodal Stress Detection**

TimePatch-SSL is a research implementation for multimodal physiological
stress detection using a hierarchical temporal patch encoder and
contrastive self-supervised learning (SSL). The framework operates on
wrist-derived physiological signals and is evaluated using the **WESAD**
and **Stress-Predict** datasets under within-dataset, cross-dataset, and
combined-dataset protocols.

This repository accompanies the manuscript:

> **TimePatch-SSL: Contrastive Self-Supervised Learning with
> Hierarchical Time-Patch Transformers for Multimodal Stress Detection**

The repository is intended to provide the code, experimental
configurations, data-split definitions, and analysis procedures needed
to reproduce the reported experiments.

## Overview

The framework uses three physiological modalities:

-   Blood Volume Pulse (BVP)
-   Electrodermal Activity (EDA)
-   Skin Temperature (TEMP)

The model converts synchronized physiological windows into temporal
patches. Three encoder variants are evaluated:

-   **Standard encoder:** direct linear patch projection followed by a
    global Transformer encoder.
-   **Hierarchical encoder:** a local patch encoder first models
    within-patch temporal structure, followed by a global Transformer
    encoder across patches.
-   **Standard-Matched encoder (Capacity Control):** retains the direct linear
    patch projection but scales global Transformer depth to match the trainable
    parameter capacity of the Hierarchical encoder (exact relative differences of
    **-2.61%** on WESAD 60s, **+3.83%** on WESAD 30s and Stress-Predict 30s, and
    **+4.80%** on Stress-Predict 60s). Introduced specifically to address
    **Reviewer Point 5** and **Reviewer 3's follow-up** (matched-capacity ablation across both WESAD and Stress-Predict).

For SSL experiments, two independently augmented views of each
physiological window are encoded and optimized using the established
NT-Xent contrastive objective. The projection head is used during
pretraining and discarded before downstream stress classification.

The principal methodological contribution of this work is the
**hierarchical temporal patch representation and its evaluation within a
contrastive SSL framework**. NT-Xent itself is an established
contrastive-learning objective and is not presented as a new
contribution.

## Repository Structure

``` text
TimePatch-SSL/
├── README.md
├── LICENSE
├── requirements.txt
├── environment.yml
│
├── configs/
│   ├── selected_hyperparameters.csv
│   └── experiment_configs/
│
├── splits/
│   ├── wesad_subject_splits.json
│   ├── stress_predict_subject_splits.json
│   └── experiment4_split.json
│
├── src/
│   ├── preprocessing.py
│   ├── datasets.py
│   ├── augmentations.py
│   ├── models.py
│   ├── ssl.py
│   ├── train.py
│   ├── evaluate.py
│   └── statistics.py
│
├── experiments/
│   ├── experiment1_within_supervised.py
│   ├── experiment2_cross_dataset.py
│   ├── experiment3_within_ssl.py
│   └── experiment4_combined.py
│
├── analysis/
│   ├── dataset_counts.py
│   ├── table2_audit.py
│   ├── table4_statistics.py
│   ├── parameter_counts.py
│   └── per_class_metrics.py
│
├── experiments_audit/                         # Reviewer 3 Points 1–4 & Point 7 Audit Suite
│   ├── audit_point1.py                        # Point 1: Cross-dataset anomalous AUROC (0.1049) & ROC audit
│   ├── audit_point2.py                        # Point 2: Per-class metrics, confusion matrices & Always-Stress audit
│   ├── audit_point3.py                        # Point 3: Standard vs. Hierarchical convergence & stability curves
│   ├── audit_point4.py                        # Point 4: Training fairness, Optuna budget & leakage audit
│   └── audit_point7_master_consistency.py     # Point 7: Master consistency audit (Tables 2, 3, 4, 6, 7)
│
├── ablation_matched_capacity/                 # Reviewer 3 Points 5 & 6: Capacity-matched ablation suite
│   ├── README.md                              # Scientific rationale & execution guide
│   ├── models.py                              # StandardMatchedTransformer & matched depth logic
│   ├── verify_capacity.py                     # Parameter parity verification script
│   ├── run_ablation.py                        # LOGO 3-model ablation runner (WESAD & Stress-Predict)
│   ├── generate_deliverables.py               # Extended Table 7 & comparative figure generator
│   └── statistical_analysis.py                # Point 6: Paired Wilcoxon, t/bootstrap CIs, r_rb, d_z & plots
│
├── data_cache/                                # Preprocessed portable window caches (.npz) for fast reproduction
│   ├── wesad_30s.npz
│   ├── wesad_60s.npz
│   ├── stress_predict_30s.npz
│   └── stress_predict_60s.npz
│
├── docs/                                      # Reviewer 3 Technical Reports & Supplementary Material Guides
│   ├── reviewer_3_master_response_and_traceability.md
│   ├── reviewer_3_experiments_report_points_1_to_4.md
│   ├── reviewer_3_point5_and_6_ablation_statistical_report.md
│   └── supplementary_material_figures_and_tables_guide.md
│
├── results/
│   ├── README.md                              # Global results overview & navigation guide
│   ├── metadata/
│   │   └── selected_hyperparameters_60s_from_colab.json
│   ├── predictions/
│   │   ├── participant_metrics_60s_from_colab.csv
│   │   ├── table4_statistics_60s_from_colab.csv
│   │   └── audit_point1_stresspredict_to_wesad_30s_standard.csv
│   ├── tables/
│   │   ├── README.md                          # Tables guide (Manuscript, Audit & Ablation)
│   │   ├── reported_metrics_summary.csv       # Unified summary of manuscript Exp 1–4
│   │   ├── reviewer3_evidence_traceability_matrix.csv
│   │   ├── audit_point7_tables_consistency_summary.csv
│   │   ├── audit_point7_positive_class_verification.csv
│   │   ├── audit_point1_auroc_table.csv
│   │   ├── audit_point2_per_class_metrics.csv
│   │   ├── audit_point2_always_stress_comparison.csv
│   │   ├── audit_point3_convergence_summary.csv
│   │   ├── audit_point4_fairness_hyperparameters_budget.csv
│   │   ├── source_result_summaries/           # Raw Colab logs for Exp 1–4
│   │   └── ablation_matched_capacity/         # Capacity-matched ablation tables (WESAD & Stress-Predict)
│   │       ├── README.md                      # Detailed ablation tables summary
│   │       ├── table7_extended_ablation_summary.csv
│   │       ├── ablation_matched_capacity_parameters.csv
│   │       ├── ablation_matched_capacity_paired_comparisons.csv
│   │       ├── ablation_matched_capacity_statistical_analysis.csv
│   │       ├── ablation_matched_capacity_results_stress_predict_60s.csv
│   │       ├── ablation_matched_capacity_participants_stress_predict_60s.csv
│   │       ├── ablation_matched_capacity_results_stress_predict_30s.csv
│   │       ├── ablation_matched_capacity_participants_stress_predict_30s.csv
│   │       ├── ablation_matched_capacity_results_wesad_60s.csv
│   │       ├── ablation_matched_capacity_participants_wesad_60s.csv
│   │       ├── ablation_matched_capacity_results_wesad_30s.csv
│   │       └── ablation_matched_capacity_participants_wesad_30s.csv
│   └── figures/
│       ├── README.md
│       ├── audit_point1_roc_curve.png
│       ├── audit_point2_confusion_matrices.png
│       ├── audit_point3_loss_curves.png
│       ├── audit_point3_val_metrics_curves.png
│       ├── ablation_matched_capacity_stress_predict.{png,pdf}
│       ├── ablation_matched_capacity_all_datasets.{png,pdf}
│       ├── ablation_paired_differences_stress_predict.{png,pdf}
│       └── ablation_paired_differences_summary.{png,pdf}
│
└── notebooks/
    └── original_colab.ipynb
```


## Datasets

### WESAD

WESAD is a publicly available multimodal wearable dataset. The study uses recordings from **15 subjects** collected under a controlled laboratory protocol with a chest-mounted RespiBAN device and an Empatica E4 wristband. The original protocol includes baseline, stress, amusement, and meditation conditions.

For TimePatch-SSL, the wrist-derived **BVP, EDA, and TEMP** modalities are used. The original stress condition is mapped to **stress (`y = 1`)**, while retained non-stress conditions are mapped to **non-stress (`y = 0`)**.

### Stress-Predict

Stress-Predict is a publicly available physiological stress dataset containing recordings from **35 healthy volunteers** exposed to stress-inducing tasks, including the **Stroop color test, Trier Social Stress Test, and Hyperventilation Provocation Test**, interleaved with rest periods.

For TimePatch-SSL, **BVP, EDA, and TEMP** are used. Stress-inducing periods are mapped to **stress (`y = 1`)**, and rest periods are mapped to **non-stress (`y = 0`)**.

The datasets are **not redistributed** in this repository. Users must obtain them from their original/public sources in accordance with their respective terms of use. The exact dataset citations used by the manuscript should be retained in the accompanying publication.

## Data Preparation

Expected local layout:

``` text
data/
├── WESAD/
└── Stress-Predict/
```

The `data/` directory should not be committed to the repository.

The preprocessing pipeline performs dataset-specific signal loading,
synchronization/resampling, robust normalization, missing-value
handling, window construction, and label-consistency filtering.

BVP is converted from its original sampling frequency to the common
sampling rate used by the model. The public implementation in
`src/preprocessing.py` should be treated as the authoritative
specification of the exact resampling procedure.


## Windowing and Label Filtering

Experiments are conducted using **30-second and 60-second windows**.

A window is retained only when its stress-label proportion satisfies the
label-consistency criterion defined in the manuscript:

-   stress proportion `<= 0.20` → non-stress;
-   stress proportion `>= 0.80` → stress;
-   intermediate windows are excluded.

Exact subject, retained-window, and class counts after filtering are
generated by:

``` bash
python analysis/dataset_counts.py
```

The resulting statistics should be stored under `results/tables/`.

## Experimental Protocols

### Experiment 1 --- Within-Dataset Supervised Evaluation

The Standard and Hierarchical encoders are trained from random
initialization using supervised learning.

Evaluation uses subject-level leave-one-group-out (LOGO) validation so
that windows from the held-out participant are excluded from model
development.

### Experiment 2 --- Cross-Dataset Transfer

Cross-dataset evaluation is performed in both directions:

``` text
WESAD → Stress-Predict
Stress-Predict → WESAD
```

For each direction, model development is restricted to the **source
dataset**.

The source data are divided into an **85% training partition and 15%
validation partition**. SSL pretraining and supervised fine-tuning use
source data only. The complete encoder is fine-tuned end-to-end, and
checkpoint/model selection is based on source-validation F1.

The target dataset is used only for final evaluation. Predictions use a
fixed classification threshold of **0.5**, and target labels are not
used for model fitting, checkpoint selection, or threshold calibration.

Exact split definitions needed to reproduce the reported experiments are
provided in `splits/`.

### Experiment 3 --- Within-Dataset SSL Evaluation

Contrastive self-supervised pretraining is followed by supervised
fine-tuning and subject-level LOGO evaluation.

The Standard and Hierarchical encoders are evaluated under the same
downstream protocol.

### Experiment 4 --- Combined-Dataset Evaluation

WESAD and Stress-Predict are combined after preprocessing.

The reported protocol uses subject-level partitions, with held-out
participants excluded from training, validation, and SSL pretraining.
The exact participant IDs used in the reported split are provided in:

``` text
splits/experiment4_split.json
```

This is important because results from a single subject-level partition
characterize that particular split and should not be interpreted as an
estimate of variability across all possible participant partitions.

## Self-Supervised Learning

SSL pretraining uses two independently augmented views of each
physiological window.

The augmentations used in the study include:

-   additive Gaussian noise;
-   amplitude scaling;
-   temporal shifting;
-   contiguous masking.

The encoded representations are passed through a projection head and
optimized using the **NT-Xent** contrastive loss.

The projection head is discarded after pretraining, and the pretrained
encoder is subsequently fine-tuned for binary stress classification.

Exact augmentation probabilities, ranges, NT-Xent temperature,
projection-head configuration, and other selected parameters are
provided in the experiment configuration files.

## Hyperparameter Optimization

Hyperparameters are optimized using Optuna with the Tree-structured
Parzen Estimator (TPE).

The repository distinguishes between:

1.  the **hyperparameter search space**, and
2.  the **actual selected configuration used for each reported result**.

The latter is provided in:

``` text
configs/selected_hyperparameters.csv
```

This file should be used when reproducing the manuscript results.

## Reproducibility

The experiments use fixed random seeds where specified by the
corresponding configuration.

The software environment is documented through:

``` text
requirements.txt
environment.yml
```

Before publication, these files should contain the exact package
versions used to generate the final manuscript results, including at
minimum:

-   Python
-   PyTorch
-   NumPy
-   SciPy
-   scikit-learn
-   Optuna

To avoid machine-specific behavior, no absolute local paths are required
by the public implementation.

## Model Parameter Counts

Trainable parameter counts for the reported Standard and Hierarchical
configurations can be regenerated using:

``` bash
python analysis/parameter_counts.py
```

These values are reported alongside the selected model configurations to
make differences in model capacity explicit.

## Table 2 Audit: Cross-Dataset Evaluation

Because cross-dataset distribution shift can produce substantial
differences between threshold-dependent metrics and
threshold-independent AUROC, the repository includes an explicit audit
of the cross-dataset results.

Run:

``` bash
python analysis/table2_audit.py
```

The audit should regenerate, from saved true labels and predicted stress
probabilities:

-   Accuracy
-   F1 score
-   Balanced Accuracy
-   AUROC
-   confusion matrices
-   ROC curves
-   participant-level metrics

Prediction files should include at least:

``` text
subject_id
y_true
probability_stress
y_pred
```

The audit explicitly verifies that the probability used for AUROC
corresponds to the positive **stress** class.

Generated outputs are stored under:

``` text
results/predictions/
results/tables/
results/figures/
```

## Statistical Analysis

Subject-level comparisons between SSL-pretrained and
supervised-from-scratch models use paired **Wilcoxon signed-rank
tests**.

The reported analysis evaluates:

-   F1 score
-   Balanced Accuracy
-   AUROC

Rank-biserial correlation is reported as an effect-size measure. For the
six planned comparisons, the manuscript uses a Bonferroni-corrected
significance threshold of:

``` text
alpha = 0.05 / 6 ≈ 0.0083
```

Table 4 and its underlying participant-level values can be regenerated
using:

``` bash
python analysis/table4_statistics.py
```

The statistical analysis should be interpreted as a test of whether SSL
provides a systematic participant-level advantage over supervised
initialization. The manuscript does **not** claim a systematic SSL
advantage when this comparison is not supported by the subject-level
statistical results.

## Per-Class Metrics

Per-class precision, recall, and F1 can be regenerated using:

``` bash
python analysis/per_class_metrics.py
```

These metrics complement aggregate Accuracy, F1, Balanced Accuracy, and
AUROC by making performance for both stress and non-stress classes
explicit.

## Reviewer Point 5: Matched-Capacity Ablation Study

### Motivation and Scientific Context

During peer review, the following objection was raised regarding novelty and causal attribution:

> *"The proposed encoder is described as a conv-plus-local-Transformer replacement for a linear patch projection (pp. 6–8), but the comparisons principally contrast it with the authors' own standard embedding. A convincing contribution requires matched-capacity ablations (convolution only, local Transformer only, SSL only, and their combination), parameter/computation reporting, and meaningful published baselines rerun under the same subject-level splits and preprocessing. Claims that the representation suppresses noise or motion artifacts (p. 7) are mechanistic claims with no artifact analysis or ablation and should be removed or supported."*

In the original manuscript, the Hierarchical encoder utilized ~1.73M parameters (60s) and ~7.82M parameters (30s), whereas the baseline Standard encoder had ~692K parameters (60s) and ~4.96M parameters (30s). A critical question was therefore whether the performance of the Hierarchical encoder stemmed from its **hierarchical temporal patch representation** (multi-scale inductive bias) or simply from its **larger parameter capacity** (~2.17× parameters in default configurations).

To resolve this, we implemented a dedicated capacity-matched model variant: **Standard-Matched**, isolated cleanly in [`ablation_matched_capacity/`](ablation_matched_capacity/README.md).

### Model Design and Capacity Matching

1. **Standard:** Direct 1D linear patch projection followed by a global Transformer encoder of base depth $L$.
2. **Hierarchical:** Intra-patch local Transformer (TimeMAE patch encoder) followed by a global Transformer encoder of base depth $L$.
3. **Standard-Matched:** Retains the simple 1D linear patch projection but deepens the global Transformer to $L' > L$ layers such that its total trainable parameter count closely approximates that of the Hierarchical encoder (**-2.61%** on WESAD 60s, **+3.83%** on WESAD 30s and Stress-Predict 30s, and **+4.80%** on Stress-Predict 60s).

The capacity matching formula determines $L'$ dynamically:
$$\Delta P = P_{\text{Hierarchical}} - P_{\text{Standard}}$$
$$L' = L + \text{round}\left(\frac{\Delta P}{P_{\text{layer}}}\right)$$

### 1. Model Parameter Parity Audit

The script [`ablation_matched_capacity/verify_capacity.py`](ablation_matched_capacity/verify_capacity.py) audits parameter parity across all experimental setups in the paper. The generated audit is stored in [`results/tables/ablation_matched_capacity/ablation_matched_capacity_parameters.csv`](results/tables/ablation_matched_capacity/ablation_matched_capacity_parameters.csv):

| Configuration | Base Depth ($L$) | Matched Depth ($L'$) | Standard Trainable | Hierarchical Trainable | Standard-Matched Trainable | Parity Ratio | Rel. Diff (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Base Default** | 4 | 9 | 887,234 | 1,923,782 | 1,878,594 | 0.9765 | -2.35% |
| **WESAD 60s** | 3 | 8 | 692,034 | 1,728,582 | 1,683,394 | 0.9739 | **-2.61%** |
| **WESAD 30s** | 6 | 10 | 4,959,554 | 7,819,078 | 8,118,594 | 1.0383 | **+3.83%** |
| **Stress-Predict 60s** | 4 | 8 | 3,380,034 | 6,239,558 | 6,539,074 | 1.0480 | **+4.80%** |
| **Stress-Predict 30s** | 6 | 10 | 4,959,554 | 7,819,078 | 8,118,594 | 1.0383 | **+3.83%** |

### 2. Experimental Evaluation on WESAD and Stress-Predict (Extended Table 7)

The ablation suite evaluates all three models (`Standard`, `Hierarchical`, and `Standard-Matched`) under identical Experiment 1 conditions:
- **Evaluation Protocol:** Strict Leave-One-Group-Out (LOGO) cross-validation across all 15 WESAD subjects ($n=15$) and all 33 retained Stress-Predict participants ($n=33$, $S02 \dots S35$ excluding $S01, S06$).
- **Preprocessing:** Per-subject robust normalization (`median` and `IQR`), matching the manuscript's pipeline.
- **Windowing & Overlap:** 50% overlap (60s window / 30s stride; 30s window / 15s stride).
- **Statistics:** Sample standard deviation ($ddof=1$), 95% Confidence Intervals via two-sided Student's $t$-distribution, and paired subject-level Wilcoxon signed-rank tests ([`ablation_matched_capacity_paired_comparisons.csv`](results/tables/ablation_matched_capacity/ablation_matched_capacity_paired_comparisons.csv)).

#### WESAD 60-Second Windows ($n=15$ Participants)

Stored in [`results/tables/ablation_matched_capacity/ablation_matched_capacity_results_wesad_60s.csv`](results/tables/ablation_matched_capacity/ablation_matched_capacity_results_wesad_60s.csv):

| Variant | Params (Rel. Diff) | Accuracy (Mean ± Std) | 95% CI | F1-Score (Mean ± Std) | 95% CI | Balanced Acc (Mean ± Std) | 95% CI | AUROC (Mean ± Std) | 95% CI |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Standard** | 692,034 (-59.97%) | 0.8752 ± 0.1587 | [0.7873, 0.9631] | 0.7746 ± 0.3113 | [0.6022, 0.9470] | 0.8557 ± 0.2026 | [0.7435, 0.9679] | 0.9162 ± 0.1616 | [0.8267, 1.0000] |
| **Hierarchical** | 1,728,582 (0.00%) | 0.8602 ± 0.1722 | [0.7649, 0.9556] | 0.7653 ± 0.3159 | [0.5904, 0.9403] | 0.8466 ± 0.2087 | [0.7310, 0.9621] | 0.9098 ± 0.1656 | [0.8180, 1.0000] |
| **Standard-Matched** | 1,683,394 (**-2.61%**) | 0.8614 ± 0.1820 | [0.7606, 0.9622] | 0.7430 ± 0.3539 | [0.5470, 0.9390] | 0.8342 ± 0.2351 | [0.7040, 0.9643] | 0.8688 ± 0.2457 | [0.7327, 1.0000] |

#### WESAD 30-Second Windows ($n=15$ Participants)

Stored in [`results/tables/ablation_matched_capacity/ablation_matched_capacity_results_wesad_30s.csv`](results/tables/ablation_matched_capacity/ablation_matched_capacity_results_wesad_30s.csv):

| Variant | Params (Rel. Diff) | Accuracy (Mean ± Std) | 95% CI | F1-Score (Mean ± Std) | 95% CI | Balanced Acc (Mean ± Std) | 95% CI | AUROC (Mean ± Std) | 95% CI |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Standard** | 4,959,554 (-36.57%) | 0.8661 ± 0.1272 | [0.7957, 0.9365] | 0.7604 ± 0.2639 | [0.6143, 0.9065] | 0.8462 ± 0.1718 | [0.7510, 0.9413] | 0.9004 ± 0.1765 | [0.8026, 0.9982] |
| **Hierarchical** | 7,819,078 (0.00%) | 0.8630 ± 0.2203 | [0.7410, 0.9850] | 0.7721 ± 0.3572 | [0.5743, 0.9699] | 0.8417 ± 0.2582 | [0.6987, 0.9847] | 0.8582 ± 0.2882 | [0.6985, 1.0000] |
| **Standard-Matched** | 8,118,594 (**+3.83%**) | 0.8678 ± 0.1779 | [0.7692, 0.9664] | 0.7753 ± 0.2993 | [0.6095, 0.9411] | 0.8545 ± 0.1999 | [0.7438, 0.9652] | 0.8798 ± 0.2181 | [0.7589, 1.0000] |

#### Stress-Predict 60-Second Windows ($n=33$ Participants)

Generated by `ablation_matched_capacity/run_ablation.py --dataset stress_predict --window-seconds 60` and stored in [`results/tables/ablation_matched_capacity/ablation_matched_capacity_results_stress_predict_60s.csv`](results/tables/ablation_matched_capacity/ablation_matched_capacity_results_stress_predict_60s.csv):

| Variant | Params (Rel. Diff) | Accuracy (Mean ± Std) | 95% CI | F1-Score (Mean ± Std) | 95% CI | Balanced Acc (Mean ± Std) | 95% CI | AUROC (Mean ± Std) | 95% CI |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Standard** | 3,380,034 (-45.83%) | 0.8254 ± 0.1644 | [0.7671, 0.8837] | 0.8495 ± 0.1492 | [0.7966, 0.9025] | 0.8231 ± 0.1730 | [0.7618, 0.8845] | 0.9020 ± 0.1537 | [0.8474, 0.9565] |
| **Hierarchical** | 6,239,558 (0.00%) | 0.8282 ± 0.1724 | [0.7671, 0.8893] | 0.8550 ± 0.1470 | [0.8028, 0.9071] | 0.8226 ± 0.1841 | [0.7573, 0.8879] | 0.8996 ± 0.1548 | [0.8447, 0.9545] |
| **Standard-Matched** | 6,539,074 (**+4.80%**) | 0.8332 ± 0.1545 | [0.7785, 0.8880] | 0.8579 ± 0.1347 | [0.8101, 0.9056] | 0.8314 ± 0.1644 | [0.7731, 0.8896] | 0.8878 ± 0.1754 | [0.8256, 0.9500] |

#### Stress-Predict 30-Second Windows ($n=33$ Participants)

Generated by `ablation_matched_capacity/run_ablation.py --dataset stress_predict --window-seconds 30` and stored in [`results/tables/ablation_matched_capacity/ablation_matched_capacity_results_stress_predict_30s.csv`](results/tables/ablation_matched_capacity/ablation_matched_capacity_results_stress_predict_30s.csv):

| Variant | Params (Rel. Diff) | Accuracy (Mean ± Std) | 95% CI | F1-Score (Mean ± Std) | 95% CI | Balanced Acc (Mean ± Std) | 95% CI | AUROC (Mean ± Std) | 95% CI |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Standard** | 4,959,554 (-36.57%) | 0.8351 ± 0.1508 | [0.7816, 0.8885] | 0.8609 ± 0.1302 | [0.8148, 0.9071] | 0.8307 ± 0.1594 | [0.7741, 0.8872] | 0.9030 ± 0.1478 | [0.8506, 0.9554] |
| **Hierarchical** | 7,819,078 (0.00%) | 0.8222 ± 0.1581 | [0.7661, 0.8783] | 0.8517 ± 0.1354 | [0.8036, 0.8997] | 0.8163 ± 0.1693 | [0.7563, 0.8763] | 0.8947 ± 0.1561 | [0.8393, 0.9500] |
| **Standard-Matched** | 8,118,594 (**+3.83%**) | 0.8178 ± 0.1639 | [0.7597, 0.8759] | 0.8447 ± 0.1425 | [0.7942, 0.8952] | 0.8161 ± 0.1742 | [0.7544, 0.8779] | 0.8874 ± 0.1606 | [0.8304, 0.9443] |

### 3. Key Scientific Conclusions for the Manuscript Revision

1. **Exact Capacity Parity Reporting:**
   The parameter audit shows that `Standard-Matched` matches `Hierarchical` with exact relative differences of **-2.61%** (WESAD 60s), **+3.83%** (WESAD 30s and Stress-Predict 30s), and **+4.80%** (Stress-Predict 60s) due to discrete global Transformer layer increments.
2. **Capacity vs. Architecture on Stress-Predict & WESAD (Experiment 1 Protocol):**
   - On **Stress-Predict 30s**, scaling parameter count via deeper global layers (`Standard-Matched`, 8.12M) degrades supervised performance compared to the lighter `Standard` baseline (4.96M) across all metrics (F1 drops from 0.8609 to 0.8447, Wilcoxon $p=0.0201$), whereas `Hierarchical` (7.82M) retains slightly higher Accuracy (0.8222 vs. 0.8178), F1 (0.8517 vs. 0.8447), and AUROC (0.8947 vs. 0.8874) than `Standard-Matched` ($p \ge 0.186$).
   - On **Stress-Predict 60s**, all three models perform within a tight range (F1: 0.8495–0.8579), with no statistically significant difference between `Hierarchical` and `Standard-Matched` ($p \ge 0.307$ across all metrics).
   - Consequently, under purely supervised within-dataset training from scratch (Experiment 1), increasing parameter capacity does not yield a consistent performance advantage, and the manuscript's discussion in §4.6 is restricted accordingly.

### 4. Reproduction Commands

To audit parameter counts and rerun the ablation study from scratch:

```bash
# 1. Verify parameter parity across all configurations
python ablation_matched_capacity/verify_capacity.py

# 2. Run fast dry-run on synthetic data (3 seconds verification)
python ablation_matched_capacity/run_ablation.py --dry-run

# 3. Run complete LOSO ablation on Stress-Predict 60s and 30s (GPU)
python ablation_matched_capacity/run_ablation.py --dataset stress_predict --window-seconds 60 --device cuda
python ablation_matched_capacity/run_ablation.py --dataset stress_predict --window-seconds 30 --device cuda

# 4. Run complete LOSO ablation on WESAD 60s and 30s (GPU)
python ablation_matched_capacity/run_ablation.py --dataset wesad --window-seconds 60 --device cuda
python ablation_matched_capacity/run_ablation.py --dataset wesad --window-seconds 30 --device cuda

# 5. Generate Extended Table 7, paired Wilcoxon comparisons, and comparative plots
python ablation_matched_capacity/generate_deliverables.py

# 6. Run complete paired statistical analysis (CIs, bootstrap, effect sizes, paired difference figures)
python ablation_matched_capacity/statistical_analysis.py
```

## Reviewer 3 Audit & Reproducibility Suite (Points 1–7)

To ensure complete transparency, positive-class consistency, and end-to-end reproducibility across Manuscript **Tables 2, 3, 4, 6, and 7**, the repository includes an automated 7-point audit suite (`experiments_audit/` and `ablation_matched_capacity/`), portable preprocessed window caches (`data_cache/*.npz`), and publication-ready Supplementary Material guides (`docs/`).

### Positive-Class & Evaluation Convention (Verified Across All Scripts)
- **Positive Class:** `y = 1` (**Stress**) across all datasets (`WESAD` and `Stress-Predict`), protocols (Experiments 1–4), and audits (Points 1–7).
- **Predicted Probability:** `p_stress = torch.softmax(logits, dim=1)[:, 1]`, verified programmatically by [`experiments_audit/audit_point7_master_consistency.py`](experiments_audit/audit_point7_master_consistency.py) and exported in [`results/tables/audit_point7_positive_class_verification.csv`](results/tables/audit_point7_positive_class_verification.csv).
- **Default Decision Threshold:** $\tau = 0.50$ (`y_pred = (p_stress >= 0.50).astype(int)`).
- **Seeds & Determinism:** Base random seed `42` (`set_seed(42)`), deterministic per-participant fold seeds (`seed + fold_idx`), and `GroupShuffleSplit(n_splits=5, test_size=0.20, random_state=42)` for multi-split sensitivity analysis.

### Master Traceability & Reproduction Commands (Points 1–7)

See [`docs/reviewer_3_master_response_and_traceability.md`](docs/reviewer_3_master_response_and_traceability.md) and [`results/tables/reviewer3_evidence_traceability_matrix.csv`](results/tables/reviewer3_evidence_traceability_matrix.csv) for the full mapping between Reviewer 3's comments and repository deliverables:

```bash
# Point 1: Cross-dataset probability calibration, ROC curve, and threshold sweep (Table 2 audit)
python experiments_audit/audit_point1.py

# Point 2: Per-class metrics (Precision/Recall/F1 for Non-Stress & Stress) and confusion matrices (Tables 2, 3, 6)
python experiments_audit/audit_point2.py

# Point 3: SSL pretraining & fine-tuning learning curves (Experiment 3 / Table 3)
python experiments_audit/audit_point3.py

# Point 4: Multi-split sensitivity analysis (K=5 subject-grouped splits, Experiment 4 / Table 6)
python experiments_audit/audit_point4.py

# Point 5: Capacity-matched LOGO ablation on Stress-Predict & WESAD (Extended Table 7)
python ablation_matched_capacity/verify_capacity.py
python ablation_matched_capacity/run_ablation.py --dataset stress_predict --window-seconds 60 --device cuda
python ablation_matched_capacity/run_ablation.py --dataset stress_predict --window-seconds 30 --device cuda
python ablation_matched_capacity/generate_deliverables.py

# Point 6: Paired statistical analysis (Wilcoxon, Student-t & bootstrap 95% CIs, r_rb, Cohen's d_z, Figures S7–S8)
python ablation_matched_capacity/statistical_analysis.py

# Point 7: Master consistency audit across Tables 2, 3, 4, 6, 7 & positive-class verification
python experiments_audit/audit_point7_master_consistency.py
```

## Running the Experiments

After installing dependencies and obtaining the datasets, the experimental protocols can be run separately:

``` bash
# Main manuscript benchmark experiments
python experiments/experiment1_within_supervised.py
python experiments/experiment2_cross_dataset.py
python experiments/experiment3_within_ssl.py
python experiments/experiment4_combined.py

# Reviewer Point 5: Capacity-matched ablation study (Standard vs. Hierarchical vs. Standard-Matched)
python ablation_matched_capacity/verify_capacity.py
python ablation_matched_capacity/run_ablation.py --window-seconds 60 --device cuda
python ablation_matched_capacity/run_ablation.py --window-seconds 30 --device cuda
```

Experiment-specific options should be defined through configuration
files rather than hard-coded local paths.



## Installation

Create a virtual environment and install the required dependencies.

``` bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows:

``` text
.venv\Scripts\activate
```

For Conda environments:

``` bash
conda env create -f environment.yml
conda activate timepatch-ssl
```

The exact environment name should match the finalized `environment.yml`.

## Expected Outputs

Reproduction scripts should write outputs to `results/` without
modifying source data.

Structured directory layout:

``` text
results/
├── README.md                                  # Directory organization and benchmark guide
├── metadata/
│   └── selected_hyperparameters_60s_from_colab.json
├── predictions/
│   ├── participant_metrics_60s_from_colab.csv # Participant-level metrics from Colab
│   └── table4_statistics_60s_from_colab.csv   # Wilcoxon signed-rank statistics
├── tables/
│   ├── README.md                              # Tables guide
│   ├── reported_metrics_summary.csv           # Consolidated table of Experiments 1–4
│   ├── source_result_summaries/               # 18 original Colab execution summaries
│   └── ablation_matched_capacity/             # Reviewer Point 5 ablation results
│       ├── README.md                          # Scientific summary of ablation findings
│       ├── ablation_matched_capacity_parameters.csv          # Parameter parity audit
│       ├── ablation_matched_capacity_results_wesad_60s.csv   # 60s summary metrics
│       ├── ablation_matched_capacity_participants_wesad_60s.csv # 60s subject breakdown
│       ├── ablation_matched_capacity_results_wesad_30s.csv   # 30s summary metrics
│       └── ablation_matched_capacity_participants_wesad_30s.csv # 30s subject breakdown
└── figures/
    ├── ROC curves
    └── confusion matrices
```


## Data Availability Statement

The datasets analyzed in this study are publicly available from their respective open-access sources and are not redistributed in this repository:

1. **WESAD (Wearable Stress and Affect Detection):**
   - **Repository:** UCI Machine Learning Repository
   - **URL:** [https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection](https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection)
   - **DOI:** [10.1145/3242969.3242985](https://doi.org/10.1145/3242969.3242985)
   - **Citation:** Schmidt, P., Reiss, A., Duerichen, R., Marberger, C., & Van Laerhoven, K. (2018). *Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection*. In Proceedings of the 20th ACM International Conference on Multimodal Interaction (ICMI '18), pp. 400–408.
   - **Access:** Publicly accessible for academic research without restrictions.

2. **Stress-Predict (Stress-ID):**
   - **Repository:** Open-access release repository
   - **URL:** [https://github.com/dalinvance/Stress-Predict-Dataset](https://github.com/dalinvance/Stress-Predict-Dataset)
   - **DOI:** [10.1371/journal.pone.0289878](https://doi.org/10.1371/journal.pone.0289878)
   - **Citation:** Vance, D., et al. (2023). *Stress-Predict: A dataset of physiological signals for stress prediction using consumer-grade wearables*. PLOS ONE.
   - **Access:** Open access under Creative Commons licensing.

All subject-level partition definitions, participant splits, and cross-validation index definitions are openly provided in machine-readable JSON format in the [`splits/`](splits/README.md) directory.

## Code Availability Statement

The complete source code, neural network implementations (Standard, Hierarchical, and Capacity-Matched Standard-Matched variants), self-supervised pretraining modules, experiment runners, configuration files, and evaluation scripts supporting the findings of this study are openly available on GitHub at:

- **Repository:** [https://github.com/MatiasMaravi/TimePatch-SSL](https://github.com/MatiasMaravi/TimePatch-SSL)
- **License:** GNU General Public License v3.0 (GPL-3.0)
- **Primary Modules:**
  - Core pipeline and models: [`src/`](src/)
  - Main manuscript benchmarks (Experiments 1–4): [`experiments/`](experiments/)
  - Reviewer Point 5 matched-capacity ablation: [`ablation_matched_capacity/`](ablation_matched_capacity/)
  - Quantitative reproduction tables and participant logs: [`results/tables/`](results/tables/)
  - Exact environment specifications: [`requirements.txt`](requirements.txt) and [`environment.yml`](environment.yml)


## Citation

If you use this repository, please cite the associated manuscript.

``` bibtex
@article{timepatch_ssl,
  title   = {TimePatch-SSL: Contrastive Self-Supervised Learning with Hierarchical Time-Patch Transformers for Multimodal Stress Detection},
  author  = {Leonardo Isidro-Salazar, Matias Maravi-Anyosa, Rensso Mora-Colque, Cristian Lopez, and Aurea Soriano-Vargas},
  journal = {Sensors},
  year    = {YEAR TO BE ADDED}
}
```

The citation information should be updated after publication.

## License

This software is released under the **GNU General Public License, Version 3, 29 June 2007 (GPL-3.0)**.

```text
TimePatch-SSL
Copyright (C) 2026 Leonardo Isidro-Salazar, Matias Maravi-Anyosa,
Rensso Mora-Colque, Cristian Lopez, and Aurea Soriano-Vargas

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.
```

The complete **GNU GENERAL PUBLIC LICENSE, Version 3, 29 June 2007** is provided in the repository's `LICENSE` file.

## Contact

For questions concerning reproduction of the experiments, please open an
issue in this repository.

------------------------------------------------------------------------

### Reproducibility Status

This repository is being prepared as the reproducibility package
accompanying the revised manuscript. Any placeholder marked **"Before
publication"** must be replaced or verified before the repository URL is
included in the manuscript.
