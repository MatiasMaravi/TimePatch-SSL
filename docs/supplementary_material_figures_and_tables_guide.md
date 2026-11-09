# Supplementary Material: Figures and Tables Catalog with Publication Legends

This document provides the complete catalog of **Supplementary Figures (Figures S1–S8)** and **Supplementary Tables (Tables S1–S8)** generated across the diagnostic audits and capacity-matched ablation experiments to accompany the manuscript. Both English (ready to include in the *Sensors* Supplementary Material) and Spanish legends are provided for each item.

---

## Part I. Supplementary Figures (`results/figures/`)

### Figure S1 — ROC Curve and Probability-Inversion Audit (Audit Point 1)
- **File:** [`results/figures/audit_point1_roc_curve.png`](../results/figures/audit_point1_roc_curve.png)
- **Reproduction Script:** `python experiments_audit/audit_point1.py`
- **Manuscript Reference:** Section 4.3 (Cross-Dataset Transfer, Table 3)
- **Caption (English — Ready for Manuscript/Supplement):**
  > **Figure S1. Receiver Operating Characteristic (ROC) audit for cross-dataset transfer from Stress-Predict to WESAD using the Standard encoder on 30-second windows ($n = 2,188$ test windows).** The solid blue curve depicts the ROC trajectory computed using the positive stress-class probability $p_{\text{stress}} = \text{softmax}(\text{logits})[:, 1]$ ($\text{AUROC} = 0.4844$), while the dashed orange curve depicts the complementary non-stress probability $1 - p_{\text{stress}}$ ($\text{AUROC} = 0.5156$), illustrating the exact mathematical symmetry $\text{AUROC}(p) + \text{AUROC}(1-p) = 1.0000$. At the fixed decision threshold $\tau = 0.5$, the 30 s Standard encoder collapses to predicting the majority stress class for 100% of target windows ($\text{Balanced Accuracy} = 0.5000$, $\text{Specificity} = 0.0000$).
- **Leyenda (Español):**
  > **Figura S1. Auditoría de la curva ROC para la transferencia *cross-dataset* de Stress-Predict a WESAD con el codificador Standard en ventanas de 30 s ($n = 2,188$ ventanas de prueba).** Se contrasta la evaluación sobre la probabilidad de la clase positiva de estrés $p_{\text{stress}} = \text{softmax}(\text{logits})[:, 1]$ ($\text{AUROC} = 0.4844$) frente a su complemento $1 - p_{\text{stress}}$ ($\text{AUROC} = 0.5156$), verificando la identidad $\text{AUROC}(p) + \text{AUROC}(1-p) = 1.0000$ y el colapso hacia la clase mayoritaria bajo umbral fijo $\tau = 0.5$.

---

### Figure S2 — Confusion Matrices Across Protocols and Architectures (Audit Point 2)
- **File:** [`results/figures/audit_point2_confusion_matrices.png`](../results/figures/audit_point2_confusion_matrices.png)
- **Reproduction Script:** `python experiments_audit/audit_point2.py`
- **Manuscript Reference:** Sections 4.2, 4.3, and 4.5 (Tables 2, 3, and 6)
- **Caption (English — Ready for Manuscript/Supplement):**
  > **Figure S2. Window-level confusion matrices across 12 evaluation configurations spanning Experiment 2 (cross-dataset Stress-Predict $\to$ WESAD), Experiment 1 (within-dataset Stress-Predict holdout), and Experiment 4 (combined dataset $\to$ Stress-Predict holdout) at 30 s and 60 s window lengths.** Each panel reports True Negatives (TN, Non-Stress), False Positives (FP), False Negatives (FN), and True Positives (TP, Stress) evaluated at the fixed threshold $\tau = 0.5$. On 30 s windows, the Standard encoder exhibits complete majority-class collapse ($\text{TN} = 0$, $\text{Specificity} = 0.0000$) in both cross-dataset and within-dataset holdout evaluations, whereas the Hierarchical encoder preserves class discrimination ($\text{Specificity} = 0.8108$, $\text{Balanced Accuracy} = 0.8269$ on Stress-Predict 30 s). At 60 s, the extended temporal context prevents collapse in both architectures.
- **Leyenda (Español):**
  > **Figura S2. Matrices de confusión a nivel de ventana para las 12 configuraciones auditadas en los Experimentos 2 (cruzado), 1 (intra-dataset holdout) y 4 (combinado) con ventanas de 30 s y 60 s.** Se reportan TN, FP, FN y TP bajo umbral fijo $\tau = 0.5$, evidenciando el colapso del modelo Standard a 30 s ($\text{TN} = 0$) y la preservación de especificidad en el modelo Hierarchical y en las ventanas de 60 s.

---

### Figure S3 — Training and Validation Loss Convergence Curves (Audit Point 3)
- **File:** [`results/figures/audit_point3_loss_curves.png`](../results/figures/audit_point3_loss_curves.png)
- **Reproduction Script:** `python experiments_audit/audit_point3.py`
- **Manuscript Reference:** Section 4.2 (Training Stability and Convergence Analysis)
- **Caption (English — Ready for Manuscript/Supplement):**
  > **Figure S3. Epoch-by-epoch training and validation cross-entropy loss trajectories for the Standard and Hierarchical encoders on Stress-Predict (30 s and 60 s windows).** Both architectures were trained under identical optimization settings (AdamW, learning rate $2 \times 10^{-4}$, weight decay $10^{-4}$, batch size 16) on the 28-subject development split. Smooth, monotonic loss reduction without NaN values, gradient explosion, or divergence confirms that both encoders optimize stably.
- **Leyenda (Español):**
  > **Figura S3. Curvas de pérdida (*cross-entropy loss*) de entrenamiento y validación época por época para los codificadores Standard y Hierarchical en Stress-Predict (30 s y 60 s).** Ambas arquitecturas convergen de manera monótona y estable sin valores NaN ni divergencia numérica.

---

### Figure S4 — Validation F1-Score and Balanced Accuracy Trajectories (Audit Point 3)
- **File:** [`results/figures/audit_point3_val_metrics_curves.png`](../results/figures/audit_point3_val_metrics_curves.png)
- **Reproduction Script:** `python experiments_audit/audit_point3.py`
- **Manuscript Reference:** Section 4.2 (Checkpoint Selection Audit)
- **Caption (English — Ready for Manuscript/Supplement):**
  > **Figure S4. Validation F1-score and Balanced Accuracy trajectories across training epochs for the Standard and Hierarchical encoders on Stress-Predict (30 s and 60 s windows).** Vertical markers indicate the best-performing validation F1 checkpoint selected for downstream inference (epochs 6–8). The Standard encoder achieves strong internal validation F1 ($0.8865$ at 30 s; $0.8986$ at 60 s), demonstrating that its degradation on unseen target domains/participants at 30 s is a generalization property rather than an optimization failure.
- **Leyenda (Español):**
  > **Figura S4. Evolución época por época del F1-score y Balanced Accuracy de validación para Standard y Hierarchical en Stress-Predict (30 s y 60 s).** El modelo Standard alcanza un F1 de validación interna elevado ($0.8865$ a 30 s y $0.8986$ a 60 s), descartando fallas de optimización durante el entrenamiento.

---

### Figure S5 — Capacity-Matched Ablation on Stress-Predict (Audit Point 5)
- **Files:**
  - PNG: [`results/figures/ablation_matched_capacity_stress_predict.png`](../results/figures/ablation_matched_capacity_stress_predict.png)
  - PDF: [`results/figures/ablation_matched_capacity_stress_predict.pdf`](../results/figures/ablation_matched_capacity_stress_predict.pdf)
- **Reproduction Script:** `python ablation_matched_capacity/generate_deliverables.py`
- **Manuscript Reference:** Section 4.6 (Extended Table 7 — Capacity-Matched Ablation)
- **Caption (English — Ready for Manuscript/Supplement):**
  > **Figure S5. Capacity-matched ablation on Stress-Predict ($n = 33$ participants) under the Experiment 1 supervised Leave-One-Group-Out (LOGO) protocol for 30 s (left) and 60 s (right) windows.** Bars indicate participant-level mean Accuracy, F1-score, Balanced Accuracy, and AUROC ($\pm 1\text{ SD}$, upper-bounded at $1.0$) for `Standard` ($L=6$, $4.96\text{M}$ params at 30 s; $L=4$, $3.38\text{M}$ at 60 s), `Hierarchical` ($L=6$, $7.82\text{M}$ at 30 s; $L=4$, $6.24\text{M}$ at 60 s), and `Standard-Matched` ($L'=10$, $8.12\text{M}$ [+3.83%] at 30 s; $L'=8$, $6.54\text{M}$ [+4.80%] at 60 s). Deepening the global Transformer in `Standard-Matched` slightly reduces 30 s performance relative to `Standard` and `Hierarchical`, whereas all three variants perform comparably at 60 s.
- **Leyenda (Español):**
  > **Figura S5. Estudio de ablación con capacidad equiparada en Stress-Predict ($n = 33$ participantes) bajo el protocolo supervisado LOGO del Experimento 1 para ventanas de 30 s (izquierda) y 60 s (derecha).** Se comparan `Standard`, `Hierarchical` y `Standard-Matched` (+3.83% parámetros en 30 s; +4.80% en 60 s) en Accuracy, F1-score, Balanced Accuracy y AUROC (Media $\pm$ DE entre participantes).

---

### Figure S6 — Four-Panel Capacity-Matched Ablation Across WESAD and Stress-Predict (Audit Point 5)
- **Files:**
  - PNG: [`results/figures/ablation_matched_capacity_all_datasets.png`](../results/figures/ablation_matched_capacity_all_datasets.png)
  - PDF: [`results/figures/ablation_matched_capacity_all_datasets.pdf`](../results/figures/ablation_matched_capacity_all_datasets.pdf)
- **Reproduction Script:** `python ablation_matched_capacity/generate_deliverables.py`
- **Manuscript Reference:** Section 4.6 (Table 7 Extension)
- **Caption (English — Ready for Manuscript/Supplement):**
  > **Figure S6. Complete four-panel comparison of `Standard`, `Hierarchical`, and `Standard-Matched` encoders across WESAD ($n = 15$, top row) and Stress-Predict ($n = 33$, bottom row) for 30 s (left column) and 60 s (right column) windows under the Experiment 1 supervised LOGO protocol.** Exact trainable parameter counts are shown in each panel's legend (relative differences between `Standard-Matched` and `Hierarchical`: $-2.61\%$ on WESAD 60 s, $+3.83\%$ on WESAD 30 s and Stress-Predict 30 s, and $+4.80\%$ on Stress-Predict 60 s).
- **Leyenda (Español):**
  > **Figura S6. Comparación integral en cuatro paneles de los modelos `Standard`, `Hierarchical` y `Standard-Matched` en WESAD ($n = 15$, fila superior) y Stress-Predict ($n = 33$, fila inferior) para ventanas de 30 s y 60 s bajo el protocolo supervisado LOGO (Experimento 1).**

---

### Figure S7 — Participant-Level Paired Differences on Stress-Predict (Audit Point 6)
- **Files:**
  - PNG: [`results/figures/ablation_paired_differences_stress_predict.png`](../results/figures/ablation_paired_differences_stress_predict.png)
  - PDF: [`results/figures/ablation_paired_differences_stress_predict.pdf`](../results/figures/ablation_paired_differences_stress_predict.pdf)
- **Reproduction Script:** `python ablation_matched_capacity/statistical_analysis.py`
- **Manuscript Reference:** Section 4.6 (Paired Subject-Level Ablation Analysis)
- **Caption (English — Ready for Manuscript/Supplement):**
  > **Figure S7. Subject-by-subject paired performance differences ($\Delta_i$) across all $n = 33$ Stress-Predict participants ($S02 \dots S35$) in F1-score (top row) and Balanced Accuracy (bottom row) for 30 s (left) and 60 s (right) windows.** Blue bars show the capacity-controlled architectural contrast $\Delta_i = \text{Hierarchical}_i - \text{Standard-Matched}_i$, and orange bars show the total contrast $\Delta_i = \text{Hierarchical}_i - \text{Standard}_i$. Dashed horizontal lines mark the mean paired differences ($\bar{\Delta}$), and inset boxes report the 95% confidence intervals, rank-biserial effect sizes ($r_{\text{rb}}$), two-sided Wilcoxon signed-rank $p$-values, and participant Win/Tie/Loss counts.
- **Leyenda (Español):**
  > **Figura S7. Diferencias pareadas individuales por participante ($\Delta_i$) a lo largo de los $n = 33$ sujetos de Stress-Predict ($S02 \dots S35$) en F1-score (fila superior) y Balanced Accuracy (fila inferior) para ventanas de 30 s y 60 s.** Se contrastan $\Delta(\text{Hierarchical} - \text{Standard-Matched})$ (azul) y $\Delta(\text{Hierarchical} - \text{Standard})$ (naranja), incluyendo $\text{IC}_{95\%}$, correlación rango-biserial $r_{\text{rb}}$, $p$-valor de Wilcoxon y conteo de sujetos Gana/Empata/Pierde.

---

### Figure S8 — Paired Statistical Forest Plot and Effect-Size Distribution (Audit Point 6)
- **Files:**
  - PNG: [`results/figures/ablation_paired_differences_summary.png`](../results/figures/ablation_paired_differences_summary.png)
  - PDF: [`results/figures/ablation_paired_differences_summary.pdf`](../results/figures/ablation_paired_differences_summary.pdf)
- **Reproduction Script:** `python ablation_matched_capacity/statistical_analysis.py`
- **Manuscript Reference:** Section 4.6 (Statistical Significance and Effect Size Summary)
- **Caption (English — Ready for Manuscript/Supplement):**
  > **Figure S8. Forest and strip distribution plots of subject-level paired differences ($\Delta$) for `Hierarchical - Standard-Matched` (blue circles) and `Hierarchical - Standard` (brown squares) across Stress-Predict ($n = 33$, top row) and WESAD ($n = 15$, bottom row) at 30 s and 60 s.** Horizontal error bars represent two-sided 95% Student's $t$ confidence intervals around the mean paired difference $\bar{\Delta}$, overlaid on individual participant differences (semi-transparent dots). Annotations report $\bar{\Delta}$, rank-biserial correlation ($r_{\text{rb}}$), and raw two-sided Wilcoxon $p$-values. All 95% CIs span zero ($\Delta = 0$), confirming no statistically significant systematic advantage under purely supervised LOGO training from scratch.
- **Leyenda (Español):**
  > **Figura S8. Gráfico *forest* y distribución de diferencias pareadas por sujeto ($\bar{\Delta} \pm \text{IC}_{95\%}$) para `Hierarchical - Standard-Matched` y `Hierarchical - Standard` en Stress-Predict ($n = 33$) y WESAD ($n = 15$) a 30 s y 60 s.** Todos los intervalos de confianza al 95% incluyen el cero, confirmando la ausencia de diferencias estadísticamente significativas tras corrección por multiplicidad bajo entrenamiento supervisado desde cero.

---

## Part II. Supplementary Tables (`results/tables/`)

| Table ID | File Path | Description / Legend |
| :--- | :--- | :--- |
| **Table S1** | [`results/tables/audit_point1_auroc_table.csv`](../results/tables/audit_point1_auroc_table.csv) | **Point 1 AUROC Audit Table:** Compares positive-class evaluation ($p_{\text{stress}}$, $\text{AUROC}=0.4844$) vs. inverted probability ($1-p_{\text{stress}}$, $\text{AUROC}=0.5156$) and the Always-Stress baseline on Stress-Predict $\to$ WESAD (30 s, Standard). |
| **Table S2** | [`results/tables/audit_point2_per_class_metrics.csv`](../results/tables/audit_point2_per_class_metrics.csv) | **Point 2 Per-Class & Confusion Matrix Breakdown:** Full TN, FP, FN, TP, class-wise Precision, Recall/Sensitivity, Specificity, F1, and majority-collapse flags across 12 configurations in Experiments 1, 2, and 4. |
| **Table S3** | [`results/tables/audit_point2_always_stress_comparison.csv`](../results/tables/audit_point2_always_stress_comparison.csv) | **Point 2 Always-Stress Baseline Comparison:** Quantifies $\Delta\text{Balanced Accuracy}$ and $\Delta\text{F1}$ relative to a trivial majority-class predictor across all 12 audited configurations. |
| **Table S4** | [`results/tables/audit_point3_convergence_summary.csv`](../results/tables/audit_point3_convergence_summary.csv) | **Point 3 Convergence Summary:** Best validation F1 epoch, maximum validation F1, minimum validation loss, final balanced accuracy, and numerical stability verification for Standard and Hierarchical on Stress-Predict (30 s and 60 s). |
| **Table S5** | [`results/tables/audit_point4_fairness_hyperparameters_budget.csv`](../results/tables/audit_point4_fairness_hyperparameters_budget.csv) | **Point 4 Fairness & Hyperparameter Budget Table:** Documents 28/7 subject splits, zero test leakage, 5 Optuna trials per variant, shared search space, and exact trainable parameter deltas. |
| **Table S6** | [`results/tables/ablation_matched_capacity/table7_extended_ablation_summary.csv`](../results/tables/ablation_matched_capacity/table7_extended_ablation_summary.csv) | **Point 5 Extended Table 7 (WESAD & Stress-Predict):** Aggregate LOGO metrics (Mean $\pm$ SD) and exact parameter counts/relative differences for `Standard`, `Hierarchical`, and `Standard-Matched` across both datasets (30 s and 60 s). |
| **Table S7** | [`results/tables/ablation_matched_capacity/ablation_matched_capacity_statistical_analysis.csv`](../results/tables/ablation_matched_capacity/ablation_matched_capacity_statistical_analysis.csv) | **Point 6 Paired Statistical Comparison Table:** Mean paired differences, Student's $t$ and 10,000-resample bootstrap 95% CIs, medians/IQRs, Win/Tie/Loss counts, Cohen's $d_z$, rank-biserial $r_{\text{rb}}$, Wilcoxon $W$, raw $p$-values, and Bonferroni/Holm-Bonferroni adjusted $p$-values. |
| **Table S8** | [`results/tables/experimental_evidence_traceability_matrix.csv`](../results/tables/experimental_evidence_traceability_matrix.csv) | **Experimental Evidence Traceability Matrix:** Maps every diagnostic audit and ablation experiment (Points 1–7) to its reproduction scripts, CSV tables, supplementary figures, and scientific conclusions. |
