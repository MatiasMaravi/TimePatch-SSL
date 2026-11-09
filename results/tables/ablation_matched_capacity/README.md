# Matched-Capacity Ablation Results (WESAD & Stress-Predict)

This folder contains the reproducible tables, participant-level evaluation metrics, and statistical comparisons generated for the **Matched-Capacity Ablation Study (`Standard-Matched`)** across both **WESAD** and **Stress-Predict** under the **Experiment 1** supervised Leave-One-Group-Out (LOGO) protocol.

To isolate the effect of the **hierarchical temporal patch representation** from model parameter count across both datasets (**WESAD** and **Stress-Predict**) and both window lengths (**30 s** and **60 s**), we evaluate three encoder variants under the exact **Experiment 1** supervised Leave-One-Group-Out (LOGO) protocol.

---

## 1. Architecture Variants Compared

| Variant | Patch Embedding | Global Transformer Depth | Capacity Parity Strategy |
| :--- | :--- | :---: | :--- |
| **Standard** | 1D Linear projection | Base depth ($L$) | Baseline capacity (0.69M–4.96M params) |
| **Hierarchical** | Intra-patch Transformer (TimeMAE) | Base depth ($L$) | Multi-scale representation (1.73M–7.82M params) |
| **Standard-Matched** | 1D Linear projection | Scaled depth ($L' > L$) | **Capacity matched** to Hierarchical (-2.61% to +4.80% diff) |

---

## 2. Contents of this Directory

| File | Description | Participants |
| :--- | :--- | :---: |
| [`table7_extended_ablation_summary.csv`](./table7_extended_ablation_summary.csv) | **Extended Table 7** combining WESAD & Stress-Predict (30s and 60s) with exact parameter counts and relative differences. | $n=15$ / $n=33$ |
| [`ablation_matched_capacity_parameters.csv`](./ablation_matched_capacity_parameters.csv) | Full parameter parity audit across all 5 experimental configurations (-2.61% to +4.80% relative diff). | N/A |
| [`ablation_matched_capacity_statistical_analysis.csv`](./ablation_matched_capacity_statistical_analysis.csv) | **Phase 6 full paired statistical analysis** (Mean $\Delta$, 95% CI $t$ & Bootstrap, Median/IQR, Wins/Ties/Losses, Cohen's $d_z$, Rank-Biserial $r_{\text{rb}}$, Wilcoxon $W$, $p_{\text{raw}}$, Bonferroni & Holm-Bonferroni adjusted $p$-values). | $n=15$ / $n=33$ |
| [`ablation_matched_capacity_paired_comparisons.csv`](./ablation_matched_capacity_paired_comparisons.csv) | Paired subject-level Wilcoxon signed-rank tests (`Hierarchical` vs. `Standard-Matched` vs. `Standard`). | $n=15$ / $n=33$ |
| [`ablation_matched_capacity_results_stress_predict_60s.csv`](./ablation_matched_capacity_results_stress_predict_60s.csv) | Aggregate metrics (Mean, Std, 95% CI) on **Stress-Predict (60s windows, 30s stride)**. | $n=33$ |
| [`ablation_matched_capacity_participants_stress_predict_60s.csv`](./ablation_matched_capacity_participants_stress_predict_60s.csv) | Participant-by-participant metrics ($S02 \dots S35$, excl. $S01, S06$) on **Stress-Predict 60s**. | $n=33$ |
| [`ablation_matched_capacity_results_stress_predict_30s.csv`](./ablation_matched_capacity_results_stress_predict_30s.csv) | Aggregate metrics (Mean, Std, 95% CI) on **Stress-Predict (30s windows, 15s stride)**. | $n=33$ |
| [`ablation_matched_capacity_participants_stress_predict_30s.csv`](./ablation_matched_capacity_participants_stress_predict_30s.csv) | Participant-by-participant metrics ($S02 \dots S35$, excl. $S01, S06$) on **Stress-Predict 30s**. | $n=33$ |
| [`ablation_matched_capacity_results_wesad_60s.csv`](./ablation_matched_capacity_results_wesad_60s.csv) | Aggregate metrics (Mean, Std, 95% CI) on **WESAD (60s windows, 30s stride)**. | $n=15$ |
| [`ablation_matched_capacity_participants_wesad_60s.csv`](./ablation_matched_capacity_participants_wesad_60s.csv) | Participant-by-participant metrics ($S2 \dots S17$) on **WESAD 60s**. | $n=15$ |
| [`ablation_matched_capacity_results_wesad_30s.csv`](./ablation_matched_capacity_results_wesad_30s.csv) | Aggregate metrics (Mean, Std, 95% CI) on **WESAD (30s windows, 15s stride)**. | $n=15$ |
| [`ablation_matched_capacity_participants_wesad_30s.csv`](./ablation_matched_capacity_participants_wesad_30s.csv) | Participant-by-participant metrics ($S2 \dots S17$) on **WESAD 30s**. | $n=15$ |

---

## 3. Summary of Results

### 3.1 Parameter Parity Audit

| Configuration | Emb. Dim | Patch Len | Base Depth ($L$) | Matched Depth ($L'$) | Standard (Params) | Hierarchical (Params) | Standard-Matched (Params) | Diff vs. Hierarchical | Rel. Diff (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Base Default** | 128 | 8 | 4 | 9 | 887,234 | 1,923,782 | 1,878,594 | -45,188 | -2.35% |
| **WESAD 30s** | 256 | 8 | 6 | 10 | 4,959,554 | 7,819,078 | 8,118,594 | +299,516 | **+3.83%** |
| **WESAD 60s** | 128 | 16 | 3 | 8 | 692,034 | 1,728,582 | 1,683,394 | -45,188 | **-2.61%** |
| **Stress-Predict 30s** | 256 | 8 | 6 | 10 | 4,959,554 | 7,819,078 | 8,118,594 | +299,516 | **+3.83%** |
| **Stress-Predict 60s** | 256 | 8 | 4 | 8 | 3,380,034 | 6,239,558 | 6,539,074 | +299,516 | **+4.80%** |

*Note on discrete layer matching:* Because capacity in `Standard-Matched` is adjusted by adding integer numbers of global Transformer layers ($748,790$ parameters per layer at $\text{emb\_dim}=256$), the closest discrete depth match yields a relative parameter difference of **+3.83%** for 30 s ($L=6 \to 10$) and **+4.80%** for 60 s ($L=4 \to 8$) on Stress-Predict, and **-2.61%** (60 s) / **+3.83%** (30 s) on WESAD.

---

### 3.2 Extended Table 7: Supervised LOGO Ablation Across WESAD and Stress-Predict (Experiment 1 Protocol)

| Dataset | Window | Model Variant | Depth | Trainable Params | Rel. Diff vs. Hier. | Accuracy (Mean ± SD) | F1-Score (Mean ± SD) | Balanced Acc (Mean ± SD) | AUROC (Mean ± SD) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **WESAD** ($n=15$) | 30 s | Standard | 6 | 4,959,554 | -36.57% | 0.8661 ± 0.1272 | 0.7604 ± 0.2639 | 0.8462 ± 0.1718 | 0.9004 ± 0.1765 |
| **WESAD** ($n=15$) | 30 s | Hierarchical | 6 | 7,819,078 | 0.00% | 0.8630 ± 0.2203 | 0.7721 ± 0.3572 | 0.8417 ± 0.2582 | 0.8582 ± 0.2882 |
| **WESAD** ($n=15$) | 30 s | Standard-Matched | 10 | 8,118,594 | **+3.83%** | 0.8678 ± 0.1779 | 0.7753 ± 0.2993 | 0.8545 ± 0.1999 | 0.8798 ± 0.2181 |
| **WESAD** ($n=15$) | 60 s | Standard | 3 | 692,034 | -59.97% | 0.8752 ± 0.1587 | 0.7746 ± 0.3113 | 0.8557 ± 0.2026 | 0.9162 ± 0.1616 |
| **WESAD** ($n=15$) | 60 s | Hierarchical | 3 | 1,728,582 | 0.00% | 0.8602 ± 0.1722 | 0.7653 ± 0.3159 | 0.8466 ± 0.2087 | 0.9098 ± 0.1656 |
| **WESAD** ($n=15$) | 60 s | Standard-Matched | 8 | 1,683,394 | **-2.61%** | 0.8614 ± 0.1820 | 0.7430 ± 0.3539 | 0.8342 ± 0.2351 | 0.8688 ± 0.2457 |
| **Stress-Predict** ($n=33$) | 30 s | Standard | 6 | 4,959,554 | -36.57% | 0.8351 ± 0.1508 | 0.8609 ± 0.1302 | 0.8307 ± 0.1594 | 0.9030 ± 0.1478 |
| **Stress-Predict** ($n=33$) | 30 s | Hierarchical | 6 | 7,819,078 | 0.00% | 0.8222 ± 0.1581 | 0.8517 ± 0.1354 | 0.8163 ± 0.1693 | 0.8947 ± 0.1561 |
| **Stress-Predict** ($n=33$) | 30 s | Standard-Matched | 10 | 8,118,594 | **+3.83%** | 0.8178 ± 0.1639 | 0.8447 ± 0.1425 | 0.8161 ± 0.1742 | 0.8874 ± 0.1606 |
| **Stress-Predict** ($n=33$) | 60 s | Standard | 4 | 3,380,034 | -45.83% | 0.8254 ± 0.1644 | 0.8495 ± 0.1492 | 0.8231 ± 0.1730 | 0.9020 ± 0.1537 |
| **Stress-Predict** ($n=33$) | 60 s | Hierarchical | 4 | 6,239,558 | 0.00% | 0.8282 ± 0.1724 | 0.8550 ± 0.1470 | 0.8226 ± 0.1841 | 0.8996 ± 0.1548 |
| **Stress-Predict** ($n=33$) | 60 s | Standard-Matched | 8 | 6,539,074 | **+4.80%** | 0.8332 ± 0.1545 | 0.8579 ± 0.1347 | 0.8314 ± 0.1644 | 0.8878 ± 0.1754 |

---

### 3.3 Paired Subject-Level Comparisons on Stress-Predict (`Hierarchical` vs. `Standard-Matched`)

| Window | Metric | Hierarchical Mean | Standard-Matched Mean | Paired Diff (Hier − Matched) | Paired SD | Wilcoxon Stat | $p$-value |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **30 s** ($n=33$) | Accuracy | 0.8222 | 0.8178 | **+0.0044** | 0.0411 | 139.0 | 0.5272 |
| **30 s** ($n=33$) | F1-Score | 0.8517 | 0.8447 | **+0.0069** | 0.0329 | 134.0 | 0.1864 |
| **30 s** ($n=33$) | Balanced Accuracy | 0.8163 | 0.8161 | **+0.0002** | 0.0480 | 184.0 | 0.9044 |
| **30 s** ($n=33$) | AUROC | 0.8947 | 0.8874 | **+0.0073** | 0.0438 | 173.0 | 0.2210 |
| **60 s** ($n=33$) | Accuracy | 0.8282 | 0.8332 | **-0.0050** | 0.0523 | 146.5 | 0.3072 |
| **60 s** ($n=33$) | F1-Score | 0.8550 | 0.8579 | **-0.0029** | 0.0428 | 155.5 | 0.4209 |
| **60 s** ($n=33$) | Balanced Accuracy | 0.8226 | 0.8314 | **-0.0088** | 0.0549 | 148.0 | 0.3246 |
| **60 s** ($n=33$) | AUROC | 0.8996 | 0.8878 | **+0.0118** | 0.0495 | 161.0 | 0.9678 |

---

## 4. Key Conclusions on Capacity vs. Architecture

1. **Exact Parameter Parity Disclosure:**
   Adding global Transformer layers to the linear-patch `Standard` encoder (`Standard-Matched`) achieves a relative parameter difference wrt `Hierarchical` of **+3.83%** on Stress-Predict 30 s ($8.12\text{M}$ vs. $7.82\text{M}$) and **+4.80%** on Stress-Predict 60 s ($6.54\text{M}$ vs. $6.24\text{M}$), and **-2.61%** (60 s) / **+3.83%** (30 s) on WESAD. We report these exact figures without claiming `< 4%` equivalence across all configurations.
2. **Effect of Model Capacity under Supervised LOGO Training (Experiment 1):**
   - In **Stress-Predict 30 s**, increasing global depth from $L=6$ (`Standard`, $4.96\text{M}$) to $L'=10$ (`Standard-Matched`, $8.12\text{M}$) slightly **degrades** supervised performance across all metrics (Accuracy: $0.8351 \to 0.8178$; F1: $0.8609 \to 0.8447$, paired Wilcoxon $p = 0.0201$; AUROC: $0.9030 \to 0.8874$), indicating mild overfitting when stacking deeper global layers on small subject-level sample sizes without self-supervised pretraining. Despite having $3.83\%$ fewer parameters than `Standard-Matched`, `Hierarchical` achieves slightly higher Accuracy ($0.8222$), F1 ($0.8517$), and AUROC ($0.8947$), though paired differences between `Hierarchical` and `Standard-Matched` are small and not statistically significant ($p \ge 0.186$).
   - In **Stress-Predict 60 s**, all three variants perform within a narrow margin ($0.8254$–$0.8332$ Accuracy; $0.8495$–$0.8579$ F1). Increasing parameter count in `Standard-Matched` ($6.54\text{M}$) yields a slight gain in thresholded metrics (+0.0050 Accuracy, +0.0029 F1 over `Hierarchical`, $p \ge 0.307$) while `Hierarchical` retains a higher AUROC ($0.8996$ vs. $0.8878$).
3. **Nuanced Interpretation for Manuscript §4.6:**
   Under purely supervised within-dataset training (Experiment 1), neither the intra-patch hierarchical module nor brute-force global depth scaling produces a statistically significant gain on WESAD or Stress-Predict; in fact, on 30 s windows, larger capacity slightly penalizes generalization from scratch. Consequently, we restrict the claim in §4.6: parameter count alone does not systematically improve supervised LOGO performance (and degrades it at 30 s), while the advantages of the `Hierarchical` architecture reported in SSL and transfer regimes (Tables 4 and 6) should be interpreted in conjunction with self-supervised pretraining rather than claimed from supervised Experiment 1 alone.

---

## 5. Reproduction Commands

```bash
# 1. Audit parameter parity
python ablation_matched_capacity/verify_capacity.py

# 2. Run Stress-Predict 60s and 30s LOSO ablations on GPU
python ablation_matched_capacity/run_ablation.py --dataset stress_predict --window-seconds 60 --device cuda
python ablation_matched_capacity/run_ablation.py --dataset stress_predict --window-seconds 30 --device cuda

# 3. Run WESAD 60s and 30s LOSO ablations on GPU
python ablation_matched_capacity/run_ablation.py --dataset wesad --window-seconds 60 --device cuda
python ablation_matched_capacity/run_ablation.py --dataset wesad --window-seconds 30 --device cuda

# 4. Generate Extended Table 7, paired Wilcoxon comparisons, and comparative figures
python ablation_matched_capacity/generate_deliverables.py
```
