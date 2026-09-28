# Dataset Splits and Participant Partition Definitions

This directory contains the authoritative, machine-readable participant split definitions used across all experiments in **TimePatch-SSL**.

Strict **subject-level independence** is enforced in all protocols to guarantee **zero data leakage** between training, validation, and testing partitions.

---

## 1. Summary of Experimental Split Protocols

| Experiment | Dataset(s) | Evaluation Protocol | Partition Strategy | Participant Split File |
| :--- | :--- | :--- | :--- | :--- |
| **Exp 1: Supervised** | WESAD / Stress-Predict | Within-dataset | Leave-One-Subject-Out (LOSO) | Evaluated across all $N$ subjects iteratively |
| **Exp 2: Cross-Dataset** | WESAD $\leftrightarrow$ Stress-Predict | Cross-dataset transfer | Source 85% dev / 15% val; Target 100% test | [`wesad_subject_splits.json`](./wesad_subject_splits.json), [`stress_predict_subject_splits.json`](./stress_predict_subject_splits.json) |
| **Exp 3: SSL Evaluation** | WESAD / Stress-Predict | Within-dataset SSL | Pretrain + Fine-tune under LOSO | Evaluated across all $N$ subjects iteratively |
| **Exp 4: Combined** | WESAD + Stress-Predict | Multi-dataset pooling | 80% development ($n=40$) / 20% test ($n=10$) | [`experiment4_split.json`](./experiment4_split.json) |

---

## 2. Participant Identifiers

### WESAD ($N = 15$ Participants)
- **All Subject IDs:** `S2`, `S3`, `S4`, `S5`, `S6`, `S7`, `S8`, `S9`, `S10`, `S11`, `S13`, `S14`, `S15`, `S16`, `S17`
- *Note:* Subject `S1` is not included in the standard WESAD release due to sensor malfunction in the original collection. Subject `S12` was omitted by the dataset authors.
- **Fixed Holdout Split (Exp 2 & 4):**
  - **Development ($n=12$, 80%):** `S3`, `S4`, `S5`, `S6`, `S7`, `S8`, `S9`, `S10`, `S13`, `S15`, `S16`, `S17`
  - **Test ($n=3$, 20%):** `S2`, `S11`, `S14`

### Stress-Predict ($N = 35$ Participants)
- **All Subject IDs:** `S1` through `S35`
- **Fixed Holdout Split (Exp 2 & 4):**
  - **Development ($n=28$, 80%):** `S1`, `S2`, `S3`, `S4`, `S5`, `S6`, `S7`, `S8`, `S9`, `S10`, `S11`, `S12`, `S13`, `S15`, `S17`, `S18`, `S19`, `S21`, `S23`, `S24`, `S26`, `S28`, `S29`, `S31`, `S32`, `S33`, `S34`, `S35`
  - **Test ($n=7$, 20%):** `S14`, `S16`, `S20`, `S22`, `S25`, `S27`, `S30`

---

## 3. Split Files Specification

### A. [`wesad_subject_splits.json`](./wesad_subject_splits.json)
- Defines the development and test partitions for the WESAD dataset.
- Seed: `42`, Shuffle: `True`.
- Guarantees `participant_overlap: []`.

### B. [`stress_predict_subject_splits.json`](./stress_predict_subject_splits.json)
- Defines the development and test partitions for the Stress-Predict dataset.
- Seed: `42`, Shuffle: `True`.
- Guarantees `participant_overlap: []`.

### C. [`experiment4_split.json`](./experiment4_split.json)
- Defines the unified subject-level split for the combined dataset benchmark (Exp 4).
- Development set: 40 participants (12 WESAD + 28 Stress-Predict).
- Test set: 10 participants (3 WESAD + 7 Stress-Predict).
- All signal windows strictly follow the subject assignment. Held-out test participants are completely excluded from SSL pretraining, representation learning, and hyperparameter selection.

---

## 4. Verification of Zero Data Leakage

To verify split integrity programmatically:
```python
import json

with open("splits/experiment4_split.json") as f:
    split = json.load(f)

for ds in ["WESAD", "Stress-Predict"]:
    dev = set(split["datasets"][ds]["development_participant_ids"])
    test = set(split["datasets"][ds]["test_participant_ids"])
    overlap = dev.intersection(test)
    assert len(overlap) == 0, f"Leakage detected in {ds}: {overlap}"
print("Zero participant leakage verified across all splits.")
```
