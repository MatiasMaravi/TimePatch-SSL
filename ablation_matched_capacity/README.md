# Matched-Capacity Ablation Study (Reviewer Point 5 & Reviewer 3 Follow-Up)

Este módulo implementa el estudio de ablación con capacidad equiparada (**Standard-Matched**) tanto en **WESAD** como en **Stress-Predict** bajo el protocolo supervisado Leave-One-Group-Out (LOGO) del **Experiment 1**:

> *"1) The capacity-matched ablation (§4.6, Table 7) is reported for WESAD only... Please run the Standard-Matched control on Stress-Predict, at least under the Experiment 1 protocol. Until then, please restrict the statement that the ablation 'reduces parameter count as a sufficient explanation for the observed differences'."*

Todo el código de esta ablación está aislado en este directorio (`ablation_matched_capacity/`) para **preservar el 100% de los archivos originales** de `src/`, `analysis/` y `experiments/`.

---

## 1. Justificación Científica

- **Standard:** Patch embedding 1D lineal + Transformer global de profundidad base ($L$).
- **Hierarchical:** Patch embedding con Transformer intra-parche (TimeMAE) + Transformer global ($L$).
- **Standard-Matched:** Conserva el patch embedding 1D lineal pero compensa la capacidad aumentando la profundidad ($L' > L$) del Transformer global hasta aproximar el número de parámetros entrenables del modelo Jerárquico (**-2.61%** en WESAD 60 s, **+3.83%** en WESAD 30 s y Stress-Predict 30 s, y **+4.80%** en Stress-Predict 60 s).

---

## 2. Archivos del Módulo

| Archivo | Descripción |
| :--- | :--- |
| [`models.py`](./models.py) | Define `StandardMatchedTransformer`, `calculate_matched_depth()` y la fábrica `build_model()`. |
| [`verify_capacity.py`](./verify_capacity.py) | Audita y genera la tabla formal de paridad de parámetros entre los 3 modelos. |
| [`run_ablation.py`](./run_ablation.py) | Ejecuta el protocolo Experiment 1 (LOGO) evaluando simultáneamente los 3 modelos en `wesad` o `stress_predict` (30 s y 60 s). |
| [`generate_deliverables.py`](./generate_deliverables.py) | Genera la **Tabla 7 Extendida**, las comparaciones estadísticas pareadas (Wilcoxon) y las figuras comparativas en `.png` y `.pdf`. |

---

## 3. Auditoría de Parámetros (Valores Reales Verificados)

La ejecución de `verify_capacity.py` reporta las diferencias exactas de parámetros entrenables entre `Hierarchical` y `Standard-Matched`:

```text
==============================================================================================================
CAPACITY-MATCHED ABLATION AUDIT (Reviewer Point 5)
==============================================================================================================
                 configuration  emb_dim  base_depth  matched_depth  patch_len  standard_trainable  hierarchical_trainable  standard_matched_trainable  diff_matched_vs_hierarchical  rel_diff_pct  parity_ratio
                  Base Default      128           4              9          8              887234                 1923782                     1878594                        -45188         -2.35        0.9765
                     WESAD 30s      256           6             10          8             4959554                 7819078                     8118594                        299516          3.83        1.0383
                     WESAD 60s      128           3              8         16              692034                 1728582                     1683394                        -45188         -2.61        0.9739
Stress-Predict / Stress-ID 30s      256           6             10          8             4959554                 7819078                     8118594                        299516          3.83        1.0383
Stress-Predict / Stress-ID 60s      256           4              8          8             3380034                 6239558                     6539074                        299516          4.80        1.0480
==============================================================================================================
```

---

## 4. Instrucciones de Uso

### A. Verificar Conteos de Parámetros
```bash
python ablation_matched_capacity/verify_capacity.py
```

### B. Ejecutar Ablación en Stress-Predict (30 s y 60 s)
```bash
python ablation_matched_capacity/run_ablation.py --dataset stress_predict --window-seconds 60 --device cuda
python ablation_matched_capacity/run_ablation.py --dataset stress_predict --window-seconds 30 --device cuda
```

### C. Ejecutar Ablación en WESAD (30 s y 60 s)
```bash
python ablation_matched_capacity/run_ablation.py --dataset wesad --window-seconds 60 --device cuda
python ablation_matched_capacity/run_ablation.py --dataset wesad --window-seconds 30 --device cuda
```

### D. Generar Tabla 7 Extendida, Pruebas Pareadas y Gráficos Comparativos
```bash
python ablation_matched_capacity/generate_deliverables.py
```
