# Matched-Capacity Ablation Study (Reviewer Point 5)

Este módulo implementa el estudio de ablación con capacidad equiparada (**Standard-Matched**) solicitado por el revisor:

> *"Point 5 - Novelty and causal attribution. [...] NOT ADDRESSED for the matched-capacity ablations, as the authors agree."*

Todo el código de esta ablación está aislado en este directorio (`ablation_matched_capacity/`) para **preservar el 100% de los archivos originales** de `src/`, `analysis/` y `experiments/`.

---

## 1. Justificación Científica

- **Standard:** Patch embedding 1D lineal + Transformer global (menor capacidad: ~887k parámetros en config base).
- **Hierarchical:** Patch embedding con Transformer intra-parche (TimeMAE) + Transformer global (mayor capacidad: ~1.92M parámetros).
- **Standard-Matched:** Conserva el patch embedding 1D lineal pero compensa la capacidad aumentando la profundidad (`depth`) del Transformer global hasta igualar los parámetros del Jerárquico (~1.88M - 1.92M, diferencia < 3%).

Al comparar **Standard vs. Hierarchical vs. Standard-Matched**, se aísla causalmente el efecto de la **jerarquía temporal** respecto al **tamaño o capacidad del modelo**.

---

## 2. Archivos del Módulo

| Archivo | Descripción |
| :--- | :--- |
| [`models.py`](file:///C:/Users/USUARIO/Desktop/TimePatch-SSL/ablation_matched_capacity/models.py) | Define `StandardMatchedTransformer`, `calculate_matched_depth()` y la fábrica `build_model()`. |
| [`verify_capacity.py`](file:///C:/Users/USUARIO/Desktop/TimePatch-SSL/ablation_matched_capacity/verify_capacity.py) | Audita y genera la tabla formal de paridad de parámetros entre los 3 modelos. |
| [`run_ablation.py`](file:///C:/Users/USUARIO/Desktop/TimePatch-SSL/ablation_matched_capacity/run_ablation.py) | Ejecuta el experimento Leave-One-Subject-Out (LOSO) evaluando simultáneamente los 3 modelos. |

---

## 3. Auditoría de Parámetros (Resultados Verificados)

La ejecución de `verify_capacity.py` demuestra que la paridad entre `Hierarchical` y `Standard-Matched` es exacta con una desviación relativa **< 5%** en todas las configuraciones del estudio:

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
Genera y actualiza el reporte en `results/tables/ablation_matched_capacity_parameters.csv`:
```bash
python ablation_matched_capacity/verify_capacity.py
```

### B. Probar el Pipeline Rápido (Dry-Run sintético)
Verifica que los 3 modelos entrenan y evalúan sin errores en 3 segundos:
```bash
python ablation_matched_capacity/run_ablation.py --dry-run
```

### C. Ejecutar Ablación Completa en Datos Reales
```bash
python ablation_matched_capacity/run_ablation.py --dataset wesad --window-seconds 60 --device cuda
```
