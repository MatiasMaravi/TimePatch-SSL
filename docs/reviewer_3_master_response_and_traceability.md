# Informe Maestro de Auditoría Final, Trazabilidad y Entrega del Equipo (Puntos 1 al 7 — Reviewer 3)

**Proyecto:** TimePatch-SSL — Contrastive Self-Supervised Learning with Hierarchical Time-Patch Transformers for Multimodal Stress Detection  
**Repositorio:** `https://github.com/MatiasMaravi/TimePatch-SSL`  
**Documento de Referencia:** *Sensors_ TimePatch v3.pdf* (`TimePatch-SSL — Experimentos pendientes para Reviewer 3`)

---

## 1. Tabla de Trazabilidad: Comentarios del Revisor 3 vs. Evidencia Generada

La siguiente tabla (exportada también en formato CSV en [`results/tables/reviewer3_evidence_traceability_matrix.csv`](../results/tables/reviewer3_evidence_traceability_matrix.csv)) relaciona cada uno de los **7 puntos del plan de respuesta al Reviewer 3** con sus scripts ejecutables, tablas CSV, figuras suplementarias y conclusión científica defendible:

| Punto | Solicitud del Revisor / Actividad | Scripts de Reproducción | Tablas CSV Generadas | Figuras Generadas | Conclusión Científica Defendible |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Punto 1** | **Auditoría del AUROC anómalo (0.1049)** en Exp 2 (`Stress-Predict -> WESAD`, 30 s, Standard). Verificar clase positiva (`y=1`), $p_{\text{stress}} = \text{softmax}[:,1]$ e inversión $(1-p)$. | [`experiments_audit/audit_point1.py`](../experiments_audit/audit_point1.py) | [`audit_point1_auroc_table.csv`](../results/tables/audit_point1_auroc_table.csv)<br>[`audit_point1_stresspredict_to_wesad_30s_standard.csv`](../results/predictions/audit_point1_stresspredict_to_wesad_30s_standard.csv) | [`audit_point1_roc_curve.png`](../results/figures/audit_point1_roc_curve.png) (Fig. S1) | Evaluando estrictamente $p_{\text{stress}} = \text{softmax}(\text{logits})[:,1]$, el modelo predice estrés en el 100% de las ventanas ($\text{BAcc}=0.5000$) con $\text{AUROC}(p)=0.4844$ y $\text{AUROC}(1-p)=0.5156$. El valor histórico `0.1049` provino del complemento $1 - 0.8951$ al indexar la columna opuesta bajo inversión por *domain shift*. |
| **Punto 2** | **Matrices de confusión, métricas por clase y colapso a la clase mayoritaria** en Exp 2, Exp 1 y Exp 4 (30 s y 60 s, Standard vs. Hierarchical) frente a *Always-Stress*. | [`experiments_audit/audit_point2.py`](../experiments_audit/audit_point2.py) | [`audit_point2_per_class_metrics.csv`](../results/tables/audit_point2_per_class_metrics.csv)<br>[`audit_point2_always_stress_comparison.csv`](../results/tables/audit_point2_always_stress_comparison.csv) | [`audit_point2_confusion_matrices.png`](../results/figures/audit_point2_confusion_matrices.png) (Fig. S2) | En particiones *holdout* a 30 s, `Standard` colapsa al 100% de predicciones de estrés ($\text{TN}=0, \text{Specificity}=0.0000, \text{BAcc}=0.5000$, idéntico a *Always-Stress*), mientras que `Hierarchical` evita el colapso ($\text{BAcc}=0.8269, \text{Spec}=0.8108, \text{AUROC}=0.9030$ en Stress-Predict 30 s). En 60 s ningún modelo colapsa. |
| **Punto 3** | **Verificación de convergencia del Standard encoder** época por época (Train/Val Loss, Val F1, Val BAcc, estabilidad sin NaN) en Stress-Predict (30 s y 60 s). | [`experiments_audit/audit_point3.py`](../experiments_audit/audit_point3.py) | [`audit_point3_convergence_summary.csv`](../results/tables/audit_point3_convergence_summary.csv) | [`audit_point3_loss_curves.png`](../results/figures/audit_point3_loss_curves.png) (Fig. S3)<br>[`audit_point3_val_metrics_curves.png`](../results/figures/audit_point3_val_metrics_curves.png) (Fig. S4) | Ambos modelos convergen de manera monótona y estable sin NaNs ni explosión de gradientes. `Standard` alcanza $\text{Val F1}=0.8865$ (30 s) y $0.8986$ (60 s), demostrando que su colapso en test a 30 s no se debe a fallas de optimización numérica. |
| **Punto 4** | **Auditoría de equidad (fairness) y presupuesto de hiperparámetros** (particiones sin fuga, presupuesto de Optuna, hiperparámetros y conteo de parámetros). | [`experiments_audit/audit_point4.py`](../experiments_audit/audit_point4.py) | [`audit_point4_fairness_hyperparameters_budget.csv`](../results/tables/audit_point4_fairness_hyperparameters_budget.csv)<br>[`splits/*.json`](../splits/) | N/A (Evidencia tabular y JSON) | Se verifica 0% de solapamiento entre desarrollo (28 sujetos) y prueba (7 sujetos), 5 *trials* de Optuna por configuración bajo Macro F1, optimizador AdamW ($lr=2\times 10^{-4}$), y diferencia exacta de $+2,859,524$ parámetros en el codificador intra-parche. |
| **Punto 5** | **Capacity-Matched Ablation en Stress-Predict** (30 s y 60 s; `Standard`, `Hierarchical`, `Standard-Matched`) bajo el protocolo LOGO del Experimento 1 y extensión de la Tabla 7. | [`ablation_matched_capacity/run_ablation.py`](../ablation_matched_capacity/run_ablation.py)<br>[`ablation_matched_capacity/verify_capacity.py`](../ablation_matched_capacity/verify_capacity.py)<br>[`ablation_matched_capacity/generate_deliverables.py`](../ablation_matched_capacity/generate_deliverables.py) | [`table7_extended_ablation_summary.csv`](../results/tables/ablation_matched_capacity/table7_extended_ablation_summary.csv)<br>[`ablation_matched_capacity_parameters.csv`](../results/tables/ablation_matched_capacity/ablation_matched_capacity_parameters.csv)<br>[`ablation_matched_capacity_results_stress_predict_{30,60}s.csv`](../results/tables/ablation_matched_capacity/)<br>[`ablation_matched_capacity_participants_stress_predict_{30,60}s.csv`](../results/tables/ablation_matched_capacity/) | [`ablation_matched_capacity_stress_predict.png`](../results/figures/ablation_matched_capacity_stress_predict.png) (Fig. S5)<br>[`ablation_matched_capacity_all_datasets.png`](../results/figures/ablation_matched_capacity_all_datasets.png) (Fig. S6) | Se reportan las diferencias reales de parámetros (**+3.83%** en 30 s y **+4.80%** en 60 s en Stress-Predict; **-2.61%** en 60 s y **+3.83%** en 30 s en WESAD). En Stress-Predict LOGO ($n=33$), aumentar la profundidad global (`Standard-Matched`) reduce ligeramente el F1 a 30 s ($0.8609 \to 0.8447$) frente a `Hierarchical` ($0.8517$), mientras que a 60 s los tres modelos convergen ($0.8495$–$0.8579$). |
| **Punto 6** | **Análisis estadístico pareado de la ablación** por sujeto ($\Delta$, $\text{IC}_{95\%}$ $t$ y Bootstrap, $r_{\text{rb}}$, $d_z$, Wilcoxon y corrección de Bonferroni/Holm). | [`ablation_matched_capacity/statistical_analysis.py`](../ablation_matched_capacity/statistical_analysis.py) | [`ablation_matched_capacity_statistical_analysis.csv`](../results/tables/ablation_matched_capacity/ablation_matched_capacity_statistical_analysis.csv)<br>[`ablation_matched_capacity_paired_comparisons.csv`](../results/tables/ablation_matched_capacity/ablation_matched_capacity_paired_comparisons.csv) | [`ablation_paired_differences_stress_predict.png`](../results/figures/ablation_paired_differences_stress_predict.png) (Fig. S7)<br>[`ablation_paired_differences_summary.png`](../results/figures/ablation_paired_differences_summary.png) (Fig. S8) | Todos los $\text{IC}_{95\%}$ de $\Delta(\text{Hierarchical} - \text{Standard-Matched})$ cruzan el cero ($p_{\text{raw}} \ge 0.1864$ en Stress-Predict; $p_{\text{Holm},8} = 1.0000$). Se restringe el alcance de la Sección 4.6 sin reclamar ventajas sistemáticas bajo entrenamiento puramente supervisado desde cero. |
| **Punto 7** | **Auditoría final de consistencia (Tablas 2, 3, 4, 6 y 7), verificación de clase positiva y reproducibilidad.** | [`experiments_audit/audit_point7_master_consistency.py`](../experiments_audit/audit_point7_master_consistency.py) | [`audit_point7_tables_consistency_summary.csv`](../results/tables/audit_point7_tables_consistency_summary.csv)<br>[`audit_point7_positive_class_verification.csv`](../results/tables/audit_point7_positive_class_verification.csv)<br>[`reviewer3_evidence_traceability_matrix.csv`](../results/tables/reviewer3_evidence_traceability_matrix.csv) | Guía completa de figuras y tablas suplementarias en [`supplementary_material_figures_and_tables_guide.md`](./supplementary_material_figures_and_tables_guide.md) | 100% de módulos auditados con clase positiva `y=1` (Stress) y umbral fijo `0.5`; declaraciones de disponibilidad de datos/código y entorno documentados. |

---

## 2. Resumen Corto por Punto del Revisor (Qué se hizo, Qué se encontró, Qué conclusión defendemos)

### Punto 1 — Auditoría del AUROC anómalo (`0.1049`) en Experimento 2 (`Stress-Predict -> WESAD`, 30 s, Standard)
- **¿Qué se hizo?** Se auditó la codificación de etiquetas (`0 = non-stress`, `1 = stress`) en ambos datasets y se recalculó la evaluación sobre las 2,188 ventanas de prueba de WESAD usando tanto la probabilidad de la clase positiva $p_{\text{stress}} = \text{softmax}(\text{logits})[:, 1]$ como su probabilidad invertida $1 - p_{\text{stress}}$, exportando las predicciones ventana por ventana y la curva ROC.
- **¿Qué se encontró?** Evaluando con $p_{\text{stress}}$, el modelo `Standard` a 30 s predice el 100% de las ventanas como estrés ($\text{TP}=656, \text{FP}=1532, \text{TN}=0, \text{FN}=0$), obteniendo $\text{Accuracy}=0.2998$, $\text{F1}=0.4613$, $\text{Balanced Accuracy}=0.5000$ y $\text{AUROC}(p_{\text{stress}})=0.4844$ (con $\text{AUROC}(1-p)=0.5156$, sumando exactamente $1.0000$). El valor `0.1049` reportado previamente correspondió al complemento $1 - 0.8951 = 0.1049$ originado al indexar la columna de la clase opuesta en una corrida donde el *domain shift* invirtió el ordenamiento de puntuaciones.
- **¿Qué conclusión científica podemos defender?** Confirmamos la observación del revisor sobre la inversión de probabilidad ($\text{AUROC}(1-p) = 1 - \text{AUROC}(p)$), corregimos el valor en la Tabla 3 bajo la evaluación verificada con $p_{\text{stress}} = \text{softmax}(\text{logits})[:, 1]$ y reportamos de forma transparente que en transferencia cruzada de 30 s el codificador `Standard` sufre colapso hacia la clase mayoritaria ($\text{AUROC} \approx 0.484$).

---

### Punto 2 — Matrices de Confusión, Métricas por Clase y Predicción de una Sola Clase
- **¿Qué se hizo?** Se generaron las matrices de confusión ($\text{TN, FP, FN, TP}$), métricas por clase (Precision, Recall/Sensitivity, Specificity y F1 para estrés y no-estrés), porcentaje de ventanas predichas como estrés y comparación directa contra el clasificador trivial *Always-Stress* en las 12 configuraciones de los Experimentos 2, 1 y 4 (30 s y 60 s, `Standard` vs. `Hierarchical`).
- **¿Qué se encontró?** En las evaluaciones *holdout* con ventanas de 30 s (tanto en Experimento 2 cruzado como en la partición *holdout* de Stress-Predict), el modelo `Standard` colapsa al 100% de predicciones positivas ($\text{Specificity}=0.0000, \text{Balanced Accuracy}=0.5000$, empatando con *Always-Stress*). En cambio, el modelo `Hierarchical` evita el colapso en 30 s (alcanzando $\text{Specificity}=0.8108$, $\text{Balanced Accuracy}=0.8269$ y $\text{AUROC}=0.9030$ en Stress-Predict *holdout*). En ventanas de 60 s, ambos modelos mantienen especificidad positiva y superan ampliamente al baseline *Always-Stress*.
- **¿Qué conclusión científica podemos defender?** Las matrices de confusión y métricas por clase prueban cuándo ocurre degeneración hacia la clase mayoritaria (en `Standard` con ventanas cortas de 30 s bajo cambio de sujeto/dominio) y demuestran que tanto el codificador jerárquico (en 30 s) como el mayor contexto temporal (en 60 s) previenen dicho colapso.

---

### Punto 3 — Verificación de Convergencia del Standard Encoder
- **¿Qué se hizo?** Se entrenaron `Standard` y `Hierarchical` en Stress-Predict (30 s y 60 s) bajo condiciones idénticas registrando época por época la pérdida de entrenamiento, pérdida de validación, F1 de validación y Balanced Accuracy de validación, verificando la ausencia de `NaN` o divergencia.
- **¿Qué se encontró?** Ambos modelos convergen suavemente y sin inestabilidad numérica (`Sin_NaN = True`). El codificador `Standard` alcanza su mejor checkpoint de validación en la época 7 para 30 s ($\text{Val F1} = 0.8865, \text{Min Val Loss} = 0.3293$) y en la época 8 para 60 s ($\text{Val F1} = 0.8986, \text{Min Val Loss} = 0.3600$).
- **¿Qué conclusión científica podemos defender?** El bajo desempeño del codificador `Standard` en inferencia sobre sujetos/dominios no vistos a 30 s **no se debe a un fallo de optimización ni a falta de convergencia**, sino a una limitación de generalización fuera de la distribución de entrenamiento.

---

### Punto 4 — Auditoría de Equidad del Entrenamiento y Búsqueda de Hiperparámetros
- **¿Qué se hizo?** Se auditaron los archivos JSON de particiones (`splits/`), los registros de Optuna (`results_optuna/`) y los hiperparámetros seleccionados (`configs/selected_hyperparameters.csv`) para `Standard` y `Hierarchical`.
- **¿Qué se encontró?** Ambas arquitecturas utilizaron exactamente las mismas particiones disjuntas por sujeto (28 participantes de desarrollo y 7 de prueba en Stress-Predict, con 0% de solapamiento y cero *leakage*), el mismo presupuesto de búsqueda de Optuna (5 *trials* TPE maximizando F1 de validación), el mismo optimizador AdamW ($lr=2\times 10^{-4}$, *weight decay* $10^{-4}$) y el mismo criterio de selección de checkpoints.
- **¿Qué conclusión científica podemos defender?** Las comparaciones entre `Standard` y `Hierarchical` se realizaron bajo igualdad estricta de presupuesto computacional y de optimización, sin fuga de información de los sujetos de prueba.

---

### Punto 5 — Capacity-Matched Ablation en Stress-Predict (Protocolo Experiment 1 LOGO)
- **¿Qué se hizo?** Se adaptó [`ablation_matched_capacity/run_ablation.py`](../ablation_matched_capacity/run_ablation.py) para ejecutar la validación Leave-One-Group-Out (LOGO) sobre todos los participantes de Stress-Predict ($n=33$ sujetos efectivos en las tablas de ventanas `S02`–`S35`) para `Standard`, `Hierarchical` y `Standard-Matched` en 30 s y 60 s, verificando los conteos reales de parámetros.
- **¿Qué se encontró?**
  - Las diferencias reales de parámetros de `Standard-Matched` respecto a `Hierarchical` son de **+3.83%** en 30 s ($8,118,594$ vs. $7,819,078$) y **+4.80%** en 60 s ($6,539,074$ vs. $6,239,558$), por lo que se eliminó del manuscrito cualquier afirmación de equivalencia `< 4%` para ambas configuraciones.
  - En **Stress-Predict 30 s (LOGO)**, `Standard` obtiene $\text{F1}=0.8609 \pm 0.1302$, `Hierarchical` obtiene $0.8517 \pm 0.1354$ y `Standard-Matched` cae a $0.8447 \pm 0.1425$.
  - En **Stress-Predict 60 s (LOGO)**, los tres modelos obtienen resultados muy cercanos: `Standard` ($\text{F1}=0.8495 \pm 0.1492, \text{AUROC}=0.9020$), `Hierarchical` ($\text{F1}=0.8550 \pm 0.1470, \text{AUROC}=0.8996$) y `Standard-Matched` ($\text{F1}=0.8579 \pm 0.1347, \text{AUROC}=0.8878$).
- **¿Qué conclusión científica podemos defender?** Incrementar el número de parámetros añadiendo capas globales al modelo lineal (`Standard-Matched`) no mejora sistemáticamente el rendimiento supervisado LOGO (y en 30 s lo penaliza ligeramente por sobreajuste).

---

### Punto 6 — Análisis Estadístico Pareado de la Ablación
- **¿Qué se hizo?** Se compararon sujeto a sujeto los resultados de `Hierarchical vs. Standard-Matched`, `Hierarchical vs. Standard` y `Standard-Matched vs. Standard` en Stress-Predict ($n=33$) y WESAD ($n=15$), calculando diferencias medias $\bar{\Delta}$, intervalos de confianza al 95% ($t$ de Student y Bootstrap de 10,000 remuestreos), tamaños del efecto ($r_{\text{rb}}$ y $d_z$), pruebas de Wilcoxon de dos colas y correcciones de Bonferroni y Holm-Bonferroni.
- **¿Qué se encontró?** En las cuatro configuraciones (WESAD y Stress-Predict, 30 s y 60 s), todos los intervalos de confianza al 95% de $\Delta(\text{Hierarchical} - \text{Standard-Matched})$ cruzan el cero y ninguna diferencia supera la corrección por comparaciones múltiples ($p_{\text{raw}} \ge 0.1864$ en Stress-Predict; $p_{\text{Holm},8} = 1.0000$). En Stress-Predict 30 s, el escalado puro de capacidad (`Standard-Matched − Standard`) muestra una caída de $\bar{\Delta}_{\text{F1}} = -0.0162$ ($r_{\text{rb}} = -0.494, p_{\text{raw}} = 0.0201$, no significativo tras Holm $p_{\text{Holm},8} = 0.1608$).
- **¿Qué conclusión científica podemos defender?** Tal como solicitó el Revisor 3, **restringimos la afirmación en la Sección 4.6**: bajo el protocolo puramente supervisado LOGO desde cero (Experimento 1), no existe una diferencia estadísticamente significativa entre las tres variantes tras corregir por multiplicidad; por ende, el control supervisado del Experimento 1 demuestra que añadir más capas globales no beneficia el entrenamiento desde cero, pero no debe presentarse como explicación causal suficiente de las diferencias observadas en otros protocolos (como SSL o transferencia).

---

### Punto 7 — Auditoría Final de Consistencia y Reproducibilidad
- **¿Qué se hizo?** Se ejecutó [`experiments_audit/audit_point7_master_consistency.py`](../experiments_audit/audit_point7_master_consistency.py) para auditar la consistencia de las Tablas 2, 3, 4, 6 y 7, verificar programáticamente la definición de la clase positiva (`y=1` = estrés, `softmax(logits)[:, 1]`, umbral `0.5`) en todo el código y archivos de predicciones, eliminar rutas locales absolutas y habilitar el fallback automático a `data_cache/*.npz`.
- **¿Qué se encontró?** El 100% de los módulos del repositorio utilizan de forma consistente `y=1` como clase positiva de estrés y umbral `0.5`, y todos los scripts son ejecutables de extremo a extremo en cualquier computador usando los cachés incluidos en `data_cache/`.

---

## 3. Guía de Reproducción Completa (Comandos de Ejecución)

Cualquier revisor o investigador puede reproducir todas las tablas y figuras desde la raíz del repositorio con los siguientes comandos:

```bash
# 1. Instalar dependencias (Python >= 3.10, PyTorch con soporte CUDA opcional)
pip install -r requirements.txt

# 2. Reproducir Auditorías de Puntos 1 a 4 (Tablas y Figuras S1–S4)
python experiments_audit/audit_point1.py
python experiments_audit/audit_point2.py
python experiments_audit/audit_point3.py
python experiments_audit/audit_point4.py

# 3. Reproducir Punto 5: Ablación Capacity-Matched en Stress-Predict y WESAD (30s y 60s)
python ablation_matched_capacity/verify_capacity.py
python ablation_matched_capacity/run_ablation.py --dataset stress_predict --window-seconds 60 --device cuda
python ablation_matched_capacity/run_ablation.py --dataset stress_predict --window-seconds 30 --device cuda
python ablation_matched_capacity/run_ablation.py --dataset wesad --window-seconds 60 --device cuda
python ablation_matched_capacity/run_ablation.py --dataset wesad --window-seconds 30 --device cuda
python ablation_matched_capacity/generate_deliverables.py

# 4. Reproducir Punto 6: Análisis Estadístico Pareado y Figuras S7–S8
python ablation_matched_capacity/statistical_analysis.py

# 5. Reproducir Punto 7: Verificación de Clase Positiva, Consistencia de Tablas 2–7 y Matriz de Trazabilidad
python experiments_audit/audit_point7_master_consistency.py
```
