# Reporte Experimental y Estadístico: Ablación Capacity-Matched en Stress-Predict y WESAD (Puntos 5 y 6)

**Proyecto:** TimePatch-SSL — Contrastive Self-Supervised Learning with Hierarchical Time-Patch Transformers for Multimodal Stress Detection  
**Protocolo Evaluado:** Experiment 1 — Clasificación supervisada con validación Leave-One-Group-Out (LOGO) por participante  
**Datasets:** Stress-Predict ($n = 33$ participantes efectivos en tablas de ventanas, de 35 voluntarios) y WESAD ($n = 15$ participantes)  
**Scripts Reproducibles:**
- Ejecución LOSO/LOGO de los 3 modelos: [`ablation_matched_capacity/run_ablation.py`](../ablation_matched_capacity/run_ablation.py)
- Verificación de paridad de parámetros: [`ablation_matched_capacity/verify_capacity.py`](../ablation_matched_capacity/verify_capacity.py)
- Generación de Tabla 7 Extendida y gráficos de barras: [`ablation_matched_capacity/generate_deliverables.py`](../ablation_matched_capacity/generate_deliverables.py)
- Análisis estadístico pareado y gráficos de diferencias (Fase 6): [`ablation_matched_capacity/statistical_analysis.py`](../ablation_matched_capacity/statistical_analysis.py)

---

## 1. Motivación Metodológica y Objetivos

Para aislar el aporte arquitectónico de la representación jerárquica intra-parche frente al número total de parámetros entrenables tanto en **WESAD** como en **Stress-Predict**:
1. **Fase 5 (Experimento de Ablación en Stress-Predict):** Ejecutamos el control `Standard-Matched` junto con `Standard` y `Hierarchical` en **Stress-Predict** (30 s y 60 s) bajo el protocolo **Experiment 1** (validación LOGO por participante, normalización robusta por sujeto `median/IQR` y los mismos hiperparámetros seleccionados por Optuna).
2. **Fase 6 (Análisis Estadístico Pareado):** Evaluamos las diferencias emparejadas sujeto a sujeto ($\Delta_i = A_i - B_i$) entre `Hierarchical vs. Standard-Matched`, `Hierarchical vs. Standard` y `Standard-Matched vs. Standard`, reportando intervalos de confianza al 95% (paramétricos $t$ de Student y no paramétricos por Bootstrap), tamaños del efecto ($r_{\text{rb}}$ de rango biserial y $d_z$ de Cohen), pruebas de rangos con signo de Wilcoxon de dos colas y correcciones por comparaciones múltiples (Bonferroni y Holm-Bonferroni).

---

## 2. Auditoría Exacta de Parámetros Entrenables

Debido a que el ajuste de capacidad en `Standard-Matched` se realiza incrementando un número entero de capas del Transformer global ($L \to L'$), reportamos las diferencias relativas **reales** sin afirmar una equivalencia inferior al 4% para todas las configuraciones:

| Configuración | `emb_dim` | `patch_len` | Depth Base ($L$) | Depth Matched ($L'$) | Standard (Params) | Hierarchical (Params) | Standard-Matched (Params) | $\Delta$ Params (Matched − Hier.) | Dif. Relativa Real (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **WESAD 30 s** | 256 | 8 | 6 | 10 | 4,959,554 | 7,819,078 | 8,118,594 | +299,516 | **+3.83%** |
| **WESAD 60 s** | 128 | 16 | 3 | 8 | 692,034 | 1,728,582 | 1,683,394 | -45,188 | **-2.61%** |
| **Stress-Predict 30 s** | 256 | 8 | 6 | 10 | 4,959,554 | 7,819,078 | 8,118,594 | +299,516 | **+3.83%** |
| **Stress-Predict 60 s** | 256 | 8 | 4 | 8 | 3,380,034 | 6,239,558 | 6,539,074 | +299,516 | **+4.80%** |

- **Archivo fuente:** [`results/tables/ablation_matched_capacity/ablation_matched_capacity_parameters.csv`](../results/tables/ablation_matched_capacity/ablation_matched_capacity_parameters.csv)

---

## 3. Extensión de Table 7: Desempeño Agregado (Media ± DE e IC 95%)

- **Archivo consolidado:** [`results/tables/ablation_matched_capacity/table7_extended_ablation_summary.csv`](../results/tables/ablation_matched_capacity/table7_extended_ablation_summary.csv)
- **Figuras comparativas:**
  - [`results/figures/ablation_matched_capacity_stress_predict.png`](../results/figures/ablation_matched_capacity_stress_predict.png)
  - [`results/figures/ablation_matched_capacity_all_datasets.png`](../results/figures/ablation_matched_capacity_all_datasets.png)

| Dataset | Ventana | Modelo | Depth | Parámetros (Dif. vs. Hier.) | Accuracy (Media ± DE) [IC 95%] | F1-Score (Media ± DE) [IC 95%] | Balanced Acc (Media ± DE) [IC 95%] | AUROC (Media ± DE) [IC 95%] |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stress-Predict** ($n=33$) | **30 s** | Standard | 6 | 4,959,554 (-36.57%) | 0.8351 ± 0.1508 [0.7816, 0.8885] | 0.8609 ± 0.1302 [0.8148, 0.9071] | 0.8307 ± 0.1594 [0.7741, 0.8872] | 0.9030 ± 0.1478 [0.8506, 0.9554] |
| **Stress-Predict** ($n=33$) | **30 s** | Hierarchical | 6 | 7,819,078 (0.00%) | 0.8222 ± 0.1581 [0.7661, 0.8783] | 0.8517 ± 0.1354 [0.8036, 0.8997] | 0.8163 ± 0.1693 [0.7563, 0.8763] | 0.8947 ± 0.1561 [0.8393, 0.9500] |
| **Stress-Predict** ($n=33$) | **30 s** | Standard-Matched | 10 | 8,118,594 (**+3.83%**) | 0.8178 ± 0.1639 [0.7597, 0.8759] | 0.8447 ± 0.1425 [0.7942, 0.8952] | 0.8161 ± 0.1742 [0.7544, 0.8779] | 0.8874 ± 0.1606 [0.8304, 0.9443] |
| **Stress-Predict** ($n=33$) | **60 s** | Standard | 4 | 3,380,034 (-45.83%) | 0.8254 ± 0.1644 [0.7671, 0.8837] | 0.8495 ± 0.1492 [0.7966, 0.9025] | 0.8231 ± 0.1730 [0.7618, 0.8845] | 0.9020 ± 0.1537 [0.8474, 0.9565] |
| **Stress-Predict** ($n=33$) | **60 s** | Hierarchical | 4 | 6,239,558 (0.00%) | 0.8282 ± 0.1724 [0.7671, 0.8893] | 0.8550 ± 0.1470 [0.8028, 0.9071] | 0.8226 ± 0.1841 [0.7573, 0.8879] | 0.8996 ± 0.1548 [0.8447, 0.9545] |
| **Stress-Predict** ($n=33$) | **60 s** | Standard-Matched | 8 | 6,539,074 (**+4.80%**) | 0.8332 ± 0.1545 [0.7785, 0.8880] | 0.8579 ± 0.1347 [0.8101, 0.9056] | 0.8314 ± 0.1644 [0.7731, 0.8896] | 0.8878 ± 0.1754 [0.8256, 0.9500] |
| **WESAD** ($n=15$) | **30 s** | Standard | 6 | 4,959,554 (-36.57%) | 0.8661 ± 0.1272 [0.7957, 0.9365] | 0.7604 ± 0.2639 [0.6143, 0.9065] | 0.8462 ± 0.1718 [0.7510, 0.9413] | 0.9004 ± 0.1765 [0.8026, 0.9982] |
| **WESAD** ($n=15$) | **30 s** | Hierarchical | 6 | 7,819,078 (0.00%) | 0.8630 ± 0.2203 [0.7410, 0.9850] | 0.7721 ± 0.3572 [0.5743, 0.9699] | 0.8417 ± 0.2582 [0.6987, 0.9847] | 0.8582 ± 0.2882 [0.6985, 1.0000] |
| **WESAD** ($n=15$) | **30 s** | Standard-Matched | 10 | 8,118,594 (**+3.83%**) | 0.8678 ± 0.1779 [0.7692, 0.9664] | 0.7753 ± 0.2993 [0.6095, 0.9411] | 0.8545 ± 0.1999 [0.7438, 0.9652] | 0.8798 ± 0.2181 [0.7589, 1.0000] |
| **WESAD** ($n=15$) | **60 s** | Standard | 3 | 692,034 (-59.97%) | 0.8752 ± 0.1587 [0.7873, 0.9631] | 0.7746 ± 0.3113 [0.6022, 0.9470] | 0.8557 ± 0.2026 [0.7435, 0.9679] | 0.9162 ± 0.1616 [0.8267, 1.0000] |
| **WESAD** ($n=15$) | **60 s** | Hierarchical | 3 | 1,728,582 (0.00%) | 0.8602 ± 0.1722 [0.7649, 0.9556] | 0.7653 ± 0.3159 [0.5904, 0.9403] | 0.8466 ± 0.2087 [0.7310, 0.9621] | 0.9098 ± 0.1656 [0.8180, 1.0000] |
| **WESAD** ($n=15$) | **60 s** | Standard-Matched | 8 | 1,683,394 (**-2.61%**) | 0.8614 ± 0.1820 [0.7606, 0.9622] | 0.7430 ± 0.3539 [0.5470, 0.9390] | 0.8342 ± 0.2351 [0.7040, 0.9643] | 0.8688 ± 0.2457 [0.7327, 1.0000] |

---

## 4. Fase 6: Análisis Estadístico Pareado por Participante

- **Tabla estadística completa:** [`results/tables/ablation_matched_capacity/ablation_matched_capacity_statistical_analysis.csv`](../results/tables/ablation_matched_capacity/ablation_matched_capacity_statistical_analysis.csv)
- **Figuras de diferencias pareadas:**
  - Diferencias individuales por sujeto en Stress-Predict ($S02 \dots S35$): [`results/figures/ablation_paired_differences_stress_predict.png`](../results/figures/ablation_paired_differences_stress_predict.png)
  - Distribución de diferencias pareadas, IC 95%, tamaño del efecto y $p$-valores: [`results/figures/ablation_paired_differences_summary.png`](../results/figures/ablation_paired_differences_summary.png)

### 4.1 Metodología Estadística
Para cada participante $i \in \{1, \dots, n\}$ se calcula la diferencia pareada $\Delta_i = M_{A,i} - M_{B,i}$ bajo exactamente las mismas particiones de entrenamiento/validación/prueba:
1. **Estimadores de magnitud y dispersión:** Media $\bar{\Delta}$, desviación estándar muestral $s_{\Delta}$ ($ddof=1$), mediana $\text{Med}(\Delta)$ y rango intercuartílico ($\text{IQR}$).
2. **Intervalos de confianza al 95% para $\bar{\Delta}$:**
   - Paramétrico por distribución $t$ de Student ($\bar{\Delta} \pm t_{0.975, n-1} \cdot s_{\Delta}/\sqrt{n}$).
   - No paramétrico por Bootstrap percentil ($10,000$ remuestreos con reemplazo, semilla fija `42`).
3. **Tamaños del efecto:**
   - **Correlación rango-biserial ($r_{\text{rb}}$):** $\frac{R^+ - R^-}{R^+ + R^-}$ sobre los rangos con signo no nulos (misma métrica de tamaño del efecto empleada en la Tabla 4 del manuscrito).
   - **$d_z$ de Cohen para muestras pareadas:** $d_z = \bar{\Delta} / s_{\Delta}$.
4. **Pruebas de hipótesis y corrección por multiplicidad:**
   - Prueba no paramétrica de **rangos con signo de Wilcoxon** de dos colas ($W$, $p_{\text{raw}}$) y prueba $t$ pareada de referencia ($t$, $p_t$).
   - Corrección de **Holm-Bonferroni** ($p_{\text{Holm},8}$) y **Bonferroni** ($\alpha = 0.05/8 = 0.00625$ para las 8 métricas por dataset [2 ventanas × 4 métricas], así como el umbral de la Tabla 4 del manuscrito $\alpha = 0.05/6 \approx 0.0083$).

---

### 4.2 Tabla Estadística de Comparaciones en Stress-Predict ($n = 33$)

| Ventana | Comparación ($A$ vs. $B$) | Métrica | $\bar{\Delta} \pm s_{\Delta}$ | $\text{IC}_{95\%}(t)$ | $\text{IC}_{95\%}(\text{Boot})$ | Mediana (IQR) | Gana/Empata/Pierde | $d_z$ | $r_{\text{rb}}$ | Wilcoxon $W$ | $p_{\text{raw}}$ | $p_{\text{Holm},8}$ | ¿Sig. Corregida? |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **30 s** | **Hierarchical vs. Standard-Matched** | Accuracy | +0.0044 ± 0.0411 | [-0.0101, +0.0190] | [-0.0091, +0.0184] | 0.0000 (0.0415) | 14 / 8 / 11 | +0.108 | +0.145 | 139.0 | 0.5272 | 1.0000 | No |
| **30 s** | **Hierarchical vs. Standard-Matched** | F1-Score | +0.0069 ± 0.0329 | [-0.0048, +0.0186] | [-0.0039, +0.0181] | +0.0008 (0.0332) | 16 / 6 / 11 | +0.210 | +0.291 | 134.0 | 0.1864 | 1.0000 | No |
| **30 s** | **Hierarchical vs. Standard-Matched** | Balanced Acc | +0.0002 ± 0.0480 | [-0.0168, +0.0172] | [-0.0156, +0.0165] | 0.0000 (0.0471) | 14 / 6 / 13 | +0.004 | -0.027 | 184.0 | 0.9044 | 1.0000 | No |
| **30 s** | **Hierarchical vs. Standard-Matched** | AUROC | +0.0073 ± 0.0438 | [-0.0082, +0.0228] | [-0.0074, +0.0221] | +0.0004 (0.0294) | 17 / 3 / 13 | +0.167 | +0.256 | 173.0 | 0.2210 | 1.0000 | No |
| **30 s** | **Hierarchical vs. Standard** | Accuracy | -0.0129 ± 0.0421 | [-0.0278, +0.0021] | [-0.0272, +0.0008] | -0.0095 (0.0385) | 11 / 6 / 16 | -0.306 | -0.423 | 109.0 | 0.0546 | 0.3822 | No |
| **30 s** | **Hierarchical vs. Standard** | F1-Score | -0.0093 ± 0.0322 | [-0.0207, +0.0021] | [-0.0203, +0.0011] | -0.0051 (0.0326) | 12 / 5 / 16 | -0.289 | -0.301 | 142.0 | 0.1648 | 0.9889 | No |
| **30 s** | **Hierarchical vs. Standard** | Balanced Acc | -0.0144 ± 0.0482 | [-0.0314, +0.0027] | [-0.0309, +0.0012] | -0.0083 (0.0352) | 12 / 5 / 16 | -0.298 | -0.434 | 115.0 | 0.0451 | 0.3606 | No |
| **30 s** | **Hierarchical vs. Standard** | AUROC | -0.0083 ± 0.0346 | [-0.0206, +0.0040] | [-0.0207, +0.0022] | -0.0004 (0.0110) | 11 / 5 / 17 | -0.240 | -0.222 | 158.0 | 0.3055 | 1.0000 | No |
| **30 s** | **Standard-Matched vs. Standard** | Accuracy | -0.0173 ± 0.0513 | [-0.0355, +0.0009] | [-0.0353, -0.0006] | -0.0096 (0.0420) | 8 / 5 / 20 | -0.337 | -0.429 | 116.0 | 0.0476 | 0.3329 | No |
| **30 s** | **Standard-Matched vs. Standard** | F1-Score | -0.0162 ± 0.0387 | [-0.0299, -0.0025] | [-0.0297, -0.0035] | -0.0091 (0.0356) | 8 / 4 / 21 | -0.419 | -0.494 | 110.0 | 0.0201 | 0.1608 | No |
| **30 s** | **Standard-Matched vs. Standard** | Balanced Acc | -0.0146 ± 0.0609 | [-0.0361, +0.0070] | [-0.0361, +0.0049] | -0.0099 (0.0419) | 10 / 4 / 19 | -0.239 | -0.287 | 155.0 | 0.1765 | 0.8826 | No |
| **30 s** | **Standard-Matched vs. Standard** | AUROC | -0.0156 ± 0.0478 | [-0.0326, +0.0013] | [-0.0326, -0.0004] | -0.0005 (0.0281) | 13 / 3 / 17 | -0.327 | -0.303 | 162.0 | 0.1470 | 0.8822 | No |
| **60 s** | **Hierarchical vs. Standard-Matched** | Accuracy | -0.0050 ± 0.0523 | [-0.0236, +0.0135] | [-0.0226, +0.0124] | 0.0000 (0.0408) | 10 / 6 / 17 | -0.097 | -0.225 | 146.5 | 0.3072 | 1.0000 | No |
| **60 s** | **Hierarchical vs. Standard-Matched** | F1-Score | -0.0029 ± 0.0428 | [-0.0181, +0.0122] | [-0.0172, +0.0114] | -0.0046 (0.0330) | 10 / 6 / 17 | -0.068 | -0.177 | 155.5 | 0.4209 | 1.0000 | No |
| **60 s** | **Hierarchical vs. Standard-Matched** | Balanced Acc | -0.0088 ± 0.0549 | [-0.0283, +0.0107] | [-0.0273, +0.0093] | -0.0105 (0.0468) | 10 / 6 / 17 | -0.160 | -0.217 | 148.0 | 0.3246 | 1.0000 | No |
| **60 s** | **Hierarchical vs. Standard-Matched** | AUROC | +0.0118 ± 0.0495 | [-0.0058, +0.0294] | [-0.0033, +0.0292] | 0.0000 (0.0129) | 9 / 8 / 16 | +0.238 | -0.009 | 161.0 | 0.9678 | 1.0000 | No |
| **60 s** | **Hierarchical vs. Standard** | Accuracy | +0.0028 ± 0.0700 | [-0.0220, +0.0277] | [-0.0209, +0.0265] | 0.0000 (0.0426) | 13 / 8 / 12 | +0.040 | +0.059 | 153.5 | 0.8086 | 1.0000 | No |
| **60 s** | **Hierarchical vs. Standard** | F1-Score | +0.0054 ± 0.0503 | [-0.0124, +0.0233] | [-0.0114, +0.0226] | 0.0000 (0.0420) | 15 / 6 / 12 | +0.108 | +0.090 | 172.0 | 0.6830 | 1.0000 | No |
| **60 s** | **Hierarchical vs. Standard** | Balanced Acc | -0.0006 ± 0.0788 | [-0.0285, +0.0274] | [-0.0275, +0.0258] | 0.0000 (0.0546) | 13 / 6 / 14 | -0.007 | +0.053 | 179.0 | 0.8101 | 1.0000 | No |
| **60 s** | **Hierarchical vs. Standard** | AUROC | -0.0024 ± 0.0236 | [-0.0108, +0.0060] | [-0.0107, +0.0053] | 0.0000 (0.0102) | 13 / 8 / 12 | -0.103 | +0.027 | 146.0 | 0.9090 | 1.0000 | No |
| **60 s** | **Standard-Matched vs. Standard** | Accuracy | +0.0079 ± 0.0559 | [-0.0120, +0.0277] | [-0.0107, +0.0268] | 0.0000 (0.0408) | 16 / 5 / 12 | +0.141 | +0.203 | 186.0 | 0.3388 | 1.0000 | No |
| **60 s** | **Standard-Matched vs. Standard** | F1-Score | +0.0083 ± 0.0467 | [-0.0082, +0.0249] | [-0.0070, +0.0242] | +0.0046 (0.0407) | 17 / 4 / 12 | +0.179 | +0.175 | 205.0 | 0.3994 | 1.0000 | No |
| **60 s** | **Standard-Matched vs. Standard** | Balanced Acc | +0.0082 ± 0.0607 | [-0.0133, +0.0297] | [-0.0120, +0.0289] | +0.0091 (0.0510) | 16 / 5 / 12 | +0.136 | +0.219 | 195.0 | 0.2989 | 1.0000 | No |
| **60 s** | **Standard-Matched vs. Standard** | AUROC | -0.0142 ± 0.0578 | [-0.0347, +0.0063] | [-0.0355, +0.0033] | 0.0000 (0.0145) | 15 / 6 / 12 | -0.246 | -0.061 | 177.0 | 0.7731 | 1.0000 | No |

---

### 4.3 Tabla Estadística de Comparaciones en WESAD ($n = 15$)

| Ventana | Comparación ($A$ vs. $B$) | Métrica | $\bar{\Delta} \pm s_{\Delta}$ | $\text{IC}_{95\%}(t)$ | Mediana (IQR) | Gana/Empata/Pierde | $d_z$ | $r_{\text{rb}}$ | Wilcoxon $W$ | $p_{\text{raw}}$ | $p_{\text{Holm},8}$ | ¿Sig. Corregida? |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **30 s** | **Hierarchical vs. Standard-Matched** | Accuracy | -0.0048 ± 0.0674 | [-0.0421, +0.0325] | 0.0000 (0.0346) | 7 / 1 / 7 | -0.071 | -0.010 | 52.0 | 0.9750 | 1.0000 | No |
| **30 s** | **Hierarchical vs. Standard-Matched** | F1-Score | -0.0032 ± 0.1144 | [-0.0665, +0.0602] | 0.0000 (0.0472) | 8 / 1 / 6 | -0.028 | +0.124 | 46.0 | 0.6832 | 1.0000 | No |
| **30 s** | **Hierarchical vs. Standard-Matched** | Balanced Acc | -0.0127 ± 0.0766 | [-0.0552, +0.0297] | 0.0000 (0.0247) | 7 / 1 / 7 | -0.166 | -0.067 | 49.0 | 0.8261 | 1.0000 | No |
| **30 s** | **Hierarchical vs. Standard-Matched** | AUROC | -0.0216 ± 0.1815 | [-0.1221, +0.0789] | 0.0000 (0.0110) | 5 / 3 / 7 | -0.119 | -0.128 | 34.0 | 0.7221 | 1.0000 | No |
| **30 s** | **Hierarchical vs. Standard** | Accuracy | -0.0031 ± 0.1247 | [-0.0721, +0.0660] | +0.0068 (0.0345) | 8 / 2 / 5 | -0.025 | +0.187 | 37.0 | 0.5525 | 1.0000 | No |
| **30 s** | **Hierarchical vs. Standard** | F1-Score | +0.0117 ± 0.1332 | [-0.0621, +0.0855] | +0.0112 (0.0569) | 8 / 2 / 5 | +0.088 | +0.187 | 37.0 | 0.5525 | 1.0000 | No |
| **30 s** | **Hierarchical vs. Standard** | Balanced Acc | -0.0045 ± 0.1066 | [-0.0635, +0.0545] | +0.0049 (0.0372) | 8 / 2 / 5 | -0.042 | +0.121 | 40.0 | 0.7007 | 1.0000 | No |
| **30 s** | **Hierarchical vs. Standard** | AUROC | -0.0422 ± 0.1327 | [-0.1156, +0.0313] | 0.0000 (0.0108) | 7 / 3 / 5 | -0.318 | -0.128 | 34.0 | 0.6949 | 1.0000 | No |
| **60 s** | **Hierarchical vs. Standard-Matched** | Accuracy | -0.0012 ± 0.0266 | [-0.0159, +0.0136] | 0.0000 (0.0144) | 3 / 7 / 5 | -0.043 | -0.222 | 14.0 | 0.5749 | 1.0000 | No |
| **60 s** | **Hierarchical vs. Standard-Matched** | F1-Score | +0.0223 ± 0.1010 | [-0.0336, +0.0782] | 0.0000 (0.0191) | 3 / 7 / 5 | +0.221 | 0.000 | 18.0 | 1.0000 | 1.0000 | No |
| **60 s** | **Hierarchical vs. Standard-Matched** | Balanced Acc | +0.0124 ± 0.0659 | [-0.0241, +0.0489] | 0.0000 (0.0102) | 3 / 7 / 5 | +0.188 | -0.111 | 16.0 | 0.7794 | 1.0000 | No |
| **60 s** | **Hierarchical vs. Standard-Matched** | AUROC | +0.0410 ± 0.0970 | [-0.0128, +0.0947] | 0.0000 (0.0246) | 4 / 9 / 2 | +0.422 | +0.714 | 3.0 | 0.1159 | 0.9268 | No |
| **60 s** | **Hierarchical vs. Standard** | Accuracy | -0.0150 ± 0.0717 | [-0.0547, +0.0247] | 0.0000 (0.0144) | 5 / 7 / 3 | -0.210 | -0.167 | 15.0 | 0.6737 | 1.0000 | No |
| **60 s** | **Hierarchical vs. Standard** | F1-Score | -0.0093 ± 0.0746 | [-0.0506, +0.0320] | 0.0000 (0.0169) | 5 / 8 / 2 | -0.125 | +0.214 | 11.0 | 0.6121 | 1.0000 | No |
| **60 s** | **Hierarchical vs. Standard** | Balanced Acc | -0.0091 ± 0.0535 | [-0.0388, +0.0205] | 0.0000 (0.0102) | 5 / 7 / 3 | -0.171 | -0.056 | 17.0 | 0.8886 | 1.0000 | No |
| **60 s** | **Hierarchical vs. Standard** | AUROC | -0.0064 ± 0.0344 | [-0.0255, +0.0126] | 0.0000 (0.0014) | 4 / 9 / 2 | -0.187 | -0.048 | 10.0 | 0.9165 | 1.0000 | No |

---

## 5. Interpretación Científica (Sin Atribuciones Causales No Demostradas)

La evidencia estadística pareada conduce a las siguientes conclusiones objetivas y libres de sobre-atribución causal:

### 5.1 ¿Respalda la evidencia del Experimento 1 una ventaja sistemática de `Hierarchical`?
- **No bajo el protocolo supervisado LOGO desde cero (Experiment 1).** En las cuatro configuraciones evaluadas (Stress-Predict 30 s y 60 s con $n=33$; WESAD 30 s y 60 s con $n=15$), todos los intervalos de confianza al 95% para las diferencias pareadas $\Delta(\text{Hierarchical} - \text{Standard-Matched})$ y $\Delta(\text{Hierarchical} - \text{Standard})$ incluyen el cero ($0.0000$).
- Ninguna comparación entre `Hierarchical` y `Standard-Matched` alcanza significancia estadística antes ($p_{\text{raw}} \in [0.1159, 1.0000]$) ni después de corregir por comparaciones múltiples mediante Holm-Bonferroni o Bonferroni ($p_{\text{Holm},8} \ge 0.9268$).

### 5.2 Descomposición aditiva entre Capacidad y Arquitectura en Stress-Predict
Toda diferencia observada entre el modelo `Hierarchical` y el modelo base `Standard` puede descomponerse exactamente como:
$$\underbrace{\Delta(\text{Hierarchical} - \text{Standard})}_{\text{Diferencia Total}} = \underbrace{\Delta(\text{Hierarchical} - \text{Standard-Matched})}_{\text{Efecto Arquitectónico a Capacidad Equiparada}} + \underbrace{\Delta(\text{Standard-Matched} - \text{Standard})}_{\text{Efecto de Escalado de Capacidad Global}}$$

1. **En Stress-Predict 30 s ($n=33$):**
   - **Efecto de escalar capacidad global ($\text{Standard-Matched} - \text{Standard}$):** Incrementar los parámetros entrenables de $4.96\text{M}$ ($L=6$) a $8.12\text{M}$ ($L'=10$) produce un efecto **negativo** consistente en las cuatro métricas ($\bar{\Delta}_{\text{Acc}} = -0.0173$, $p_{\text{raw}} = 0.0476$; $\bar{\Delta}_{\text{F1}} = -0.0162$, $\text{IC}_{95\%} = [-0.0299, -0.0025]$, $r_{\text{rb}} = -0.494$, $p_{\text{raw}} = 0.0201$, donde `Standard-Matched` pierde en 21 de 33 sujetos y solo gana en 8). Aunque tras corrección de Holm-Bonferroni para 8 pruebas ($p_{\text{Holm},8} = 0.1608$) no supera el umbral estricto corregido, la tendencia empírica muestra que añadir capas globales profundas sin preentrenamiento tiende a sobreajustar en ventanas cortas de 30 s.
   - **Efecto arquitectónico a capacidad equiparada ($\text{Hierarchical} - \text{Standard-Matched}$):** Pese a tener un 3.83% menos de parámetros que `Standard-Matched`, `Hierarchical` atenúa parcialmente esa caída ($\bar{\Delta}_{\text{F1}} = +0.0069$, $r_{\text{rb}} = +0.291$, 16 victorias vs. 11 derrotas; $\bar{\Delta}_{\text{AUROC}} = +0.0073$, $r_{\text{rb}} = +0.256$), pero la magnitud del efecto es pequeña y estadísticamente no significativa ($p_{\text{raw}} = 0.1864$, $p_{\text{Holm},8} = 1.0000$).
2. **En Stress-Predict 60 s ($n=33$):**
   - Las diferencias medias entre los tres modelos son inferiores a $0.009$ puntos en Accuracy, F1 y Balanced Accuracy ($\bar{\Delta}_{\text{F1}}(\text{Hier} - \text{Matched}) = -0.0029$, $r_{\text{rb}} = -0.177$, $p_{\text{raw}} = 0.4209$; $\bar{\Delta}_{\text{F1}}(\text{Hier} - \text{Standard}) = +0.0054$, $r_{\text{rb}} = +0.090$, $p_{\text{raw}} = 0.6830$). Los tamaños del efecto son despreciables a pequeños ($|r_{\text{rb}}| \le 0.225$, $|d_z| \le 0.238$), indicando equivalencia práctica bajo entrenamiento supervisado directo en ventanas de 60 s.

### 5.3 Restricción Expresa de Afirmaciones Causales en el Manuscrito (§4.6)
1. **Lo que la ablación sí demuestra:** Aumentar el número de parámetros mediante mayor profundidad global (`Standard-Matched`) no produce mejoras sistemáticas bajo entrenamiento supervisado LOGO en WESAD ni en Stress-Predict (e incluso penaliza ligeramente el desempeño en ventanas de 30 s en Stress-Predict).
2. **Lo que no debe afirmarse a partir de esta ablación:** Dado que bajo el protocolo supervisado desde cero (Experiment 1 con normalización robusta por participante) las diferencias entre `Standard`, `Hierarchical` y `Standard-Matched` son pequeñas y estadísticamente no significativas tras corrección por multiplicidad, **no se puede atribuir una ventaja sistemática al codificador `Hierarchical` basándose únicamente en el Experimento 1**, ni extrapolar directamente este control supervisado como explicación causal suficiente de las diferencias observadas en regímenes con preentrenamiento auto-supervisado (SSL) o transferencia entre dominios. El texto de la Sección 4.6 se restringe explícitamente a este alcance.
