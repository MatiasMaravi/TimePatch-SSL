"""Generate dedicated Figure 3 for paper: Cross-dataset transfer confusion matrices.

Evaluation: Stress-Predict -> WESAD
Models: Standard vs. Hierarchical (30 s and 60 s)
Outputs:
- results/figures/figure3_cross_dataset_confusion_matrices_2x2.png
- results/figures/figure3_cross_dataset_confusion_matrices_1x4.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = '/home/leoisidro/CICLOS/X/PFC_II/TimePatch-SSL'
FIG_DIR = os.path.join(BASE_DIR, 'results/figures')
TABLE_DIR = os.path.join(BASE_DIR, 'results/tables')

os.makedirs(FIG_DIR, exist_ok=True)

def generate_figure3():
    metrics_path = os.path.join(TABLE_DIR, 'audit_point2_per_class_metrics.csv')
    df_cm = pd.read_csv(metrics_path)
    exp2 = df_cm[df_cm['Experimento'] == 'Exp2_Cross'].reset_index(drop=True)

    configs = [
        ('(a) Standard Encoder (30 s)', exp2.iloc[0]),
        ('(b) Hierarchical Encoder (30 s)', exp2.iloc[1]),
        ('(c) Standard Encoder (60 s)', exp2.iloc[2]),
        ('(d) Hierarchical Encoder (60 s)', exp2.iloc[3]),
    ]

    # 1. 2x2 Layout (ideal for 2-column format)
    fig, axes = plt.subplots(2, 2, figsize=(9, 8), dpi=300)
    axes = axes.flatten()

    for idx, (panel_title, row) in enumerate(configs):
        ax = axes[idx]
        cm = np.array([
            [int(row['TN']), int(row['FP'])],
            [int(row['FN']), int(row['TP'])]
        ])
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title(panel_title, fontsize=11, fontweight='bold', pad=8)
        tick_marks = np.arange(2)
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(['Non-stress (0)', 'Stress (1)'], fontsize=10)
        ax.set_yticklabels(['Non-stress (0)', 'Stress (1)'], fontsize=10)
        ax.set_ylabel('True Label', fontsize=10, fontweight='medium')
        ax.set_xlabel('Predicted Label', fontsize=10, fontweight='medium')

        thresh = cm.max() / 2.
        for i in range(2):
            for j in range(2):
                total_in_class = cm[i].sum()
                pct = (cm[i, j] / total_in_class * 100) if total_in_class > 0 else 0
                ax.text(j, i, f'{cm[i, j]:,}\n({pct:.1f}%)',
                        ha='center', va='center',
                        color='white' if cm[i, j] > thresh else 'black',
                        fontsize=10, fontweight='bold')

    plt.tight_layout()
    out_2x2 = os.path.join(FIG_DIR, 'figure3_cross_dataset_confusion_matrices_2x2.png')
    plt.savefig(out_2x2)
    plt.close()
    print(f'[+] Figure 3 (2x2) saved to: {out_2x2}')

    # 2. 1x4 Layout (ideal for full page-width spanning)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), dpi=300)

    for idx, (panel_title, row) in enumerate(configs):
        ax = axes[idx]
        cm = np.array([
            [int(row['TN']), int(row['FP'])],
            [int(row['FN']), int(row['TP'])]
        ])
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title(panel_title, fontsize=11, fontweight='bold', pad=8)
        tick_marks = np.arange(2)
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(['Non-stress (0)', 'Stress (1)'], fontsize=9.5)
        ax.set_yticklabels(['Non-stress (0)', 'Stress (1)'], fontsize=9.5)
        ax.set_ylabel('True Label', fontsize=10, fontweight='medium')
        ax.set_xlabel('Predicted Label', fontsize=10, fontweight='medium')

        thresh = cm.max() / 2.
        for i in range(2):
            for j in range(2):
                total_in_class = cm[i].sum()
                pct = (cm[i, j] / total_in_class * 100) if total_in_class > 0 else 0
                ax.text(j, i, f'{cm[i, j]:,}\n({pct:.1f}%)',
                        ha='center', va='center',
                        color='white' if cm[i, j] > thresh else 'black',
                        fontsize=9.5, fontweight='bold')

    plt.tight_layout()
    out_1x4 = os.path.join(FIG_DIR, 'figure3_cross_dataset_confusion_matrices_1x4.png')
    plt.savefig(out_1x4)
    plt.close()
    print(f'[+] Figure 3 (1x4) saved to: {out_1x4}')

if __name__ == '__main__':
    generate_figure3()
