"""Audit for Reviewer 3 - Point 1: Auditoría del AUROC anómalo (0.1049).

Configuration: Stress-Predict -> WESAD, Standard encoder, 30s windows.
Activities:
- Verify label coding: 0 = non-stress, 1 = stress in both datasets.
- Confirm probability used: softmax(logits)[:, 1] (stress class probability).
- Recalculate AUROC with original p_stress and inverted (1 - p_stress).
- Generate ROC curve figure.
- Save predictions CSV with participant, y_true, p_stress, p_inverted, y_pred.
- Explain whether 0.1049 comes from probability inversion or cross-dataset domain collapse.
"""
import os
import sys
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score,
    roc_auc_score, confusion_matrix, roc_curve, auc
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Working directory and seed
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(BASE_DIR, 'data_cache')
PRED_DIR = os.path.join(BASE_DIR, 'results/predictions')
FIG_DIR = os.path.join(BASE_DIR, 'results/figures')
TABLE_DIR = os.path.join(BASE_DIR, 'results/tables')

os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# Standard (Base) Encoder
class PatchEmbed1D(nn.Module):
    def __init__(self, in_ch=3, patch_len=8, emb_dim=128):
        super().__init__()
        self.patch_len = patch_len
        self.proj = nn.Linear(in_ch * patch_len, emb_dim)

    def forward(self, x):
        B, T, C = x.shape
        P = self.patch_len
        T2 = (T // P) * P
        x = x[:, :T2, :].reshape(B, T2 // P, P * C)
        return self.proj(x)

class StandardTimePatchTransformer(nn.Module):
    def __init__(self, in_ch=3, patch_len=8, emb_dim=256, depth=6, heads=4, dropout=0.2059, num_classes=2):
        super().__init__()
        self.patch = PatchEmbed1D(in_ch, patch_len, emb_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_emb = nn.Parameter(torch.zeros(1, 512, emb_dim))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=heads, dim_feedforward=emb_dim * 4,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.norm = nn.LayerNorm(emb_dim)
        self.ssl_head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim), nn.GELU(), nn.Linear(emb_dim, 64)
        )
        self.cls_head = nn.Linear(emb_dim, num_classes)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

    def forward_features(self, x):
        x = self.patch(x)
        B, N, D = x.shape
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_emb[:, :N+1, :]
        x = self.encoder(x)
        return self.norm(x)[:, 0]

    def forward_ssl(self, x):
        z = self.forward_features(x)
        return F.normalize(self.ssl_head(z), dim=-1)

    def forward(self, x):
        return self.cls_head(self.forward_features(x))

class WindowDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = None if y is None else torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx] if self.y is None else (self.X[idx], self.y[idx])

def weak_augment(x):
    x = x.clone()
    if random.random() < 0.8:
        x = x + 0.02 * torch.randn_like(x)
    if random.random() < 0.5:
        x = x * torch.empty((1, x.shape[1])).uniform_(0.9, 1.1)
    if random.random() < 0.4:
        shift = random.randint(-5, 5)
        x = torch.roll(x, shifts=shift, dims=0)
    if random.random() < 0.3:
        c = random.randrange(x.shape[1])
        start = random.randrange(0, max(1, x.shape[0] - 10))
        width = random.randrange(5, 15)
        x[start:start+width, c] = 0
    return x

class SSLPairDataset(Dataset):
    def __init__(self, X): self.X = torch.tensor(X, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        x = self.X[idx]
        return weak_augment(x), weak_augment(x)

def nt_xent(z1, z2, temp=0.2):
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=-1) / temp
    sim.fill_diagonal_(-1e9)
    targets = torch.arange(B, device=z.device)
    targets = torch.cat([targets + B, targets], dim=0)
    return F.cross_entropy(sim, targets)

def pretrain_ssl(model, X_unlabeled, epochs=4, batch_size=16, lr=1e-3):
    ds = SSLPairDataset(X_unlabeled)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()
    for ep in range(epochs):
        losses = []
        for x1, x2 in dl:
            x1, x2 = x1.to(DEVICE), x2.to(DEVICE)
            loss = nt_xent(model.forward_ssl(x1), model.forward_ssl(x2))
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
    return model

def finetune(model, X_train, y_train, X_val, y_val, epochs=7, batch_size=16, lr=2e-4):
    train_dl = DataLoader(WindowDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(WindowDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    best_state, best_f1 = None, -1
    for ep in range(epochs):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            loss = criterion(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        probs, ys = [], []
        with torch.no_grad():
            for xb, yb in val_dl:
                xb = xb.to(DEVICE)
                p = torch.softmax(model(xb), dim=-1)[:, 1].cpu().numpy()
                probs.append(p); ys.append(yb.numpy())
        probs = np.concatenate(probs); ys = np.concatenate(ys)
        pred = (probs >= 0.5).astype(int)
        val_f1 = f1_score(ys, pred, zero_division=0)
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model

def run_audit_point1():
    print("=" * 70)
    print("PUNTO 1: AUDITORÍA DEL AUROC ANÓMALO (0.1049)")
    print("Configuración: Stress-Predict -> WESAD, Standard Encoder, Ventanas 30 s")
    print("=" * 70)

    # 1. Load Data
    sp_data = np.load(os.path.join(CACHE_DIR, 'stress_predict_30s.npz'))
    wesad_data = np.load(os.path.join(CACHE_DIR, 'wesad_30s.npz'))

    X_sp, y_sp, g_sp = sp_data['X'], sp_data['y'], sp_data['groups']
    X_wesad, y_wesad, g_wesad = wesad_data['X'], wesad_data['y'], wesad_data['groups']

    print(f"[*] Dataset Stress-Predict (30s): {X_sp.shape[0]} ventanas, 3 canales.")
    print(f"    - Clase 0 (Non-stress / Baseline): {np.sum(y_sp == 0)} ({100.0 * np.mean(y_sp == 0):.2f}%)")
    print(f"    - Clase 1 (Stress / Stroop, Interview): {np.sum(y_sp == 1)} ({100.0 * np.mean(y_sp == 1):.2f}%)")
    print(f"[*] Dataset WESAD (30s): {X_wesad.shape[0]} ventanas, 3 canales.")
    print(f"    - Clase 0 (Non-stress / Baseline, Amusement): {np.sum(y_wesad == 0)} ({100.0 * np.mean(y_wesad == 0):.2f}%)")
    print(f"    - Clase 1 (Stress): {np.sum(y_wesad == 1)} ({100.0 * np.mean(y_wesad == 1):.2f}%)")

    # 2. Source train/val split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_sp, y_sp, test_size=0.15, random_state=SEED, stratify=y_sp
    )

    # 3. Model initialization & training
    model = StandardTimePatchTransformer(
        in_ch=3, patch_len=8, emb_dim=256, depth=6, heads=4, dropout=0.2059
    )
    print("\n[*] Entrenando modelo Standard...")
    model = pretrain_ssl(model, X_tr, epochs=4, batch_size=16)
    model = finetune(model, X_tr, y_tr, X_val, y_val, epochs=7, batch_size=16)

    # 4. Evaluation on Target (WESAD)
    test_dl = DataLoader(WindowDataset(X_wesad), batch_size=128, shuffle=False)
    model.eval()
    all_logits = []
    with torch.no_grad():
        for xb in test_dl:
            xb = xb.to(DEVICE)
            logits = model(xb)
            all_logits.append(logits.cpu().numpy())
    all_logits = np.concatenate(all_logits, axis=0)

    # Probabilities
    # Class 0: Non-stress; Class 1: Stress
    exp_logits = np.exp(all_logits - np.max(all_logits, axis=-1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    p_nonstress = probs[:, 0]
    p_stress = probs[:, 1]
    p_inverted = 1.0 - p_stress

    pred_original = (p_stress >= 0.5).astype(int)
    pred_inverted = (p_inverted >= 0.5).astype(int)

    # 5. Metrics computation
    auroc_original = float(roc_auc_score(y_wesad, p_stress))
    auroc_inverted = float(roc_auc_score(y_wesad, p_inverted))
    acc_orig = float(accuracy_score(y_wesad, pred_original))
    f1_orig = float(f1_score(y_wesad, pred_original, zero_division=0))
    bacc_orig = float(balanced_accuracy_score(y_wesad, pred_original))
    cm_orig = confusion_matrix(y_wesad, pred_original, labels=[0, 1])

    acc_inv = float(accuracy_score(y_wesad, pred_inverted))
    f1_inv = float(f1_score(y_wesad, pred_inverted, zero_division=0))
    bacc_inv = float(balanced_accuracy_score(y_wesad, pred_inverted))
    cm_inv = confusion_matrix(y_wesad, pred_inverted, labels=[0, 1])

    print("\n" + "=" * 50)
    print("RESULTADOS DE LA AUDITORÍA:")
    print("=" * 50)
    print(f"AUROC original (usando p_stress = softmax[:, 1]):  {auroc_original:.4f}")
    print(f"AUROC invertido (usando 1 - p_stress = softmax[:, 0]): {auroc_inverted:.4f}")
    print(f"Relación matemática exacta: AUROC_orig + AUROC_inv = {auroc_original + auroc_inverted:.4f} (identidad = 1.0000)")
    print(f"Accuracy original:           {acc_orig:.4f}")
    print(f"F1-score original:           {f1_orig:.4f}")
    print(f"Balanced Accuracy original:  {bacc_orig:.4f}")
    print(f"Matriz de confusión original [ [TN, FP], [FN, TP] ]:\n{cm_orig}")
    print(f"Ventanas predichas como Stress: {np.sum(pred_original == 1)} / {len(y_wesad)} ({100.0 * np.mean(pred_original == 1):.2f}%)")

    # 6. Save Predictions CSV
    pred_df = pd.DataFrame({
        'window_idx': np.arange(len(y_wesad)),
        'participant': [f"S{g}" for g in g_wesad],
        'y_true': y_wesad,
        'logit_nonstress': all_logits[:, 0],
        'logit_stress': all_logits[:, 1],
        'p_stress': p_stress,
        'p_inverted': p_inverted,
        'y_pred_original': pred_original,
        'y_pred_inverted': pred_inverted,
    })
    csv_path = os.path.join(PRED_DIR, 'audit_point1_stresspredict_to_wesad_30s_standard.csv')
    pred_df.to_csv(csv_path, index=False)
    print(f"\n[+] Predicciones guardadas en: {csv_path}")

    # 7. Summary Table
    audit_table = pd.DataFrame([
        {
            'Configuracion': 'Stress-Predict -> WESAD (30s, Standard)',
            'Probabilidad_Evaluada': 'p_stress (softmax[:, 1])',
            'AUROC': auroc_original,
            'Accuracy': acc_orig,
            'F1_Score': f1_orig,
            'Balanced_Accuracy': bacc_orig,
            'TN': cm_orig[0, 0], 'FP': cm_orig[0, 1],
            'FN': cm_orig[1, 0], 'TP': cm_orig[1, 1],
            'Pct_Pred_Stress': 100.0 * np.mean(pred_original == 1)
        },
        {
            'Configuracion': 'Stress-Predict -> WESAD (30s, Standard)',
            'Probabilidad_Evaluada': '1 - p_stress (p_nonstress)',
            'AUROC': auroc_inverted,
            'Accuracy': acc_inv,
            'F1_Score': f1_inv,
            'Balanced_Accuracy': bacc_inv,
            'TN': cm_inv[0, 0], 'FP': cm_inv[0, 1],
            'FN': cm_inv[1, 0], 'TP': cm_inv[1, 1],
            'Pct_Pred_Stress': 100.0 * np.mean(pred_inverted == 1)
        }
    ])
    table_path = os.path.join(TABLE_DIR, 'audit_point1_auroc_table.csv')
    audit_table.to_csv(table_path, index=False)
    print(f"[+] Tabla guardada en: {table_path}")

    # 8. Generate ROC curve figure
    fpr_orig, tpr_orig, _ = roc_curve(y_wesad, p_stress)
    fpr_inv, tpr_inv, _ = roc_curve(y_wesad, p_inverted)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    ax.plot(fpr_orig, tpr_orig, color='#d95f02', lw=2.5,
            label=f'Curva ROC Original (p_stress) [AUROC = {auroc_original:.4f}]')
    ax.plot(fpr_inv, tpr_inv, color='#1b9e77', lw=2.5, linestyle='--',
            label=f'Curva ROC Invertida (1 - p_stress) [AUROC = {auroc_inverted:.4f}]')
    ax.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle=':', label='Azar (AUROC = 0.5000)')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (1 - Especificidad)', fontsize=12)
    ax.set_ylabel('True Positive Rate (Sensibilidad / Recall)', fontsize=12)
    ax.set_title('Auditoría Punto 1: Curva ROC (Stress-Predict → WESAD, Standard 30s)', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11, frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, 'audit_point1_roc_curve.png')
    plt.savefig(fig_path)
    plt.close()
    print(f"[+] Figura ROC guardada en: {fig_path}")

    return {
        'auroc_original': auroc_original,
        'auroc_inverted': auroc_inverted,
        'cm': cm_orig.tolist(),
        'pct_pred_stress': float(np.mean(pred_original == 1))
    }

if __name__ == '__main__':
    run_audit_point1()
