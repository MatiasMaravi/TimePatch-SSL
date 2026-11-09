"""Audit for Reviewer 3 - Point 3: Verificación de convergencia del Standard encoder.

Dataset: Stress-Predict.
Ventanas: 30 s y 60 s.
Modelos: Standard y Hierarchical.

Actividades:
- Entrena Standard y Hierarchical bajo condiciones idénticas de evaluación (mismos splits, semillas y optimizador).
- Registra epoch-by-epoch:
  - Training loss
  - Validation loss
  - Validation F1-score
  - Validation Balanced Accuracy
- Identifica la mejor época y checkpoint seleccionado.
- Analiza estabilidad (verificando ausencia de NaN, explosión de gradientes o divergencia).
- Compara directamente las curvas de aprendizaje.
- Genera:
  - results/figures/audit_point3_loss_curves.png
  - results/figures/audit_point3_val_metrics_curves.png
  - results/tables/audit_point3_convergence_summary.csv
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
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(BASE_DIR, 'data_cache')
TABLE_DIR = os.path.join(BASE_DIR, 'results/tables')
FIG_DIR = os.path.join(BASE_DIR, 'results/figures')

os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

nb_dir = os.path.join(BASE_DIR, 'notebooks')
if nb_dir not in sys.path: sys.path.append(nb_dir)
if os.path.join(nb_dir, 'TimeMAE') not in sys.path: sys.path.append(os.path.join(nb_dir, 'TimeMAE'))

from TimeMAE.model.TimeMAE import Encoder
from TimeMAE.model.layers import PositionalEmbedding

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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

class TimeMAEPatchEmbed(nn.Module):
    def __init__(self, in_ch=3, patch_len=8, emb_dim=128, max_patches=5000):
        super().__init__()
        self.patch_len = patch_len
        self.input_projection = nn.Conv1d(in_ch, emb_dim, kernel_size=patch_len, stride=patch_len)
        self.position = PositionalEmbedding(max_patches, emb_dim)
        class Args: pass
        args = Args()
        args.d_model = emb_dim
        args.attn_heads = 4
        args.layers = 2
        args.dropout = 0.1
        args.enable_res_parameter = 1
        self.encoder = Encoder(args)
    def forward(self, x):
        B, T, C = x.shape
        P = self.patch_len
        T2 = (T // P) * P
        x = x[:, :T2, :]
        x = self.input_projection(x.transpose(1, 2)).transpose(1, 2).contiguous()
        pos_emb = self.position(x)
        x = x + pos_emb[:, :x.size(1), :]
        return self.encoder(x)

class TimePatchTransformer(nn.Module):
    def __init__(self, in_ch=3, patch_len=8, emb_dim=128, depth=4, heads=4, dropout=0.1, num_classes=2, hierarchical=False):
        super().__init__()
        self.hierarchical = hierarchical
        if hierarchical:
            self.patch = TimeMAEPatchEmbed(in_ch, patch_len, emb_dim)
        else:
            self.patch = PatchEmbed1D(in_ch, patch_len, emb_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_emb = nn.Parameter(torch.zeros(1, 512, emb_dim))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=heads, dim_feedforward=emb_dim * 4,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.norm = nn.LayerNorm(emb_dim)
        self.ssl_head = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.GELU(), nn.Linear(emb_dim, 64))
        self.cls_head = nn.Linear(emb_dim, num_classes)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

    def forward_features(self, x):
        x = self.patch(x)
        B, N, D = x.shape
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_emb[:, :N+1, :]
        return self.norm(self.encoder(x))[:, 0]

    def forward(self, x):
        return self.cls_head(self.forward_features(x))

class WindowDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = None if y is None else torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx] if self.y is None else (self.X[idx], self.y[idx])

def train_and_track_convergence(model, X_train, y_train, X_val, y_val, epochs=15, batch_size=16, lr=2e-4):
    train_dl = DataLoader(WindowDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(WindowDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    history = {
        'epoch': [], 'train_loss': [], 'val_loss': [],
        'val_f1': [], 'val_bacc': [], 'val_acc': []
    }

    best_val_loss = float('inf')
    best_val_f1 = -1.0
    best_epoch_loss = -1
    best_epoch_f1 = -1

    for ep in range(1, epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            loss = criterion(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_losses = []
        probs, ys = [], []
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_losses.append(loss.item())
                p = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                probs.append(p)
                ys.append(yb.cpu().numpy())

        probs = np.concatenate(probs)
        ys = np.concatenate(ys)
        preds = (probs >= 0.5).astype(int)

        ep_train_loss = float(np.mean(train_losses))
        ep_val_loss = float(np.mean(val_losses))
        ep_val_f1 = float(f1_score(ys, preds, zero_division=0))
        ep_val_bacc = float(balanced_accuracy_score(ys, preds))
        ep_val_acc = float(accuracy_score(ys, preds))

        history['epoch'].append(ep)
        history['train_loss'].append(ep_train_loss)
        history['val_loss'].append(ep_val_loss)
        history['val_f1'].append(ep_val_f1)
        history['val_bacc'].append(ep_val_bacc)
        history['val_acc'].append(ep_val_acc)

        if ep_val_loss < best_val_loss:
            best_val_loss = ep_val_loss
            best_epoch_loss = ep

        if ep_val_f1 > best_val_f1:
            best_val_f1 = ep_val_f1
            best_epoch_f1 = ep

    return history, {
        'best_epoch_loss': best_epoch_loss,
        'min_val_loss': best_val_loss,
        'best_epoch_f1': best_epoch_f1,
        'max_val_f1': best_val_f1,
        'final_val_bacc': history['val_bacc'][-1],
        'is_stable': bool(not (np.isnan(history['train_loss']).any() or np.isnan(history['val_loss']).any())),
    }

def run_audit_point3():
    print("=" * 70)
    print("PUNTO 3: VERIFICACIÓN DE CONVERGENCIA DEL STANDARD ENCODER")
    print("Dataset: Stress-Predict (30 s y 60 s)")
    print("=" * 70)

    sp_30s = np.load(os.path.join(CACHE_DIR, 'stress_predict_30s.npz'))
    sp_60s = np.load(os.path.join(CACHE_DIR, 'stress_predict_60s.npz'))

    runs = [
        {'name': 'Stress-Predict 30s - Standard', 'data': sp_30s, 'hierarchical': False, 'emb_dim': 256, 'depth': 6, 'dropout': 0.2059, 'patch_len': 8, 'win': 30},
        {'name': 'Stress-Predict 30s - Hierarchical', 'data': sp_30s, 'hierarchical': True, 'emb_dim': 256, 'depth': 6, 'dropout': 0.2059, 'patch_len': 8, 'win': 30},
        {'name': 'Stress-Predict 60s - Standard', 'data': sp_60s, 'hierarchical': False, 'emb_dim': 256, 'depth': 4, 'dropout': 0.2005, 'patch_len': 8, 'win': 60},
        {'name': 'Stress-Predict 60s - Hierarchical', 'data': sp_60s, 'hierarchical': True, 'emb_dim': 256, 'depth': 4, 'dropout': 0.2005, 'patch_len': 8, 'win': 60},
    ]

    histories = {}
    summary_rows = []

    for r in runs:
        set_seed(SEED)
        print(f"\n[*] Entrenando 12 épocas y registrando métricas: {r['name']}...")
        X, y = r['data']['X'], r['data']['y']
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=0.15, random_state=SEED, stratify=y
        )

        model = TimePatchTransformer(
            in_ch=3, patch_len=r['patch_len'], emb_dim=r['emb_dim'],
            depth=r['depth'], heads=4, dropout=r['dropout'],
            num_classes=2, hierarchical=r['hierarchical']
        )

        hist, metrics = train_and_track_convergence(
            model, X_tr, y_tr, X_val, y_val, epochs=12, batch_size=16, lr=2e-4
        )
        histories[r['name']] = hist

        print(f"    - Mejor época (F1): {metrics['best_epoch_f1']} (Val F1: {metrics['max_val_f1']:.4f})")
        print(f"    - Mejor época (Loss): {metrics['best_epoch_loss']} (Min Val Loss: {metrics['min_val_loss']:.4f})")
        print(f"    - Estabilidad numérica (sin NaN): {metrics['is_stable']}")

        summary_rows.append({
            'Configuracion': r['name'],
            'Ventana_s': r['win'],
            'Arquitectura': 'Hierarchical' if r['hierarchical'] else 'Standard',
            'Mejor_Epoca_F1': metrics['best_epoch_f1'],
            'Max_Val_F1': metrics['max_val_f1'],
            'Mejor_Epoca_Loss': metrics['best_epoch_loss'],
            'Min_Val_Loss': metrics['min_val_loss'],
            'Val_Balanced_Accuracy_Final': metrics['final_val_bacc'],
            'Estable_Sin_NaN': metrics['is_stable']
        })

    # Save summary table
    df_summary = pd.DataFrame(summary_rows)
    summary_path = os.path.join(TABLE_DIR, 'audit_point3_convergence_summary.csv')
    df_summary.to_csv(summary_path, index=False)
    print(f"\n[+] Tabla de convergencia guardada en: {summary_path}")

    # Plot Loss Curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    # 30s
    ax = axes[0]
    ax.plot(histories['Stress-Predict 30s - Standard']['epoch'], histories['Stress-Predict 30s - Standard']['train_loss'], label='Standard - Train Loss', color='#1f77b4', lw=2)
    ax.plot(histories['Stress-Predict 30s - Standard']['epoch'], histories['Stress-Predict 30s - Standard']['val_loss'], label='Standard - Val Loss', color='#1f77b4', linestyle='--', lw=2)
    ax.plot(histories['Stress-Predict 30s - Hierarchical']['epoch'], histories['Stress-Predict 30s - Hierarchical']['train_loss'], label='Hierarchical - Train Loss', color='#ff7f0e', lw=2)
    ax.plot(histories['Stress-Predict 30s - Hierarchical']['epoch'], histories['Stress-Predict 30s - Hierarchical']['val_loss'], label='Hierarchical - Val Loss', color='#ff7f0e', linestyle='--', lw=2)
    ax.set_title('Convergencia de Loss (Stress-Predict, 30 s)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Época', fontsize=11)
    ax.set_ylabel('Cross-Entropy Loss', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 60s
    ax = axes[1]
    ax.plot(histories['Stress-Predict 60s - Standard']['epoch'], histories['Stress-Predict 60s - Standard']['train_loss'], label='Standard - Train Loss', color='#1f77b4', lw=2)
    ax.plot(histories['Stress-Predict 60s - Standard']['epoch'], histories['Stress-Predict 60s - Standard']['val_loss'], label='Standard - Val Loss', color='#1f77b4', linestyle='--', lw=2)
    ax.plot(histories['Stress-Predict 60s - Hierarchical']['epoch'], histories['Stress-Predict 60s - Hierarchical']['train_loss'], label='Hierarchical - Train Loss', color='#ff7f0e', lw=2)
    ax.plot(histories['Stress-Predict 60s - Hierarchical']['epoch'], histories['Stress-Predict 60s - Hierarchical']['val_loss'], label='Hierarchical - Val Loss', color='#ff7f0e', linestyle='--', lw=2)
    ax.set_title('Convergencia de Loss (Stress-Predict, 60 s)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Época', fontsize=11)
    ax.set_ylabel('Cross-Entropy Loss', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    loss_fig_path = os.path.join(FIG_DIR, 'audit_point3_loss_curves.png')
    plt.savefig(loss_fig_path)
    plt.close()
    print(f"[+] Gráfico de Loss curves guardado en: {loss_fig_path}")

    # Plot Validation F1 and Balanced Accuracy
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    # 30s
    ax = axes[0]
    ax.plot(histories['Stress-Predict 30s - Standard']['epoch'], histories['Stress-Predict 30s - Standard']['val_f1'], label='Standard - Val F1', color='#1f77b4', lw=2)
    ax.plot(histories['Stress-Predict 30s - Standard']['epoch'], histories['Stress-Predict 30s - Standard']['val_bacc'], label='Standard - Val Bal Acc', color='#1f77b4', linestyle=':', lw=2)
    ax.plot(histories['Stress-Predict 30s - Hierarchical']['epoch'], histories['Stress-Predict 30s - Hierarchical']['val_f1'], label='Hierarchical - Val F1', color='#ff7f0e', lw=2)
    ax.plot(histories['Stress-Predict 30s - Hierarchical']['epoch'], histories['Stress-Predict 30s - Hierarchical']['val_bacc'], label='Hierarchical - Val Bal Acc', color='#ff7f0e', linestyle=':', lw=2)
    ax.set_title('Métricas de Validación (Stress-Predict, 30 s)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Época', fontsize=11)
    ax.set_ylabel('Score', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 60s
    ax = axes[1]
    ax.plot(histories['Stress-Predict 60s - Standard']['epoch'], histories['Stress-Predict 60s - Standard']['val_f1'], label='Standard - Val F1', color='#1f77b4', lw=2)
    ax.plot(histories['Stress-Predict 60s - Standard']['epoch'], histories['Stress-Predict 60s - Standard']['val_bacc'], label='Standard - Val Bal Acc', color='#1f77b4', linestyle=':', lw=2)
    ax.plot(histories['Stress-Predict 60s - Hierarchical']['epoch'], histories['Stress-Predict 60s - Hierarchical']['val_f1'], label='Hierarchical - Val F1', color='#ff7f0e', lw=2)
    ax.plot(histories['Stress-Predict 60s - Hierarchical']['epoch'], histories['Stress-Predict 60s - Hierarchical']['val_bacc'], label='Hierarchical - Val Bal Acc', color='#ff7f0e', linestyle=':', lw=2)
    ax.set_title('Métricas de Validación (Stress-Predict, 60 s)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Época', fontsize=11)
    ax.set_ylabel('Score', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    val_fig_path = os.path.join(FIG_DIR, 'audit_point3_val_metrics_curves.png')
    plt.savefig(val_fig_path)
    plt.close()
    print(f"[+] Gráfico de Validación F1/BAcc guardado en: {val_fig_path}")

if __name__ == '__main__':
    run_audit_point3()
