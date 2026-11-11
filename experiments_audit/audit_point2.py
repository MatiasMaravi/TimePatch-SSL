"""Diagnostic Audit - Point 2: Confusion Matrices, Per-Class Breakdown, and Majority Collapse.

Evaluates:
1. Experiment 2: Stress-Predict -> WESAD
   - Standard 30s, Hierarchical 30s, Standard 60s, Hierarchical 60s
2. Experiment 1: Stress-Predict (Within-dataset LOGO / validation split)
   - Standard 30s, Hierarchical 30s, Standard 60s, Hierarchical 60s
3. Experiment 4: Combined dataset (evaluating on Stress-Predict holdout)
   - Standard 30s, Hierarchical 30s, Standard 60s, Hierarchical 60s

Computes:
- TP, TN, FP, FN
- Precision, recall, specificity, F1 per class (Class 0: non-stress, Class 1: stress)
- % predicted as stress
- Comparison against trivial "Always-Stress" baseline
- Detection and diagnosis of majority-class collapse
- Generates publication-quality confusion matrix figures and CSV summary tables.
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
    roc_auc_score, confusion_matrix, precision_recall_fscore_support
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(BASE_DIR, 'data_cache')
TABLE_DIR = os.path.join(BASE_DIR, 'results/tables')
FIG_DIR = os.path.join(BASE_DIR, 'results/figures')

os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# Add notebooks for TimeMAE if needed
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

# Model definitions
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

    def forward_ssl(self, x):
        return F.normalize(self.ssl_head(self.forward_features(x)), dim=-1)

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
    if random.random() < 0.8: x = x + 0.02 * torch.randn_like(x)
    if random.random() < 0.5: x = x * torch.empty((1, x.shape[1])).uniform_(0.9, 1.1)
    if random.random() < 0.4: x = torch.roll(x, shifts=random.randint(-5, 5), dims=0)
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
        for x1, x2 in dl:
            x1, x2 = x1.to(DEVICE), x2.to(DEVICE)
            loss = nt_xent(model.forward_ssl(x1), model.forward_ssl(x2))
            opt.zero_grad(); loss.backward(); opt.step()
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

def predict(model, X_test, batch_size=128):
    dl = DataLoader(WindowDataset(X_test), batch_size=batch_size, shuffle=False)
    model.eval()
    probs = []
    with torch.no_grad():
        for xb in dl:
            xb = xb.to(DEVICE)
            p = torch.softmax(model(xb), dim=-1)[:, 1].cpu().numpy()
            probs.append(p)
    return np.concatenate(probs)

def evaluate_predictions(y_true, p_stress):
    y_true = np.asarray(y_true).astype(int)
    p_stress = np.asarray(p_stress, dtype=float)
    y_pred = (p_stress >= 0.5).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    total = len(y_true)

    # Class 0 (Non-stress)
    prec_0 = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    rec_0 = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # Specificity
    f1_0 = 2 * prec_0 * rec_0 / (prec_0 + rec_0) if (prec_0 + rec_0) > 0 else 0.0
    supp_0 = int(tn + fp)

    # Class 1 (Stress)
    prec_1 = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec_1 = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Sensitivity
    f1_1 = 2 * prec_1 * rec_1 / (prec_1 + rec_1) if (prec_1 + rec_1) > 0 else 0.0
    supp_1 = int(fn + tp)

    acc = (tp + tn) / total
    bacc = (rec_0 + rec_1) / 2.0
    try:
        auroc = float(roc_auc_score(y_true, p_stress))
    except Exception:
        auroc = float('nan')

    pct_pred_stress = float((tp + fp) / total * 100.0)
    is_collapsed = (pct_pred_stress > 95.0 or pct_pred_stress < 5.0 or bacc == 0.5000 and rec_0 == 0.0)

    # Always stress baseline
    prev = supp_1 / total
    always_f1 = 2 * prev / (1.0 + prev) if (1.0 + prev) > 0 else 0.0

    return {
        'total': total,
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
        'accuracy': acc, 'balanced_accuracy': bacc, 'auroc': auroc,
        'prec_nonstress': prec_0, 'rec_nonstress': rec_0, 'f1_nonstress': f1_0, 'supp_nonstress': supp_0,
        'prec_stress': prec_1, 'rec_stress': rec_1, 'f1_stress': f1_1, 'supp_stress': supp_1,
        'pct_pred_stress': pct_pred_stress,
        'is_collapsed': is_collapsed,
        'always_stress_acc': prev,
        'always_stress_bacc': 0.50,
        'always_stress_f1': always_f1,
    }

def run_audit_point2():
    print("=" * 70)
    print("PUNTO 2: ANÁLISIS DE MATRICES DE CONFUSIÓN Y COLAPSO DE CLASE")
    print("=" * 70)

    configs_to_run = [
        # Exp 2: Stress-Predict -> WESAD
        {'exp': 'Exp2_Cross', 'name': 'Stress-Predict -> WESAD (30s, Standard)', 'src_ds': 'sp_30s', 'tgt_ds': 'wesad_30s', 'hierarchical': False, 'win': 30, 'patch_len': 8, 'emb_dim': 256, 'depth': 6, 'dropout': 0.2059, 'ssl_ep': 4, 'ft_ep': 7},
        {'exp': 'Exp2_Cross', 'name': 'Stress-Predict -> WESAD (30s, Hierarchical)', 'src_ds': 'sp_30s', 'tgt_ds': 'wesad_30s', 'hierarchical': True, 'win': 30, 'patch_len': 8, 'emb_dim': 256, 'depth': 6, 'dropout': 0.2059, 'ssl_ep': 4, 'ft_ep': 7},
        {'exp': 'Exp2_Cross', 'name': 'Stress-Predict -> WESAD (60s, Standard)', 'src_ds': 'sp_60s', 'tgt_ds': 'wesad_60s', 'hierarchical': False, 'win': 60, 'patch_len': 8, 'emb_dim': 256, 'depth': 4, 'dropout': 0.2005, 'ssl_ep': 7, 'ft_ep': 6},
        {'exp': 'Exp2_Cross', 'name': 'Stress-Predict -> WESAD (60s, Hierarchical)', 'src_ds': 'sp_60s', 'tgt_ds': 'wesad_60s', 'hierarchical': True, 'win': 60, 'patch_len': 8, 'emb_dim': 256, 'depth': 4, 'dropout': 0.2005, 'ssl_ep': 7, 'ft_ep': 6},

        # Exp 1: Stress-Predict Within-Dataset
        {'exp': 'Exp1_Within', 'name': 'Stress-Predict Within (30s, Standard)', 'src_ds': 'sp_30s', 'tgt_ds': 'sp_30s', 'hierarchical': False, 'win': 30, 'patch_len': 8, 'emb_dim': 256, 'depth': 6, 'dropout': 0.2059, 'ssl_ep': 4, 'ft_ep': 7},
        {'exp': 'Exp1_Within', 'name': 'Stress-Predict Within (30s, Hierarchical)', 'src_ds': 'sp_30s', 'tgt_ds': 'sp_30s', 'hierarchical': True, 'win': 30, 'patch_len': 8, 'emb_dim': 256, 'depth': 6, 'dropout': 0.2059, 'ssl_ep': 4, 'ft_ep': 7},
        {'exp': 'Exp1_Within', 'name': 'Stress-Predict Within (60s, Standard)', 'src_ds': 'sp_60s', 'tgt_ds': 'sp_60s', 'hierarchical': False, 'win': 60, 'patch_len': 8, 'emb_dim': 256, 'depth': 4, 'dropout': 0.2005, 'ssl_ep': 7, 'ft_ep': 6},
        {'exp': 'Exp1_Within', 'name': 'Stress-Predict Within (60s, Hierarchical)', 'src_ds': 'sp_60s', 'tgt_ds': 'sp_60s', 'hierarchical': True, 'win': 60, 'patch_len': 8, 'emb_dim': 256, 'depth': 4, 'dropout': 0.2005, 'ssl_ep': 7, 'ft_ep': 6},

        # Exp 4: Combined Dataset -> Test on Stress-Predict
        {'exp': 'Exp4_Combined', 'name': 'Combined D1+D2 -> Stress-Predict (30s, Standard)', 'src_ds': 'comb_30s', 'tgt_ds': 'sp_test_30s', 'hierarchical': False, 'win': 30, 'patch_len': 8, 'emb_dim': 256, 'depth': 6, 'dropout': 0.2059, 'ssl_ep': 4, 'ft_ep': 7},
        {'exp': 'Exp4_Combined', 'name': 'Combined D1+D2 -> Stress-Predict (30s, Hierarchical)', 'src_ds': 'comb_30s', 'tgt_ds': 'sp_test_30s', 'hierarchical': True, 'win': 30, 'patch_len': 8, 'emb_dim': 256, 'depth': 6, 'dropout': 0.2059, 'ssl_ep': 4, 'ft_ep': 7},
        {'exp': 'Exp4_Combined', 'name': 'Combined D1+D2 -> Stress-Predict (60s, Standard)', 'src_ds': 'comb_60s', 'tgt_ds': 'sp_test_60s', 'hierarchical': False, 'win': 60, 'patch_len': 8, 'emb_dim': 256, 'depth': 4, 'dropout': 0.2005, 'ssl_ep': 7, 'ft_ep': 6},
        {'exp': 'Exp4_Combined', 'name': 'Combined D1+D2 -> Stress-Predict (60s, Hierarchical)', 'src_ds': 'comb_60s', 'tgt_ds': 'sp_test_60s', 'hierarchical': True, 'win': 60, 'patch_len': 8, 'emb_dim': 256, 'depth': 4, 'dropout': 0.2005, 'ssl_ep': 7, 'ft_ep': 6},
    ]

    # Pre-load datasets
    data_cache = {
        'wesad_30s': np.load(os.path.join(CACHE_DIR, 'wesad_30s.npz')),
        'wesad_60s': np.load(os.path.join(CACHE_DIR, 'wesad_60s.npz')),
        'sp_30s': np.load(os.path.join(CACHE_DIR, 'stress_predict_30s.npz')),
        'sp_60s': np.load(os.path.join(CACHE_DIR, 'stress_predict_60s.npz')),
    }

    # Splits for Stress-Predict holdout
    with open(os.path.join(BASE_DIR, 'splits/stress_predict_subject_splits.json'), 'r') as f:
        sp_split_info = json.load(f)
    sp_test_ids = [int(s.lstrip('S')) for s in sp_split_info['test_participant_ids']]
    sp_dev_ids = [int(s.lstrip('S')) for s in sp_split_info['development_participant_ids']]

    results_rows = []
    always_stress_rows = []
    cms_dict = {}

    for cfg in configs_to_run:
        set_seed(SEED)
        print(f"\n[*] Ejecutando: {cfg['name']}...")

        # Prepare train / test data according to experiment
        if cfg['exp'] == 'Exp2_Cross':
            src = data_cache[cfg['src_ds']]
            tgt = data_cache[cfg['tgt_ds']]
            X_train_full, y_train_full = src['X'], src['y']
            X_test, y_test = tgt['X'], tgt['y']
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_train_full, y_train_full, test_size=0.15, random_state=SEED, stratify=y_train_full
            )
        elif cfg['exp'] == 'Exp1_Within':
            src = data_cache[cfg['src_ds']]
            groups = src['groups']
            # Holdout test subjects (7 subjects)
            test_mask = np.isin(groups, sp_test_ids)
            dev_mask = ~test_mask
            X_dev, y_dev = src['X'][dev_mask], src['y'][dev_mask]
            X_test, y_test = src['X'][test_mask], src['y'][test_mask]
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_dev, y_dev, test_size=0.15, random_state=SEED, stratify=y_dev
            )
        elif cfg['exp'] == 'Exp4_Combined':
            win = cfg['win']
            src_sp = data_cache[f'sp_{win}s']
            src_wesad = data_cache[f'wesad_{win}s']

            # Test participants holdout for SP
            sp_test_mask = np.isin(src_sp['groups'], sp_test_ids)
            sp_dev_mask = ~sp_test_mask
            X_sp_dev, y_sp_dev = src_sp['X'][sp_dev_mask], src_sp['y'][sp_dev_mask]
            X_test, y_test = src_sp['X'][sp_test_mask], src_sp['y'][sp_test_mask]

            # Pool 80% of WESAD
            w_tr_idx, w_val_idx = train_test_split(
                np.arange(len(src_wesad['X'])), test_size=0.20, random_state=SEED, stratify=src_wesad['y']
            )
            X_w_dev, y_w_dev = src_wesad['X'][w_tr_idx], src_wesad['y'][w_tr_idx]

            # Combine
            X_comb = np.concatenate([X_sp_dev, X_w_dev], axis=0)
            y_comb = np.concatenate([y_sp_dev, y_w_dev], axis=0)
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_comb, y_comb, test_size=0.15, random_state=SEED, stratify=y_comb
            )

        # Build model
        model = TimePatchTransformer(
            in_ch=3, patch_len=cfg['patch_len'], emb_dim=cfg['emb_dim'],
            depth=cfg['depth'], heads=4, dropout=cfg['dropout'],
            num_classes=2, hierarchical=cfg['hierarchical']
        )
        model = pretrain_ssl(model, X_tr, epochs=cfg['ssl_ep'], batch_size=16)
        model = finetune(model, X_tr, y_tr, X_val, y_val, epochs=cfg['ft_ep'], batch_size=16)

        p_stress = predict(model, X_test, batch_size=128)
        metrics = evaluate_predictions(y_test, p_stress)
        cms_dict[cfg['name']] = np.array([[metrics['tn'], metrics['fp']], [metrics['fn'], metrics['tp']]])

        print(f"    - Accuracy: {metrics['accuracy']:.4f} | Balanced Acc: {metrics['balanced_accuracy']:.4f} | AUROC: {metrics['auroc']:.4f}")
        print(f"    - Stress: Prec={metrics['prec_stress']:.4f}, Rec={metrics['rec_stress']:.4f}, F1={metrics['f1_stress']:.4f}")
        print(f"    - Non-Stress: Prec={metrics['prec_nonstress']:.4f}, Rec={metrics['rec_nonstress']:.4f}, F1={metrics['f1_nonstress']:.4f}")
        print(f"    - % Pred Stress: {metrics['pct_pred_stress']:.2f}% | Colapso: {'SI (Mayoritaria)' if metrics['is_collapsed'] else 'NO'}")

        results_rows.append({
            'Experimento': cfg['exp'],
            'Configuracion': cfg['name'],
            'Ventana_s': cfg['win'],
            'Arquitectura': 'Hierarchical' if cfg['hierarchical'] else 'Standard',
            'Total_Muestras': metrics['total'],
            'TN': metrics['tn'], 'FP': metrics['fp'], 'FN': metrics['fn'], 'TP': metrics['tp'],
            'Accuracy': metrics['accuracy'],
            'Balanced_Accuracy': metrics['balanced_accuracy'],
            'AUROC': metrics['auroc'],
            'Precision_Stress': metrics['prec_stress'],
            'Recall_Stress': metrics['rec_stress'],
            'F1_Stress': metrics['f1_stress'],
            'Precision_NonStress': metrics['prec_nonstress'],
            'Recall_NonStress_Specificity': metrics['rec_nonstress'],
            'F1_NonStress': metrics['f1_nonstress'],
            'Pct_Pred_Stress': metrics['pct_pred_stress'],
            'Colapso_Clase_Mayoritaria': metrics['is_collapsed'],
        })

        always_stress_rows.append({
            'Configuracion': cfg['name'],
            'Modelo_Accuracy': metrics['accuracy'],
            'Baseline_AlwaysStress_Accuracy': metrics['always_stress_acc'],
            'Delta_Accuracy': metrics['accuracy'] - metrics['always_stress_acc'],
            'Modelo_Balanced_Accuracy': metrics['balanced_accuracy'],
            'Baseline_AlwaysStress_Balanced_Acc': 0.50,
            'Delta_Balanced_Acc': metrics['balanced_accuracy'] - 0.50,
            'Modelo_F1_Stress': metrics['f1_stress'],
            'Baseline_AlwaysStress_F1': metrics['always_stress_f1'],
            'Delta_F1': metrics['f1_stress'] - metrics['always_stress_f1'],
            'Colapso_Detectado': metrics['is_collapsed']
        })

        # Incremental CSV save
        pd.DataFrame(results_rows).to_csv(os.path.join(TABLE_DIR, 'audit_point2_per_class_metrics.csv'), index=False)
        pd.DataFrame(always_stress_rows).to_csv(os.path.join(TABLE_DIR, 'audit_point2_always_stress_comparison.csv'), index=False)

    # Save CSV tables
    df_results = pd.DataFrame(results_rows)
    csv_metrics_path = os.path.join(TABLE_DIR, 'audit_point2_per_class_metrics.csv')
    df_results.to_csv(csv_metrics_path, index=False)
    print(f"\n[+] Métricas por clase guardadas en: {csv_metrics_path}")

    df_always = pd.DataFrame(always_stress_rows)
    csv_always_path = os.path.join(TABLE_DIR, 'audit_point2_always_stress_comparison.csv')
    df_always.to_csv(csv_always_path, index=False)
    print(f"[+] Comparativa con Always-Stress guardada en: {csv_always_path}")

    # Generate Figures: Grid of Confusion Matrices
    fig, axes = plt.subplots(3, 4, figsize=(18, 13), dpi=300)
    fig.suptitle('Confusion Matrices Across 12 Experimental Configurations (Point 2 Audit)', fontsize=15, fontweight='bold', y=0.995)
    axes = axes.flatten()

    for idx, (title, cm) in enumerate(cms_dict.items()):
        ax = axes[idx]
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title(title.replace(' (', '\n('), fontsize=10, fontweight='bold')
        tick_marks = np.arange(2)
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(['Non-stress (0)', 'Stress (1)'], fontsize=9)
        ax.set_yticklabels(['Non-stress (0)', 'Stress (1)'], fontsize=9)
        ax.set_ylabel('True Label', fontsize=9)
        ax.set_xlabel('Predicted Label', fontsize=9)

        # Text annotations
        thresh = cm.max() / 2.
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}\n({cm[i, j]/cm[i].sum()*100:.1f}%)",
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=9, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    cm_fig_path = os.path.join(FIG_DIR, 'audit_point2_confusion_matrices.png')
    plt.savefig(cm_fig_path)
    plt.close()
    print(f"[+] Figura de matrices de confusión guardada en: {cm_fig_path}")

if __name__ == '__main__':
    run_audit_point2()
