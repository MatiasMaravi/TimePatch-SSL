"""Option A Multi-seed Experiments:
1. Re-run Experiment 1 (Stress-Predict LOGO Supervised) across 3 seeds (42, 123, 456)
   for Standard and Hierarchical encoders at 30s and 60s.
2. Re-run Experiment 4 (Combined Dataset SSL + Fine-tuning) across 3 seeds (42, 123, 456)
   for Standard and Hierarchical encoders at 30s and 60s.
Saves all participant-level and summary tables with Mean +- Std.
"""
import os
import sys
import time
import json
import random
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split

BASE_DIR = '/home/leoisidro/CICLOS/X/PFC_II/TimePatch-SSL'
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)
nb_dir = os.path.join(BASE_DIR, 'notebooks')
if nb_dir not in sys.path:
    sys.path.append(nb_dir)
if os.path.join(nb_dir, 'TimeMAE') not in sys.path:
    sys.path.append(os.path.join(nb_dir, 'TimeMAE'))

from experiments_audit.audit_point2 import TimePatchTransformer, WindowDataset
from src.ssl import nt_xent

CACHE_DIR = os.path.join(BASE_DIR, 'data_cache')
OUT_DIR = os.path.join(BASE_DIR, 'results/tables/multiseed_audit')
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEEDS = [42, 123, 456]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_metrics(y_true, p_stress):
    y_true = np.asarray(y_true).astype(int)
    p_stress = np.asarray(p_stress, dtype=float)
    y_pred = (p_stress >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    bacc = balanced_accuracy_score(y_true, y_pred)
    try:
        auroc = roc_auc_score(y_true, p_stress)
    except Exception:
        auroc = float('nan')
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    pct_stress = 100.0 * np.mean(y_pred == 1)
    return {
        'accuracy': float(acc),
        'f1': float(f1),
        'balanced_accuracy': float(bacc),
        'auroc': float(auroc),
        'specificity': float(spec),
        'sensitivity': float(sens),
        'pct_pred_stress': float(pct_stress),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
    }

def train_and_eval_supervised(X_tr, y_tr, X_val, y_val, X_te, y_te, cfg, seed):
    set_seed(seed)
    model = TimePatchTransformer(
        in_ch=3, patch_len=cfg['patch_len'], emb_dim=cfg['emb_dim'],
        depth=cfg['depth'], heads=4, dropout=cfg['dropout'],
        num_classes=2, hierarchical=cfg['hierarchical']
    ).to(DEVICE)
    
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    dl_tr = DataLoader(WindowDataset(X_tr, y_tr), batch_size=cfg['batch_size'], shuffle=True)
    dl_val = DataLoader(WindowDataset(X_val, y_val), batch_size=64, shuffle=False)
    dl_te = DataLoader(WindowDataset(X_te, y_te), batch_size=64, shuffle=False)
    
    best_score = -1.0
    best_weights = None
    
    for ep in range(cfg['epochs']):
        model.train()
        for xb, yb in dl_tr:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            loss = crit(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            
        model.eval()
        p_val, y_val_all = [], []
        with torch.no_grad():
            for xb, yb in dl_val:
                p_val.append(torch.softmax(model(xb.to(DEVICE)), -1)[:, 1].cpu().numpy())
                y_val_all.append(yb.numpy())
        p_val = np.concatenate(p_val)
        y_val_all = np.concatenate(y_val_all)
        
        # Macro F1 + Balanced Accuracy validation selection (prevents majority collapse!)
        val_macro_f1 = f1_score(y_val_all, (p_val >= 0.5).astype(int), average='macro')
        val_bacc = balanced_accuracy_score(y_val_all, (p_val >= 0.5).astype(int))
        score = (val_macro_f1 + val_bacc) / 2.0
        
        if score > best_score:
            best_score = score
            best_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            
    if best_weights is not None:
        model.load_state_dict(best_weights)
        
    model.eval()
    p_te = []
    with torch.no_grad():
        for xb, _ in dl_te:
            p_te.append(torch.softmax(model(xb.to(DEVICE)), -1)[:, 1].cpu().numpy())
    p_te = np.concatenate(p_te)
    return p_te, compute_metrics(y_te, p_te)


def pretrain_contrastive(model, X_unlab, cfg, seed):
    from src.augmentations import SSLPairDataset
    set_seed(seed)
    dl = DataLoader(SSLPairDataset(X_unlab), batch_size=cfg.get('ssl_batch_size', 32), shuffle=True, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    model.train()
    for ep in range(cfg.get('ssl_epochs', 6)):
        for a, b in dl:
            a, b = a.to(DEVICE), b.to(DEVICE)
            z1 = model.forward_ssl(a)
            z2 = model.forward_ssl(b)
            loss = nt_xent(z1, z2, temp=0.2)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


# ==============================================================================
# PART 1: Stress-Predict LOGO Supervised (30s and 60s)
# ==============================================================================
def run_part1_logo():
    print("\n" + "=" * 80)
    print("PARTE 1: RE-EJECUCIÓN DE LOGO SUPERVISADO EN STRESS-PREDICT (MULTI-SEMILLA)")
    print("=" * 80)
    
    configs = [
        {'win': 30, 'arch': 'Standard', 'hierarchical': False, 'emb_dim': 256, 'depth': 6, 'dropout': 0.2059, 'patch_len': 8, 'epochs': 8, 'lr': 2e-4, 'batch_size': 32},
        {'win': 30, 'arch': 'Hierarchical', 'hierarchical': True, 'emb_dim': 256, 'depth': 6, 'dropout': 0.2059, 'patch_len': 8, 'epochs': 8, 'lr': 2e-4, 'batch_size': 32},
        {'win': 60, 'arch': 'Standard', 'hierarchical': False, 'emb_dim': 256, 'depth': 4, 'dropout': 0.2005, 'patch_len': 8, 'epochs': 8, 'lr': 2e-4, 'batch_size': 32},
        {'win': 60, 'arch': 'Hierarchical', 'hierarchical': True, 'emb_dim': 256, 'depth': 4, 'dropout': 0.2005, 'patch_len': 8, 'epochs': 8, 'lr': 2e-4, 'batch_size': 32},
    ]
    
    participant_results = []
    summary_results = []
    
    for cfg in configs:
        win = cfg['win']
        arch = cfg['arch']
        print(f"\n[+] Configuración: Stress-Predict {win}s | Arquitectura: {arch}")
        data = np.load(os.path.join(CACHE_DIR, f'stress_predict_{win}s.npz'))
        X, y, groups = data['X'], data['y'], data['groups']
        subjects = np.unique(groups)
        
        for seed in SEEDS:
            t0 = time.time()
            seed_metrics = []
            
            for held_out in subjects:
                te_mask = (groups == held_out)
                tr_mask = ~te_mask
                X_train, y_train = X[tr_mask], y[tr_mask]
                groups_tr = groups[tr_mask]
                X_test, y_test = X[te_mask], y[te_mask]
                
                # Split 15% validation from development participants
                dev_subjs = np.unique(groups_tr)
                tr_s, val_s = train_test_split(dev_subjs, test_size=0.15, random_state=seed)
                tr2_idx = np.isin(groups_tr, tr_s)
                val_idx = np.isin(groups_tr, val_s)
                
                X_tr, y_tr = X_train[tr2_idx], y_train[tr2_idx]
                X_val, y_val = X_train[val_idx], y_train[val_idx]
                
                _, m = train_and_eval_supervised(X_tr, y_tr, X_val, y_val, X_test, y_test, cfg, seed)
                m.update({
                    'dataset': 'Stress-Predict',
                    'window': f'{win}s',
                    'encoder': arch,
                    'seed': seed,
                    'participant': int(held_out),
                    'n_test_samples': len(y_test)
                })
                participant_results.append(m)
                seed_metrics.append(m)
                
            # Aggregate across participants for this seed
            df_sm = pd.DataFrame(seed_metrics)
            acc_m = df_sm['accuracy'].mean()
            f1_m = df_sm['f1'].mean()
            bacc_m = df_sm['balanced_accuracy'].mean()
            auroc_m = df_sm['auroc'].mean()
            el = time.time() - t0
            print(f"    Seed {seed:3d} (33 folds in {el:.1f}s) -> Acc: {acc_m:.4f} | F1: {f1_m:.4f} | BalAcc: {bacc_m:.4f} | AUROC: {auroc_m:.4f}")
            
    # Save participant level
    df_part = pd.DataFrame(participant_results)
    part_csv = os.path.join(OUT_DIR, 'part1_stresspredict_logo_participants.csv')
    df_part.to_csv(part_csv, index=False)
    print(f"\n[+] Resultados por participante guardados en: {part_csv}")
    
    # Summary Table across seeds: Mean +- Std across seeds of participant means
    summary_rows = []
    for (win, arch), group in df_part.groupby(['window', 'encoder']):
        # Compute mean per seed first
        seed_means = group.groupby('seed')[['accuracy', 'f1', 'balanced_accuracy', 'auroc']].mean()
        summary_rows.append({
            'Dataset': 'Stress-Predict',
            'Window': win,
            'Encoder': arch,
            'Accuracy_Mean': seed_means['accuracy'].mean(),
            'Accuracy_Std': seed_means['accuracy'].std(),
            'F1_Mean': seed_means['f1'].mean(),
            'F1_Std': seed_means['f1'].std(),
            'Bal_Acc_Mean': seed_means['balanced_accuracy'].mean(),
            'Bal_Acc_Std': seed_means['balanced_accuracy'].std(),
            'AUROC_Mean': seed_means['auroc'].mean(),
            'AUROC_Std': seed_means['auroc'].std(),
            'N_Participants': 33,
            'N_Seeds': len(SEEDS)
        })
    df_summary = pd.DataFrame(summary_rows)
    sum_csv = os.path.join(OUT_DIR, 'part1_stresspredict_logo_summary.csv')
    df_summary.to_csv(sum_csv, index=False)
    print(f"[+] Resumen multi-semilla guardado en: {sum_csv}")
    print("\n" + df_summary.to_string(index=False))
    return df_summary


# ==============================================================================
# PART 2: Combined-Dataset Subject-Independent (Exp 4)
# ==============================================================================
def run_part2_combined():
    print("\n" + "=" * 80)
    print("PARTE 2: RE-EJECUCIÓN DE EXP 4 COMBINADO (MULTI-SEMILLA CON SSL + FT)")
    print("=" * 80)
    
    # Load canonical split from splits/experiment4_split.json
    with open(os.path.join(BASE_DIR, 'splits/experiment4_split.json'), 'r') as f:
        split_info = json.load(f)
    wesad_test_subjs = [int(s.lstrip('S')) for s in split_info['datasets']['WESAD']['test_participant_ids']]
    sp_test_subjs = [int(s.lstrip('S')) for s in split_info['datasets']['Stress-Predict']['test_participant_ids']]
    
    print(f"Test Subjects WESAD (3): {wesad_test_subjs}")
    print(f"Test Subjects Stress-Predict (7): {sp_test_subjs}")
    
    configs = [
        {'win': 30, 'arch': 'Standard', 'hierarchical': False, 'emb_dim': 256, 'depth': 6, 'dropout': 0.2059, 'patch_len': 8, 'ssl_ep': 6, 'ft_ep': 8, 'lr': 2e-4, 'batch_size': 32},
        {'win': 30, 'arch': 'Hierarchical', 'hierarchical': True, 'emb_dim': 256, 'depth': 6, 'dropout': 0.2059, 'patch_len': 8, 'ssl_ep': 6, 'ft_ep': 8, 'lr': 2e-4, 'batch_size': 32},
        {'win': 60, 'arch': 'Standard', 'hierarchical': False, 'emb_dim': 256, 'depth': 4, 'dropout': 0.2005, 'patch_len': 8, 'ssl_ep': 6, 'ft_ep': 8, 'lr': 2e-4, 'batch_size': 32},
        {'win': 60, 'arch': 'Hierarchical', 'hierarchical': True, 'emb_dim': 256, 'depth': 4, 'dropout': 0.2005, 'patch_len': 8, 'ssl_ep': 6, 'ft_ep': 8, 'lr': 2e-4, 'batch_size': 32},
    ]
    
    combined_results = []
    
    for cfg in configs:
        win = cfg['win']
        arch = cfg['arch']
        print(f"\n[+] Configuración Combinado: {win}s | Arquitectura: {arch}")
        
        d_w = np.load(os.path.join(CACHE_DIR, f'wesad_{win}s.npz'))
        d_sp = np.load(os.path.join(CACHE_DIR, f'stress_predict_{win}s.npz'))
        
        # WESAD split
        w_test_mask = np.isin(d_w['groups'], wesad_test_subjs)
        w_dev_mask = ~w_test_mask
        X_w_dev, y_w_dev = d_w['X'][w_dev_mask], d_w['y'][w_dev_mask]
        X_w_te, y_w_te = d_w['X'][w_test_mask], d_w['y'][w_test_mask]
        
        # Stress-Predict split
        sp_test_mask = np.isin(d_sp['groups'], sp_test_subjs)
        sp_dev_mask = ~sp_test_mask
        X_sp_dev, y_sp_dev = d_sp['X'][sp_dev_mask], d_sp['y'][sp_dev_mask]
        X_sp_te, y_sp_te = d_sp['X'][sp_test_mask], d_sp['y'][sp_test_mask]
        
        # Pool development
        X_dev = np.concatenate([X_w_dev, X_sp_dev], axis=0)
        y_dev = np.concatenate([y_w_dev, y_sp_dev], axis=0)
        groups_dev = np.concatenate([
            np.array([f'WESAD_{s}' for s in d_w['groups'][w_dev_mask]]),
            np.array([f'SP_{s}' for s in d_sp['groups'][sp_dev_mask]])
        ])
        
        for seed in SEEDS:
            t0 = time.time()
            set_seed(seed)
            
            # Split 15% validation by participant from development pool
            dev_subjs = np.unique(groups_dev)
            tr_s, val_s = train_test_split(dev_subjs, test_size=0.15, random_state=seed)
            tr_idx = np.isin(groups_dev, tr_s)
            val_idx = np.isin(groups_dev, val_s)
            
            X_tr, y_tr = X_dev[tr_idx], y_dev[tr_idx]
            X_val, y_val = X_dev[val_idx], y_dev[val_idx]
            
            # 1. Instantiate model
            model = TimePatchTransformer(
                in_ch=3, patch_len=cfg['patch_len'], emb_dim=cfg['emb_dim'],
                depth=cfg['depth'], heads=4, dropout=cfg['dropout'],
                num_classes=2, hierarchical=cfg['hierarchical']
            ).to(DEVICE)
            
            # 2. SSL Pretraining on pooled unlabeled development data
            model = pretrain_contrastive(model, X_dev, cfg, seed)
            
            # 3. Supervised fine-tuning with validation selection
            opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=1e-4)
            crit = nn.CrossEntropyLoss()
            dl_tr = DataLoader(WindowDataset(X_tr, y_tr), batch_size=cfg['batch_size'], shuffle=True)
            dl_val = DataLoader(WindowDataset(X_val, y_val), batch_size=64, shuffle=False)
            
            best_score = -1.0
            best_weights = None
            for ep in range(cfg['ft_ep']):
                model.train()
                for xb, yb in dl_tr:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    loss = crit(model(xb), yb)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                model.eval()
                p_val, y_val_all = [], []
                with torch.no_grad():
                    for xb, yb in dl_val:
                        p_val.append(torch.softmax(model(xb.to(DEVICE)), -1)[:, 1].cpu().numpy())
                        y_val_all.append(yb.numpy())
                p_val = np.concatenate(p_val)
                y_val_all = np.concatenate(y_val_all)
                val_macro_f1 = f1_score(y_val_all, (p_val >= 0.5).astype(int), average='macro')
                val_bacc = balanced_accuracy_score(y_val_all, (p_val >= 0.5).astype(int))
                score = (val_macro_f1 + val_bacc) / 2.0
                if score > best_score:
                    best_score = score
                    best_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    
            if best_weights is not None:
                model.load_state_dict(best_weights)
                
            model.eval()
            
            # 4. Evaluate on WESAD held-out test
            p_w_te = []
            dl_w_te = DataLoader(WindowDataset(X_w_te, y_w_te), batch_size=64, shuffle=False)
            with torch.no_grad():
                for xb, _ in dl_w_te:
                    p_w_te.append(torch.softmax(model(xb.to(DEVICE)), -1)[:, 1].cpu().numpy())
            p_w_te = np.concatenate(p_w_te)
            m_w = compute_metrics(y_w_te, p_w_te)
            m_w.update({
                'window': f'{win}s',
                'encoder': arch,
                'seed': seed,
                'target_dataset': 'WESAD_Heldout_20%',
                'n_test_samples': len(y_w_te)
            })
            combined_results.append(m_w)
            
            # 5. Evaluate on Stress-Predict held-out test
            p_sp_te = []
            dl_sp_te = DataLoader(WindowDataset(X_sp_te, y_sp_te), batch_size=64, shuffle=False)
            with torch.no_grad():
                for xb, _ in dl_sp_te:
                    p_sp_te.append(torch.softmax(model(xb.to(DEVICE)), -1)[:, 1].cpu().numpy())
            p_sp_te = np.concatenate(p_sp_te)
            m_sp = compute_metrics(y_sp_te, p_sp_te)
            m_sp.update({
                'window': f'{win}s',
                'encoder': arch,
                'seed': seed,
                'target_dataset': 'Stress-Predict_Heldout_20%',
                'n_test_samples': len(y_sp_te)
            })
            combined_results.append(m_sp)
            
            el = time.time() - t0
            print(f"    Seed {seed:3d} in {el:.1f}s | WESAD: Acc={m_w['accuracy']:.4f}, AUROC={m_w['auroc']:.4f} | SP: Acc={m_sp['accuracy']:.4f}, AUROC={m_sp['auroc']:.4f}")
            
    df_comb = pd.DataFrame(combined_results)
    raw_csv = os.path.join(OUT_DIR, 'part2_combined_raw_seeds.csv')
    df_comb.to_csv(raw_csv, index=False)
    print(f"\n[+] Resultados brutos de Exp 4 guardados en: {raw_csv}")
    
    # Summary Table with Mean +- Std across seeds
    summary_comb = []
    for (tgt, win, arch), group in df_comb.groupby(['target_dataset', 'window', 'encoder']):
        summary_comb.append({
            'Target_Pool': tgt,
            'Window': win,
            'Encoder': arch,
            'Accuracy_Mean': group['accuracy'].mean(),
            'Accuracy_Std': group['accuracy'].std(),
            'F1_Mean': group['f1'].mean(),
            'F1_Std': group['f1'].std(),
            'Bal_Acc_Mean': group['balanced_accuracy'].mean(),
            'Bal_Acc_Std': group['balanced_accuracy'].std(),
            'AUROC_Mean': group['auroc'].mean(),
            'AUROC_Std': group['auroc'].std(),
            'N_Seeds': len(SEEDS)
        })
    df_comb_summary = pd.DataFrame(summary_comb)
    sum_csv = os.path.join(OUT_DIR, 'part2_combined_summary.csv')
    df_comb_summary.to_csv(sum_csv, index=False)
    print(f"[+] Resumen multi-semilla de Exp 4 guardado en: {sum_csv}")
    print("\n" + df_comb_summary.to_string(index=False))
    return df_comb_summary


if __name__ == '__main__':
    t_start = time.time()
    df_logo = run_part1_logo()
    df_comb = run_part2_combined()
    total_time = time.time() - t_start
    print(f"\n{'='*80}\nEXPERIMENTOS COMPLETADOS CON ÉXITO EN {total_time/60:.2f} MINUTOS\n{'='*80}")
