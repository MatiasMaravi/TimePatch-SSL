"""Experiment runner for capacity-matched ablation study (Reviewer Point 5).

Compares:
1. Standard (base capacity, linear patch embed)
2. Hierarchical (local Transformer patch embed)
3. Standard-Matched (matched capacity to Hierarchical, linear patch embed)

Under strict Leave-One-Subject-Out (LOSO) cross-validation.
Self-contained execution to avoid dependency on unmodified/legacy source modules.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import random
import sys
from pathlib import Path

# --- Autodetectar la raíz del proyecto y TimeMAE en sys.path ---
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TIMEMAE_PATH = Path("C:/Users/USUARIO/Desktop/PFC/PFC2")
if TIMEMAE_PATH.exists() and str(TIMEMAE_PATH) not in sys.path:
    sys.path.insert(0, str(TIMEMAE_PATH))
# -------------------------------------------------------------

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.preprocessing import TrainingOnlyRobustScaler, resample_1d, segment_windows
from src.evaluate import evaluate_probs
from scipy.stats import t as _t_dist
from ablation_matched_capacity.models import build_model


# --- WESAD Dataset Loader ---
WESAD_USE_LABELS = {1, 2, 3}
WESAD_STRESS_LABEL = 2


def robust_norm(x, method="median"):
    """Match notebook robust_norm per-signal/per-participant normalization."""
    x = np.asarray(x, dtype=np.float32)
    if np.isnan(x).any():
        nan_val = np.nanmedian(x) if method == "median" else np.nanmean(x)
        x = np.nan_to_num(x, nan=nan_val if np.isnan(x).any() else 0.0)
    med = np.median(x) if method == "median" else np.mean(x)
    iqr = np.percentile(x, 75) - np.percentile(x, 25)
    scale = iqr if iqr > 1e-6 else (np.std(x) + 1e-6)
    return ((x - med) / scale).astype(np.float32)


def load_wesad_subject_raw(pkl_path: str | Path, target_fs: int = 4, norm_method: str = "median"):
    """Load, resample, and normalize one WESAD participant matching notebook."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    wrist = data["signal"]["wrist"]
    labels = np.asarray(data["label"]).astype(int).squeeze()
    bvp = resample_1d(np.asarray(wrist["BVP"]).squeeze(), 64, target_fs)
    eda = resample_1d(np.asarray(wrist["EDA"]).squeeze(), 4, target_fs)
    temp = resample_1d(np.asarray(wrist["TEMP"]).squeeze(), 4, target_fs)
    labels = resample_1d(labels, 700, target_fs).round().astype(int)
    L = min(map(len, (bvp, eda, temp, labels)))
    X = np.stack([
        robust_norm(bvp[:L], method=norm_method),
        robust_norm(eda[:L], method=norm_method),
        robust_norm(temp[:L], method=norm_method),
    ], axis=1).astype(np.float32)
    labels = labels[:L]
    keep = np.isin(labels, list(WESAD_USE_LABELS))
    return X[keep], np.where(labels[keep] == WESAD_STRESS_LABEL, 1, 0).astype(np.int64)


def load_all_wesad(root: str | Path, window_seconds: int = 60, stride_seconds: int | None = None, target_fs: int = 4, norm_method: str = "median"):
    """Load all WESAD participants into windowed arrays with 50% default overlap and per-subject normalization."""
    if stride_seconds is None:
        stride_seconds = window_seconds // 2

    Xs, ys, groups = [], [], []
    root = Path(root)
    for subj in sorted(d for d in os.listdir(root) if d.startswith("S")):
        path = root / subj / f"{subj}.pkl"
        if not path.exists():
            continue
        X, y = load_wesad_subject_raw(path, target_fs, norm_method=norm_method)
        Xw, yw = segment_windows(X, y, window_seconds, stride_seconds, target_fs)
        if len(Xw):
            Xs.append(Xw)
            ys.append(yw)
            groups.append(np.full(len(yw), int(subj[1:]), dtype=np.int64))
    if not Xs:
        raise ValueError(f"No WESAD participant data found under {root!r}.")
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(groups)


# --- PyTorch Dataset & Training Utilities ---
class WindowDataset(Dataset):
    """In-memory PyTorch dataset for physiological window tensors."""
    def __init__(self, X, y=None):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = None if y is None else torch.as_tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx] if self.y is None else (self.X[idx], self.y[idx])


def predict_probs(model, X, device, batch_size=128):
    """Compute softmax probabilities for the stress class (class 1)."""
    dl = DataLoader(WindowDataset(X), batch_size=batch_size, shuffle=False)
    model.eval()
    out = []
    with torch.no_grad():
        for xb in dl:
            probs = torch.softmax(model(xb.to(device)), dim=1)[:, 1].cpu().numpy()
            out.append(probs)
    return np.concatenate(out) if out else np.array([], dtype=np.float32)


def finetune(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    device,
    epochs=6,
    batch_size=32,
    lr=2e-4,
    weight_decay=1e-4,
    verbose=False,
):
    """End-to-end supervised training with validation-F1 checkpoint selection."""
    dl = DataLoader(WindowDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    best_f1 = -np.inf
    best = None

    for e in range(epochs):
        model.train()
        losses = []
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            loss = loss_fn(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        val_probs = predict_probs(model, X_val, device, batch_size=batch_size)
        m = evaluate_probs(y_val, val_probs, 0.5)
        if verbose:
            print(f"    Epoch {e+1}/{epochs} | Loss: {np.mean(losses):.4f} | Val F1: {m['f1']:.4f}")

        if m["f1"] >= best_f1:
            best_f1 = m["f1"]
            best = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})

    if best is None:
        best = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})

    model.load_state_dict(best)
    return model


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def participant_train_val(groups, dev_ids, val_fraction=0.15, seed=42):
    """Splits dev_ids into train and validation participant sets without data leakage."""
    rng = np.random.RandomState(seed)
    shuffled = rng.permutation(dev_ids)
    n_val = max(1, int(round(len(dev_ids) * val_fraction)))
    val_ids = set(shuffled[:n_val])
    fit_ids = set(shuffled[n_val:])
    fit_idx = np.flatnonzero(np.isin(groups, list(fit_ids)))
    val_idx = np.flatnonzero(np.isin(groups, list(val_ids)))
    return fit_idx, val_idx, sorted(fit_ids), sorted(val_ids)


def run_loso_ablation(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    device: str = "cpu",
    seed: int = 42,
    model_kwargs: dict | None = None,
    train_kwargs: dict | None = None,
    variants: tuple[str, ...] = ("standard", "hierarchical", "standard_matched"),
    max_participants: int | None = None,
) -> dict:
    """Run LOSO evaluation comparing all specified model variants."""
    model_kwargs = model_kwargs or {}
    train_kwargs = train_kwargs or {}

    participants = np.unique(groups)
    if max_participants is not None:
        participants = participants[:max_participants]

    results = {v: [] for v in variants}

    print(f"\n=======================================================")
    print(f"Starting LOSO Ablation on {len(participants)} participants")
    print(f"Variants being evaluated: {list(variants)}")
    print(f"Model parameters: {model_kwargs}")
    print(f"Device: {device}")
    print(f"=======================================================\n")

    for idx, heldout in enumerate(participants, 1):
        print(f"[{idx}/{len(participants)}] Held-out subject: {heldout}")
        test_idx = np.flatnonzero(groups == heldout)
        dev_ids = np.unique(groups[groups != heldout])
        fit_idx, val_idx, fit_ids, val_ids = participant_train_val(groups, dev_ids, 0.15, seed)

        # Matches notebook: X is already normalized per-participant by robust_norm
        Xfit = X[fit_idx]
        Xval = X[val_idx]
        Xtest = X[test_idx]

        for variant in variants:
            set_seed(seed)
            model = build_model(variant=variant, **model_kwargs)
            model = finetune(model, Xfit, y[fit_idx], Xval, y[val_idx], device, **train_kwargs)
            probs = predict_probs(model, Xtest, device)
            metrics = evaluate_probs(y[test_idx], probs, 0.5)
            metrics.update({
                "participant": str(heldout),
                "variant": variant,
                "n_windows": int(len(test_idx)),
            })
            results[variant].append(metrics)
            print(f"  -> {variant:16s} | Acc: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f} | BAcc: {metrics['balanced_accuracy']:.4f} | AUROC: {metrics['auroc']:.4f}")

    return results


def summarize_results(results: dict, alpha: float = 0.05) -> pd.DataFrame:
    """Computes mean, sample std (ddof=1), and 95% CI across participants matching notebook summarize()."""
    rows = []
    metric_keys = ("accuracy", "f1", "balanced_accuracy", "auroc")
    for variant, participant_metrics in results.items():
        row = {"variant": variant}
        print(f"\n--- Summary: {variant.upper()} ---")
        for k in metric_keys:
            vals = np.array([m[k] for m in participant_metrics if not np.isnan(m[k])], dtype=float)
            n = len(vals)
            if n > 0:
                mean_val = float(np.mean(vals))
                std_val = float(np.std(vals, ddof=1)) if n > 1 else 0.0
                sem = std_val / np.sqrt(n) if n > 1 else 0.0
                t_crit = float(_t_dist.ppf(1 - alpha / 2, df=max(n - 1, 1)))
                ci_half = t_crit * sem

                row[f"{k}_mean"] = round(mean_val, 4)
                row[f"{k}_std"] = round(std_val, 4)
                row[f"{k}_ci95_low"] = round(float(mean_val - ci_half), 4)
                row[f"{k}_ci95_high"] = round(float(mean_val + ci_half), 4)
                row["n_participants"] = n

                print(f"  {k:20s}: {mean_val:.4f} ± {std_val:.4f}  [IC95%: ({mean_val - ci_half:.4f}, {mean_val + ci_half:.4f})]  n={n}")
            else:
                row[f"{k}_mean"] = np.nan
                row[f"{k}_std"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run capacity-matched ablation experiments.")
    parser.add_argument("--dry-run", action="store_true", help="Run a quick synthetic test without loading real datasets.")
    parser.add_argument("--data-root", type=str, default="data/WESAD/WESAD", help="Path to WESAD raw data.")
    parser.add_argument("--window-seconds", type=int, default=60, choices=[30, 60], help="Window length in seconds (30 or 60).")
    parser.add_argument("--stride-seconds", type=int, default=None, help="Window stride in seconds (default: window_seconds // 2, i.e., 50 percent overlap).")
    parser.add_argument("--norm-method", type=str, default="median", choices=["median", "mean"], help="Normalization method: 'median' or 'mean' (default: median).")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None, help="Supervised fine-tuning epochs (default from paper config).")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (default from paper config).")
    parser.add_argument("--max-participants", type=int, default=None, help="Optionally limit number of subjects for a quick run.")
    parser.add_argument("--save-csv", type=str, default=None, help="Custom output CSV path.")
    args = parser.parse_args()

    if args.dry_run:
        print("\n--- Running Dry-Run with synthetic physiological signals ---")
        N_SAMPLES = 120
        SEQ_LEN = args.window_seconds * 4  # 4 Hz
        CHANNELS = 3
        X_mock = np.random.randn(N_SAMPLES, SEQ_LEN, CHANNELS).astype(np.float32)
        y_mock = np.random.randint(0, 2, size=N_SAMPLES)
        groups_mock = np.repeat([1, 2, 3], N_SAMPLES // 3)

        model_cfg = {"patch_len": 16, "emb_dim": 128, "depth": 3}
        train_cfg = {"epochs": 1, "batch_size": 16}

        res = run_loso_ablation(
            X_mock, y_mock, groups_mock,
            device="cpu",
            seed=42,
            model_kwargs=model_cfg,
            train_kwargs=train_cfg,
        )
        summary = summarize_results(res)
        print("\n--- Dry Run Summary ---")
        print(summary.to_string(index=False))
        sys.exit(0)

    # 1. Configuración de hiperparámetros según ventana (de configs/selected_hyperparameters.csv)
    if args.window_seconds == 60:
        patch_len = 16
        emb_dim = 128
        depth = 3
        dropout = 0.056
        default_epochs = 6
        default_batch_size = 32
    else:  # 30s
        patch_len = 8
        emb_dim = 256
        depth = 6
        dropout = 0.206
        default_epochs = 7
        default_batch_size = 16

    epochs = args.epochs if args.epochs is not None else default_epochs
    batch_size = args.batch_size if args.batch_size is not None else default_batch_size

    model_kwargs = {
        "patch_len": patch_len,
        "emb_dim": emb_dim,
        "depth": depth,
        "dropout": dropout,
    }
    train_kwargs = {
        "epochs": epochs,
        "batch_size": batch_size,
    }

    # 2. Cargar WESAD
    wesad_root = Path(args.data_root)
    if not wesad_root.exists():
        # Intentar ruta alternativa
        alt_root = PROJECT_ROOT / args.data_root
        if alt_root.exists():
            wesad_root = alt_root
        else:
            raise FileNotFoundError(f"Could not find WESAD data directory at '{args.data_root}'.")

    stride = args.stride_seconds if args.stride_seconds is not None else (args.window_seconds // 2)
    print(f"Loading WESAD dataset from: {wesad_root.resolve()} (window={args.window_seconds}s, stride={stride}s, norm={args.norm_method})...")
    X, y, groups = load_all_wesad(wesad_root, window_seconds=args.window_seconds, stride_seconds=stride, norm_method=args.norm_method)
    print(f"Loaded {len(X)} windows across {len(np.unique(groups))} participants: {list(np.unique(groups))}")
    print(f"Class distribution: Stress={np.sum(y == 1)} ({np.mean(y == 1)*100:.1f}%), Non-Stress={np.sum(y == 0)} ({np.mean(y == 0)*100:.1f}%)\n")

    # 3. Ejecutar LOSO Ablation
    results = run_loso_ablation(
        X, y, groups,
        device=args.device,
        seed=args.seed,
        model_kwargs=model_kwargs,
        train_kwargs=train_kwargs,
        max_participants=args.max_participants,
    )

    summary_df = summarize_results(results)
    print("\n=======================================================")
    print(f"ABLATION RESULTS SUMMARY: WESAD {args.window_seconds}s (Reviewer Point 5)")
    print("=======================================================")
    print(summary_df.to_string(index=False))
    print("=======================================================\n")

    # 4. Guardar resultados
    out_csv = args.save_csv
    if out_csv is None:
        out_csv = PROJECT_ROOT / "results" / "tables" / "ablation_matched_capacity" / f"ablation_matched_capacity_results_wesad_{args.window_seconds}s.csv"
    else:
        out_csv = Path(out_csv)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_csv, index=False)
    print(f"Summary table successfully saved to: {out_csv.resolve()}")

    # Mantener espejo en results/tables/ para retrocompatibilidad
    legacy_csv = PROJECT_ROOT / "results" / "tables" / f"ablation_matched_capacity_results_wesad_{args.window_seconds}s.csv"
    if legacy_csv.resolve() != out_csv.resolve():
        summary_df.to_csv(legacy_csv, index=False)

    # Guardar métricas detalladas por participante
    participant_rows = []
    for variant, p_metrics in results.items():
        participant_rows.extend(p_metrics)
    part_df = pd.DataFrame(participant_rows)
    part_csv = out_csv.parent / out_csv.name.replace("results_", "participants_")
    part_df.to_csv(part_csv, index=False)
    print(f"Participant-level table saved to: {part_csv.resolve()}")

    legacy_part_csv = PROJECT_ROOT / "results" / "tables" / f"ablation_matched_capacity_participants_wesad_{args.window_seconds}s.csv"
    if legacy_part_csv.resolve() != part_csv.resolve():
        part_df.to_csv(legacy_part_csv, index=False)
    print()
