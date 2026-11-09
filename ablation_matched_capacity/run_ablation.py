"""Experiment runner for capacity-matched ablation study (Reviewer Point 5 / Reviewer 3).

Compares:
1. Standard (base capacity, linear patch embed)
2. Hierarchical (local Transformer patch embed)
3. Standard-Matched (matched capacity to Hierarchical, linear patch embed)

Supports both WESAD and Stress-Predict (Stress-ID) under strict Leave-One-Group-Out (LOGO)
cross-validation (Experiment 1 supervised protocol).
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

for candidate_timemae in [
    PROJECT_ROOT / "notebooks" / "TimeMAE",
    Path("C:/Users/USUARIO/Desktop/TimePatch-SSL/notebooks/TimeMAE"),
    Path("C:/Users/USUARIO/Desktop/PFC/PFC2"),
]:
    if candidate_timemae.exists() and str(candidate_timemae) not in sys.path:
        sys.path.insert(0, str(candidate_timemae))
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
from src.models import count_trainable_parameters


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


def load_all_wesad(
    root: str | Path,
    window_seconds: int = 60,
    stride_seconds: int | None = None,
    target_fs: int = 4,
    norm_method: str = "median",
):
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


# --- Stress-Predict (Stress-ID) Dataset Loader ---
def resample_window_interp(signal_arr: np.ndarray, target_len: int) -> np.ndarray:
    """Linear interpolation resampling to target_len matching notebook build_dataset_from_processed."""
    signal_arr = np.asarray(signal_arr, dtype=np.float32)
    if len(signal_arr) == target_len:
        return signal_arr
    if len(signal_arr) < 2:
        return np.full(target_len, float(signal_arr[0]) if len(signal_arr) else 0.0, dtype=np.float32)
    xp = np.linspace(0, 1, len(signal_arr))
    x = np.linspace(0, 1, target_len)
    return np.interp(x, xp, signal_arr).astype(np.float32)


def load_all_stress_predict(
    processed_dir: str | Path,
    window_seconds: int = 60,
    target_fs: int = 4,
    norm_method: str = "median",
    label_col: str = "TASK_LABEL",
):
    """Load processed Stress-Predict BVP, EDA, TEMP CSV files matching notebook build_dataset_from_processed."""
    processed_dir = Path(processed_dir)
    sensor_files = {
        "BVP": processed_dir / f"BVP_{window_seconds}_seg.csv",
        "EDA": processed_dir / f"EDA_{window_seconds}_seg.csv",
        "TEMP": processed_dir / f"TEMP_{window_seconds}_seg.csv",
    }
    for name, path in sensor_files.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing Stress-Predict processed file for {name}: {path}")

    target_length = int(window_seconds * target_fs)
    meta_cols = {"Participant", "Activity", "PSS_BEFORE", "PSS_AFTER", "BASELINE_LABEL", "TASK_LABEL", "axis"}
    frames = {}

    for name, path in sensor_files.items():
        df = pd.read_csv(path, index_col=0)
        participants_index = df.index.astype(str)
        df = df.reset_index(drop=True)
        df["Participant"] = participants_index

        if label_col not in df.columns:
            raise ValueError(f"label_col '{label_col}' not found in {path}; columns: {list(df.columns)[:10]}")

        sig_cols = [c for c in df.columns if c not in meta_cols and c not in ("Participant", "Activity")]
        df[sig_cols] = df[sig_cols].astype(np.float32)

        # Participant-level robust normalization matching notebook
        for part, group in df.groupby("Participant"):
            flat_sig = group[sig_cols].values.astype(np.float32).flatten()
            if np.isnan(flat_sig).any():
                nan_val = np.nanmedian(flat_sig) if norm_method == "median" else np.nanmean(flat_sig)
                flat_sig = np.nan_to_num(flat_sig, nan=nan_val if np.isnan(flat_sig).any() else 0.0)
            med = np.median(flat_sig) if norm_method == "median" else np.mean(flat_sig)
            iqr = np.percentile(flat_sig, 75) - np.percentile(flat_sig, 25)
            scale = iqr if iqr > 1e-6 else (np.std(flat_sig) + 1e-6)

            norm_vals = (df.loc[group.index, sig_cols].values.astype(np.float32) - med) / scale
            df.loc[group.index, sig_cols] = norm_vals

        frames[name] = df[["Participant", "Activity"] + sig_cols + [label_col]].copy()

    for k, df in frames.items():
        df["window_idx"] = df.groupby(["Participant", "Activity"]).cumcount()
        df["_key"] = (
            df["Participant"].astype(str)
            + "||"
            + df["Activity"].astype(str)
            + "||"
            + df["window_idx"].astype(str)
        )
        frames[k] = df

    common_keys = sorted(set.intersection(*[set(frames[k]["_key"].values) for k in frames]))

    # Index by _key for fast lookup while preserving exact row alignment
    indexed_frames = {}
    sig_cols_map = {}
    for name, df in frames.items():
        counts = df["_key"].value_counts()
        valid_keys = set(counts[counts == 1].index)
        indexed_frames[name] = df[df["_key"].isin(valid_keys)].set_index("_key")
        sig_cols_map[name] = [
            c for c in df.columns
            if c not in meta_cols and c not in ("Participant", "Activity", "window_idx", "_key", label_col)
        ]

    X_list, y_list, participants_list = [], [], []
    for key in common_keys:
        if not all(key in indexed_frames[name].index for name in frames):
            continue
        channel_data = []
        label_val = None
        part = None
        for name in ("BVP", "EDA", "TEMP"):
            row = indexed_frames[name].loc[key]
            sig = row[sig_cols_map[name]].values.astype(np.float32)
            sig = resample_window_interp(sig, target_length)
            channel_data.append(sig)
            if label_val is None:
                activity = row["Activity"]
                label_val = 0 if activity == "Baseline" else 1
            if part is None:
                part = str(row["Participant"])

        X_list.append(np.stack(channel_data, axis=1))
        y_list.append(label_val)
        participants_list.append(part)

    if not X_list:
        raise ValueError(f"No aligned Stress-Predict windows built from {processed_dir}")

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    groups = np.array([int(p.lstrip("S")) for p in participants_list], dtype=np.int64)
    return X, y, groups


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

    # Log trainable parameters per variant before starting
    param_counts = {}
    for v in variants:
        tmp_model = build_model(variant=v, **model_kwargs)
        param_counts[v] = int(count_trainable_parameters(tmp_model))
        del tmp_model

    print(f"\n=======================================================")
    print(f"Starting LOSO Ablation on {len(participants)} participants")
    print(f"Variants being evaluated: {list(variants)}")
    print(f"Trainable parameters: {param_counts}")
    if "hierarchical" in param_counts and "standard_matched" in param_counts:
        diff = param_counts["standard_matched"] - param_counts["hierarchical"]
        rel_pct = (diff / param_counts["hierarchical"]) * 100.0
        print(f"Capacity parity (Standard-Matched vs Hierarchical): {diff:+d} params ({rel_pct:+.2f}%)")
    print(f"Model hyperparameters: {model_kwargs}")
    print(f"Training parameters: {train_kwargs}")
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
                "participant": int(heldout),
                "variant": variant,
                "n_windows": int(len(test_idx)),
                "trainable_params": param_counts[variant],
            })
            results[variant].append(metrics)
            print(
                f"  -> {variant:16s} | Acc: {metrics['accuracy']:.4f} | "
                f"F1: {metrics['f1']:.4f} | BAcc: {metrics['balanced_accuracy']:.4f} | "
                f"AUROC: {metrics['auroc']:.4f}"
            )

    return results


def summarize_results(results: dict, alpha: float = 0.05) -> pd.DataFrame:
    """Computes mean, sample std (ddof=1), and 95% CI across participants matching notebook summarize()."""
    rows = []
    metric_keys = ("accuracy", "f1", "balanced_accuracy", "auroc")
    for variant, participant_metrics in results.items():
        trainable_params = participant_metrics[0].get("trainable_params", None) if participant_metrics else None
        row = {"variant": variant}
        if trainable_params is not None:
            row["trainable_params"] = trainable_params
        print(f"\n--- Summary: {variant.upper()} (params={trainable_params}) ---")
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


def resolve_data_dir(dataset: str, custom_root: str | None) -> Path:
    """Locate dataset directory across local workspace and fallback paths."""
    if custom_root:
        candidates = [
            Path(custom_root),
            PROJECT_ROOT / custom_root,
            Path("C:/Users/USUARIO/Desktop/TimePatch-SSL") / custom_root,
        ]
    elif dataset == "wesad":
        candidates = [
            PROJECT_ROOT / "data" / "WESAD" / "WESAD",
            Path("C:/Users/USUARIO/Desktop/TimePatch-SSL/data/WESAD/WESAD"),
        ]
    else:  # stress_predict
        candidates = [
            PROJECT_ROOT / "data" / "Stress-Predict" / "processed_data",
            PROJECT_ROOT / "data" / "Stress-Predict" / "Processed_data",
            Path("C:/Users/USUARIO/Desktop/TimePatch-SSL/data/Stress-Predict/processed_data"),
            Path("C:/Users/USUARIO/Desktop/TimePatch-SSL/data/Stress-Predict/Processed_data"),
            Path("C:/Users/USUARIO/Desktop/Tesis/processed_data/processed_data"),
        ]

    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"Could not find data directory for dataset={dataset!r}. Tried: {[str(c) for c in candidates]}")


def get_experiment1_config(dataset: str, window_seconds: int) -> tuple[dict, dict]:
    """Return (model_kwargs, train_kwargs) matching configs/selected_hyperparameters.csv for Experiment 1."""
    if dataset == "wesad":
        if window_seconds == 60:
            return (
                {"patch_len": 16, "emb_dim": 128, "depth": 3, "dropout": 0.05587873967732661},
                {"epochs": 6, "batch_size": 32},
            )
        else:  # 30s
            return (
                {"patch_len": 8, "emb_dim": 256, "depth": 6, "dropout": 0.20593139006678607},
                {"epochs": 7, "batch_size": 16},
            )
    else:  # stress_predict
        if window_seconds == 60:
            return (
                {"patch_len": 8, "emb_dim": 256, "depth": 4, "dropout": 0.20059047112039313},
                {"epochs": 6, "batch_size": 16},
            )
        else:  # 30s
            return (
                {"patch_len": 8, "emb_dim": 256, "depth": 6, "dropout": 0.20593139006678607},
                {"epochs": 7, "batch_size": 16},
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run capacity-matched ablation experiments (WESAD & Stress-Predict).")
    parser.add_argument("--dry-run", action="store_true", help="Run a quick synthetic test without loading real datasets.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="wesad",
        choices=["wesad", "stress_predict", "stressid"],
        help="Dataset to evaluate: 'wesad' or 'stress_predict' (alias 'stressid').",
    )
    parser.add_argument("--data-root", type=str, default=None, help="Optional custom path to dataset directory.")
    parser.add_argument("--window-seconds", type=int, default=60, choices=[30, 60], help="Window length in seconds (30 or 60).")
    parser.add_argument("--stride-seconds", type=int, default=None, help="Window stride in seconds for WESAD (default: window_seconds // 2).")
    parser.add_argument("--norm-method", type=str, default="median", choices=["median", "mean"], help="Normalization method: 'median' or 'mean' (default: median).")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None, help="Supervised fine-tuning epochs (default from paper config).")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (default from paper config).")
    parser.add_argument("--max-participants", type=int, default=None, help="Optionally limit number of subjects for a quick run.")
    parser.add_argument("--save-csv", type=str, default=None, help="Custom output CSV path.")
    args = parser.parse_args()

    dataset_key = "stress_predict" if args.dataset in ("stress_predict", "stressid") else "wesad"

    if args.dry_run:
        print("\n--- Running Dry-Run with synthetic physiological signals ---")
        N_SAMPLES = 120
        SEQ_LEN = args.window_seconds * 4  # 4 Hz
        CHANNELS = 3
        X_mock = np.random.randn(N_SAMPLES, SEQ_LEN, CHANNELS).astype(np.float32)
        y_mock = np.random.randint(0, 2, size=N_SAMPLES)
        groups_mock = np.repeat([1, 2, 3], N_SAMPLES // 3)

        model_cfg, _ = get_experiment1_config(dataset_key, args.window_seconds)
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

    # 1. Configuración de hiperparámetros según dataset y ventana (de configs/selected_hyperparameters.csv)
    model_kwargs, train_kwargs = get_experiment1_config(dataset_key, args.window_seconds)
    if args.epochs is not None:
        train_kwargs["epochs"] = args.epochs
    if args.batch_size is not None:
        train_kwargs["batch_size"] = args.batch_size

    # 2. Cargar Dataset (desde data/ o fallback automático a data_cache/*.npz para reproducibilidad portable)
    try:
        data_dir = resolve_data_dir(dataset_key, args.data_root)
        if dataset_key == "wesad":
            stride = args.stride_seconds if args.stride_seconds is not None else (args.window_seconds // 2)
            print(f"Loading WESAD dataset from: {data_dir.resolve()} (window={args.window_seconds}s, stride={stride}s, norm={args.norm_method})...")
            X, y, groups = load_all_wesad(data_dir, window_seconds=args.window_seconds, stride_seconds=stride, norm_method=args.norm_method)
        else:
            print(f"Loading Stress-Predict dataset from: {data_dir.resolve()} (window={args.window_seconds}s, norm={args.norm_method})...")
            X, y, groups = load_all_stress_predict(data_dir, window_seconds=args.window_seconds, norm_method=args.norm_method)
    except FileNotFoundError:
        cache_path = PROJECT_ROOT / "data_cache" / f"{dataset_key}_{args.window_seconds}s.npz"
        if not cache_path.exists():
            raise
        print(f"Raw/processed data directory not found; loading cached tensors from: {cache_path.resolve()}...")
        cached = np.load(cache_path)
        X, y, groups = cached["X"].astype(np.float32), cached["y"].astype(np.int64), cached["groups"].astype(np.int64)

    print(f"Loaded {len(X)} windows (shape={X.shape}) across {len(np.unique(groups))} participants: {[int(g) for g in np.unique(groups)]}")
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
    print(f"ABLATION RESULTS SUMMARY: {dataset_key.upper()} {args.window_seconds}s (Reviewer Point 5)")
    print("=======================================================")
    print(summary_df.to_string(index=False))
    print("=======================================================\n")

    # 4. Guardar resultados
    out_csv = args.save_csv
    if out_csv is None:
        out_csv = (
            PROJECT_ROOT
            / "results"
            / "tables"
            / "ablation_matched_capacity"
            / f"ablation_matched_capacity_results_{dataset_key}_{args.window_seconds}s.csv"
        )
    else:
        out_csv = Path(out_csv)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_csv, index=False)
    print(f"Summary table successfully saved to: {out_csv.resolve()}")

    # Mantener espejo en results/tables/ para retrocompatibilidad
    legacy_csv = PROJECT_ROOT / "results" / "tables" / f"ablation_matched_capacity_results_{dataset_key}_{args.window_seconds}s.csv"
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

    legacy_part_csv = PROJECT_ROOT / "results" / "tables" / f"ablation_matched_capacity_participants_{dataset_key}_{args.window_seconds}s.csv"
    if legacy_part_csv.resolve() != part_csv.resolve():
        part_df.to_csv(legacy_part_csv, index=False)
    print()
