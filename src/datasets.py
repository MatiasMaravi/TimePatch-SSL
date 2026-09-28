"""Dataset loading and participant-level split utilities."""
from __future__ import annotations
import os, pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from .preprocessing import resample_wesad_1d, resample_window_to_length, segment_windows

WESAD_USE_LABELS = {1, 2, 3}
WESAD_STRESS_LABEL = 2


class WindowDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = None if y is None else torch.as_tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx] if self.y is None else (self.X[idx], self.y[idx])


def load_wesad_subject_raw(pkl_path, target_fs=4):
    """Load and resample one WESAD participant without fitting normalization."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    wrist = data["signal"]["wrist"]
    labels = np.asarray(data["label"]).astype(int).squeeze()
    bvp = resample_wesad_1d(np.asarray(wrist["BVP"]).squeeze(), 64, target_fs)
    eda = resample_wesad_1d(np.asarray(wrist["EDA"]).squeeze(), 4, target_fs)
    temp = resample_wesad_1d(np.asarray(wrist["TEMP"]).squeeze(), 4, target_fs)
    labels = resample_wesad_1d(labels, 700, target_fs).round().astype(int)
    L = min(map(len, (bvp, eda, temp, labels)))
    X = np.stack([bvp[:L], eda[:L], temp[:L]], axis=1).astype(np.float32)
    labels = labels[:L]
    keep = np.isin(labels, list(WESAD_USE_LABELS))
    return X[keep], np.where(labels[keep] == WESAD_STRESS_LABEL, 1, 0).astype(np.int64)


def load_all_wesad(root, window_seconds=30, stride_seconds=30, target_fs=4):
    Xs, ys, groups = [], [], []
    for subj in sorted(d for d in os.listdir(root) if d.startswith("S")):
        path = os.path.join(root, subj, f"{subj}.pkl")
        if not os.path.exists(path):
            continue
        X, y = load_wesad_subject_raw(path, target_fs)
        Xw, yw = segment_windows(X, y, window_seconds, stride_seconds, target_fs)
        if len(Xw):
            Xs.append(Xw); ys.append(yw)
            groups.append(np.full(len(yw), int(subj[1:]), dtype=np.int64))
    if not Xs:
        raise ValueError(f"No WESAD participant data found under {root!r}.")
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(groups)


def build_stress_predict_from_processed(sensor_files, target_length, label_col="TASK_LABEL"):
    """Build aligned Stress-Predict windows without participant-wise normalization."""
    meta = {"Participant","Activity","PSS_BEFORE","PSS_AFTER","BASELINE_LABEL",
            "TASK_LABEL","axis"}
    frames = {}
    for name, path in sensor_files.items():
        df = pd.read_csv(path, index_col=0)
        participants = df.index.astype(str)
        df = df.reset_index(drop=True)
        df["Participant"] = participants
        if label_col not in df:
            raise ValueError(f"{label_col!r} is absent from {path}.")
        sig_cols = [c for c in df.columns if c not in meta and c not in ("Participant","Activity")]
        frames[name] = df[["Participant","Activity"] + sig_cols + [label_col]].copy()
    for key, df in frames.items():
        df["window_idx"] = df.groupby(["Participant","Activity"]).cumcount()
        df["_key"] = (df["Participant"].astype(str)+"||"+df["Activity"].astype(str)+"||"+
                      df["window_idx"].astype(str))
    common = sorted(set.intersection(*(set(df["_key"]) for df in frames.values())))
    Xs, ys, gs = [], [], []
    for key in common:
        channels=[]; participant=None; label=None
        for df in frames.values():
            row=df.loc[df["_key"] == key]
            if len(row) != 1: break
            row=row.iloc[0]
            sig_cols=[c for c in df.columns if c not in meta and
                      c not in ("Participant","Activity","window_idx","_key",label_col)]
            channels.append(resample_window_to_length(row[sig_cols].to_numpy(np.float32),
                                                      target_length))
            participant = participant or str(row["Participant"])
            label = 0 if row["Activity"] == "Baseline" else 1
        else:
            Xs.append(np.stack(channels, axis=1)); ys.append(label); gs.append(participant)
    return np.asarray(Xs,np.float32), np.asarray(ys,np.int64), np.asarray(gs)


def participant_split_indices(groups, train_fraction=0.80, random_state=42):
    """Split unique participant IDs first; then map all windows to their partition."""
    groups=np.asarray(groups)
    participants=np.unique(groups)
    train_ids, test_ids=train_test_split(participants, train_size=train_fraction,
                                         random_state=random_state, shuffle=True)
    train_idx=np.flatnonzero(np.isin(groups, train_ids))
    test_idx=np.flatnonzero(np.isin(groups, test_ids))
    assert set(train_ids).isdisjoint(set(test_ids))
    assert len(np.intersect1d(train_idx,test_idx)) == 0
    return train_idx, test_idx, train_ids, test_ids


def load_wesad_window_configurations(root, stride_seconds=30, target_fs=4):
    """Load both manuscript window configurations: 30-second and 60-second."""
    configurations = {}
    for window_seconds in (30, 60):
        X, y, groups = load_all_wesad(
            root,
            window_seconds=window_seconds,
            stride_seconds=stride_seconds,
            target_fs=target_fs,
        )
        configurations[window_seconds] = {
            "X": X,
            "y": y,
            "groups": groups,
        }
    return configurations
