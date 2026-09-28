"""Signal preprocessing for TimePatch-SSL."""
from __future__ import annotations
import numpy as np
from scipy import signal

TARGET_FS = 4
SUPPORTED_WINDOW_SECONDS = (30, 60)
DEFAULT_STRIDE_SECONDS = 30


def resample_1d(x, orig_fs, target_fs=TARGET_FS):
    """Match the notebook: scipy.signal.resample Fourier resampling."""
    x = np.asarray(x).squeeze()
    if orig_fs == target_fs:
        return x.astype(np.float32)
    duration = len(x) / float(orig_fs)
    target_len = max(2, int(round(duration * target_fs)))
    return signal.resample(x, target_len).astype(np.float32)


class TrainingOnlyRobustScaler:
    """Median/IQR scaler fitted exclusively on training windows."""
    def __init__(self, eps=1e-6):
        self.eps = eps
        self.median_ = None
        self.scale_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 3:
            raise ValueError("Expected (n_windows, time, channels).")
        self.median_ = np.nanmedian(X, axis=(0, 1)).astype(np.float32)
        q25 = np.nanpercentile(X, 25, axis=(0, 1))
        q75 = np.nanpercentile(X, 75, axis=(0, 1))
        iqr = q75 - q25
        std = np.nanstd(X, axis=(0, 1))
        self.scale_ = np.where(iqr > self.eps, iqr, std + self.eps).astype(np.float32)
        return self

    def transform(self, X):
        if self.median_ is None:
            raise RuntimeError("Scaler has not been fitted.")
        X = np.asarray(X, dtype=np.float32).copy()
        for c in range(X.shape[-1]):
            bad = ~np.isfinite(X[..., c])
            X[..., c][bad] = self.median_[c]
        return ((X - self.median_) / self.scale_).astype(np.float32)

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def segment_windows(X, y, window_seconds, stride_seconds=DEFAULT_STRIDE_SECONDS, fs=TARGET_FS):
    """Create 30- or 60-s windows; retain stress proportion <=.2 or >=.8."""
    if window_seconds not in SUPPORTED_WINDOW_SECONDS:
        raise ValueError(f"window_seconds must be one of {SUPPORTED_WINDOW_SECONDS}.")
    win, stride = int(window_seconds*fs), int(stride_seconds*fs)
    Xw, yw = [], []
    for start in range(0, len(X)-win+1, stride):
        end = start + win
        rho = float(np.mean(y[start:end]))
        if rho <= 0.2 or rho >= 0.8:
            Xw.append(np.asarray(X[start:end], np.float32))
            yw.append(int(rho >= 0.8))
    shape = (0, win, X.shape[1])
    return ((np.stack(Xw) if Xw else np.empty(shape, np.float32)),
            np.asarray(yw, dtype=np.int64))
