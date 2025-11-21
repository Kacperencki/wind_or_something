# windows.py

import numpy as np
from typing import Tuple

from config import WINDOW, HORIZON, TRAIN_RATIO, VAL_RATIO


def tensor_to_timeseries(X: np.ndarray, Y: np.ndarray):
    """
    X: (D, T, F)
    Y: (D, T)
    Return:
        X_ts: (N, F)
        y_ts: (N,)
        where N = D*T
    """
    D, T, F = X.shape
    X_ts = X.reshape(D * T, F)
    y_ts = Y.reshape(D * T)
    return X_ts, y_ts


def make_sliding_windows(
    X_ts: np.ndarray, y_ts: np.ndarray,
    window: int = WINDOW, horizon: int = HORIZON
) -> Tuple[np.ndarray, np.ndarray]:
    N, F = X_ts.shape
    samples = N - window - horizon + 1
    X_win = np.zeros((samples, window, F), dtype=X_ts.dtype)
    y_win = np.zeros(samples, dtype=y_ts.dtype)

    for i in range(samples):
        X_win[i] = X_ts[i:i+window]
        y_win[i] = y_ts[i+window+horizon-1]

    return X_win, y_win


def time_split(X, y, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO):
    N = len(X)
    n_train = int(N * train_ratio)
    n_val = int(N * val_ratio)

    X_train = X[:n_train]
    y_train = y[:n_train]

    X_val = X[n_train:n_train+n_val]
    y_val = y[n_train:n_train+n_val]

    X_test = X[n_train+n_val:]
    y_test = y[n_train+n_val:]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


if __name__ == "__main__":
    from preprocess import load_dataset

    X, Y, _, _, _ = load_dataset()
    X_ts, y_ts = tensor_to_timeseries(X, Y)
    X_win, y_win = make_sliding_windows(X_ts, y_ts)
    print("X_win:", X_win.shape)
    print("y_win:", y_win.shape)
