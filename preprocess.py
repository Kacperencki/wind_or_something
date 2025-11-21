# preprocess.py

import numpy as np
import pandas as pd

from config import (
    DATA_CSV, TIMESTAMP_COL, TARGET_COL, FEATURE_COLS,
    RESAMPLE_RULE, SLOTS_PER_DAY, SEED
)


def load_and_resample() -> pd.DataFrame:
    df = pd.read_csv(DATA_CSV)

    # parse timestamp: "01 01 2018 00:00"
    df[TIMESTAMP_COL] = pd.to_datetime(
        df[TIMESTAMP_COL],
        dayfirst=True,          # 01 01 2018 -> 1st Jan, not 1st of month 1/2018
        errors="coerce"
    )
    df = df.dropna(subset=[TIMESTAMP_COL])
    df = df.sort_values(TIMESTAMP_COL).set_index(TIMESTAMP_COL)

    cols = FEATURE_COLS + [TARGET_COL]
    df = df[cols]

    df = df.resample(RESAMPLE_RULE).mean()
    df = df.interpolate(method="time").dropna()

    return df


def make_daily_tensor(df: pd.DataFrame):
    """
    df indexed by DateTime, columns: FEATURES + TARGET.
    Returns:
        X: 3D array (D, T, F) of normalized features
        Y: 2D array (D, T) of target
    """
    # add date and slot index
    df = df.copy()
    df["date"] = df.index.date
    df["slot"] = df.groupby("date").cumcount()

    # keep only full days with SLOTS_PER_DAY samples
    counts = df.groupby("date")["slot"].max() + 1
    full_dates = counts[counts == SLOTS_PER_DAY].index

    df = df[df["date"].isin(full_dates)]

    # sort by date then slot
    df = df.sort_values(["date", "slot"])

    # feature matrix and target
    F = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    # normalize features (per feature)
    mean = F.mean(axis=0, keepdims=True)
    std = F.std(axis=0, keepdims=True) + 1e-8
    F_norm = (F - mean) / std

    # reshape into (D, T, F)
    D = len(full_dates)
    T = SLOTS_PER_DAY
    num_features = len(FEATURE_COLS)

    X = F_norm.reshape(D, T, num_features)
    Y = y.reshape(D, T)

    return X, Y, full_dates, mean, std


def load_dataset():
    df = load_and_resample()
    X, Y, dates, mean, std = make_daily_tensor(df)
    return X, Y, dates, mean, std


if __name__ == "__main__":
    X, Y, dates, mean, std = load_dataset()
    print("X shape:", X.shape)  # (D, T, F)
    print("Y shape:", Y.shape)  # (D, T)
    print("Days:", len(dates))
