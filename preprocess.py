# preprocess.py

import os
import numpy as np
import pandas as pd

from config import (
    DATA_CSV,
    TIMESTAMP_COL,
    TARGET_COL,
    FEATURE_COLS,
    RESAMPLE_RULE,
    SLOTS_PER_DAY,
    SEED,
    MAX_DAYS,
)

rng = np.random.default_rng(SEED)


def _read_raw(path: str) -> pd.DataFrame:
    """
    Read the raw V52 file.
    - If .xlsx/.xls: use read_excel.
    - Otherwise: try read_csv with sensible defaults.
    """
    ext = os.path.splitext(path)[1].lower()

    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    else:
        try:
            df = pd.read_csv(path)
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="latin1", sep=None, engine="python")

    return df


def load_and_resample() -> pd.DataFrame:
    """Load raw V52 SCADA and resample to a regular 10-min grid."""
    df = _read_raw(DATA_CSV)

    if TIMESTAMP_COL not in df.columns:
        raise ValueError(
            f"Timestamp column '{TIMESTAMP_COL}' not found in file. "
            f"Columns: {list(df.columns)}"
        )

    # Parse timestamps (Excel often already gives datetime, but this is safe)
    df[TIMESTAMP_COL] = pd.to_datetime(
        df[TIMESTAMP_COL],
        dayfirst=True,
        errors="coerce",
    )
    df = df.dropna(subset=[TIMESTAMP_COL])
    df = df.sort_values(TIMESTAMP_COL)

    # Replace "not active" sentinel -999 with NaN for all numeric columns
    num_cols = df.select_dtypes(include=["number"]).columns
    if len(num_cols) > 0:
        df[num_cols] = df[num_cols].replace(-999, np.nan)

    # Basic sanity filters for wind speed / power if present
    if "WindSpeed" in df.columns:
        df = df[df["WindSpeed"] >= 0]
    if TARGET_COL in df.columns:
        df = df[df[TARGET_COL] >= 0]

    # Set index and resample to regular 10-min grid
    df = df.set_index(TIMESTAMP_COL)
    df = df.resample(RESAMPLE_RULE).mean()

    return df


def make_daily_tensor(df: pd.DataFrame):
    """
    Convert resampled dataframe into daily tensor X(D,T,F) and Y(D,T).

    Keeps only days with a full SLOTS_PER_DAY samples.
    Optionally limits to MAX_DAYS most recent days.
    Standardizes features globally (per variable).
    """
    # Ensure required columns exist
    cols_needed = [TARGET_COL] + FEATURE_COLS
    for c in cols_needed:
        if c not in df.columns:
            raise ValueError(
                f"Column '{c}' not found after resampling. "
                f"Available: {list(df.columns)}"
            )

    # Drop rows with missing target or features
    df = df.dropna(subset=cols_needed)

    # Add date and slot index (0..SLOTS_PER_DAY-1)
    df["date"] = df.index.date
    df["slot"] = df.index.hour * (60 // 10) + df.index.minute // 10

    # Keep only slots within [0, SLOTS_PER_DAY)
    df = df[(df["slot"] >= 0) & (df["slot"] < SLOTS_PER_DAY)]

    # Keep only full days
    counts = df.groupby("date")["slot"].nunique()
    full_dates_all = sorted(counts[counts == SLOTS_PER_DAY].index)

    if MAX_DAYS is not None and len(full_dates_all) > MAX_DAYS:
        # Use most recent MAX_DAYS (chronologically last)
        full_dates = full_dates_all[-MAX_DAYS:]
    else:
        full_dates = full_dates_all

    df = df[df["date"].isin(full_dates)]

    D = len(full_dates)
    T = SLOTS_PER_DAY
    F = len(FEATURE_COLS)

    if D == 0:
        raise ValueError("No full days with SLOTS_PER_DAY samples found after cleaning.")

    X = np.zeros((D, T, F), dtype=np.float32)
    Y = np.zeros((D, T), dtype=np.float32)

    for d_idx, d in enumerate(full_dates):
        sub = df[df["date"] == d].sort_values("slot")
        if len(sub) != T:
            raise ValueError(f"Date {d} has {len(sub)} slots, expected {T}")

        X[d_idx, :, :] = sub[FEATURE_COLS].values.astype(np.float32)
        Y[d_idx, :] = sub[TARGET_COL].values.astype(np.float32)

    # Standardize features globally
    X_flat = X.reshape(-1, F)
    mean = X_flat.mean(axis=0)
    std = X_flat.std(axis=0) + 1e-8
    X = (X - mean) / std

    return X, Y, full_dates, mean, std


def load_dataset():
    """Main entry used by run.py / raw_cp_pca_tucker.py."""
    df = load_and_resample()
    X, Y, dates, mean, std = make_daily_tensor(df)
    return X, Y, dates, mean, std


if __name__ == "__main__":
    X, Y, dates, mean, std = load_dataset()
    print("X shape:", X.shape)  # (D, T, F)
    print("Y shape:", Y.shape)  # (D, T)
    print("Days:", len(dates))
