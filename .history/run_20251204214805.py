# run.py

import os
import numpy as np
import pandas as pd

from preprocess import load_dataset
from decompose import (
    run_all_decompositions,
    tucker_latent_features_core,
    tucker_latent_features_mode3,
)
from windows import tensor_to_timeseries, make_sliding_windows, time_split
from models import (
    evaluate_regression,
    train_ridge, test_ridge,
    train_lstm, test_lstm,
)
from config import TUCKER_RANKS, WINDOW, HORIZON, TRAIN_RATIO, VAL_RATIO


def main():
    os.makedirs("results", exist_ok=True)

    try:
        # 1. Load daily tensor
        print("Loading dataset...")
        X, Y, dates, mean, std = load_dataset()
        print(f"Loaded tensor: X{X.shape}, Y{Y.shape}, {len(dates)} days")
        # X: (D, T, F), Y: (D, T)

        # 2. Decompositions (CP/Tucker/PCA) on the DAILY tensor
        print("Running tensor decompositions...")
        cp_recon, tucker_recon, cp_latent_dict, pca_latent_dict = run_all_decompositions(X)
        print("Decompositions completed successfully")
        
    except Exception as e:
        print(f"Error in data loading or decomposition: {e}")
        print("Check your data path configuration and tensor decomposition settings")
        raise

    # 2a. Save reconstruction errors for later analysis
    rows = []
    for r, (_, rel_err) in cp_recon.items():
        rows.append({
            "method": "cp",
            "rank": r,
            "rank1": r,
            "rank2": None,
            "rank3": None,
            "rel_error": rel_err,
        })
    for ranks, (_, rel_err) in tucker_recon.items():
        r1, r2, r3 = ranks
        rows.append({
            "method": "tucker",
            "rank": None,
            "rank1": r1,
            "rank2": r2,
            "rank3": r3,
            "rel_error": rel_err,
        })
    df_decomp = pd.DataFrame(rows)
    df_decomp.to_csv("results/decomposition_summary.csv", index=False)
    print("Saved results/decomposition_summary.csv")

    # 3. Flatten to global time series (10-min grid)
    X_ts_raw, y_ts = tensor_to_timeseries(X, Y)   # (N, F), (N,)
    X_win_raw, y_win_raw = make_sliding_windows(X_ts_raw, y_ts)

    # 3a. Persistence baseline y(t) = y(t-1) on raw target series
    n_samples = X_win_raw.shape[0]
    y_persist_all = np.zeros_like(y_win_raw)
    for i in range(n_samples):
        label_idx = i + WINDOW + HORIZON - 1      # index of y_ts used as label
        prev_idx = label_idx - HORIZON            # last observed target
        y_persist_all[i] = y_ts[prev_idx]

    # 4. Build settings: RAW, CP-latent, PCA-latent, Tucker-latent
    settings = []

    # RAW windows (already Hankel)
    settings.append(("raw", X_win_raw, y_win_raw))

    # CP latent on DAILY tensor: Z_cp has shape (D, T, R_cp)
    for r, Z_cp in cp_latent_dict.items():
        X_ts_cp, _ = tensor_to_timeseries(Z_cp, Y)        # (N, R_cp)
        X_win_cp, y_win_cp = make_sliding_windows(X_ts_cp, y_ts)
        settings.append((f"cp_latent_r{r}", X_win_cp, y_win_cp))

    # PCA latent on DAILY tensor: Z_pca has shape (D, T, k)
    for k, Z_pca in pca_latent_dict.items():
        X_ts_pca, _ = tensor_to_timeseries(Z_pca, Y)      # (N, k)
        X_win_pca, y_win_pca = make_sliding_windows(X_ts_pca, y_ts)
        settings.append((f"pca_k{k}", X_win_pca, y_win_pca))

    # Tucker latent (mode-3 "feature compression") on DAILY tensor
    for ranks in TUCKER_RANKS:
        r1, r2, r3 = ranks
        Z_tk_m3, _ = tucker_latent_features_mode3(X, ranks=ranks)
        X_ts_tk_m3, _ = tensor_to_timeseries(Z_tk_m3, Y)  # (N, r3)
        X_win_tk_m3, y_win_tk_m3 = make_sliding_windows(X_ts_tk_m3, y_ts)
        settings.append((f"tucker_latent_m3_{r1}_{r2}_{r3}", X_win_tk_m3, y_win_tk_m3))

    # OPTIONAL: Tucker latent via core contraction (more "multiway" features).
    # Comment this block out if runtime is too high.
    for ranks in TUCKER_RANKS:
        r1, r2, r3 = ranks
        Z_tk_core, _ = tucker_latent_features_core(X, ranks=ranks)
        X_ts_tk_core, _ = tensor_to_timeseries(Z_tk_core, Y)
        X_win_tk_core, y_win_tk_core = make_sliding_windows(X_ts_tk_core, y_ts)
        settings.append((f"tucker_latent_core_{r1}_{r2}_{r3}", X_win_tk_core, y_win_tk_core))

    # 5. Loop over settings and models (baseline, ridge, LSTM)
    perf_records = []
    total_settings = len(settings)

    for i, (setting_name, X_win, y_win) in enumerate(settings, 1):
        print(f"\n=== Setting {i}/{total_settings}: {setting_name} ===")
        print(f"Windows shape: X{X_win.shape}, y{y_win.shape}")
        
        try:
            (X_train, y_train), (X_val, y_val), (X_test, y_test) = time_split(X_win, y_win)
            print(f"Split sizes - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        except Exception as e:
            print(f"Error in time split for {setting_name}: {e}")
            continue

        # Standardize inputs per setting using TRAIN split only
        mean = X_train.mean(axis=(0, 1), keepdims=True)
        std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
        X_train_s = (X_train - mean) / std
        X_val_s = (X_val - mean) / std
        X_test_s = (X_test - mean) / std

        # 1) Persistence baseline (only once, for 'raw')
        if setting_name == "raw":
            N = len(y_win)
            n_train = int(N * TRAIN_RATIO)
            n_val = int(N * VAL_RATIO)
            y_p_test = y_persist_all[n_train + n_val:]

            mae, rmse = evaluate_regression(y_test, y_p_test)
            perf_records.append({
                "setting": setting_name,
                "model": "persistence",
                "mae": float(mae),
                "rmse": float(rmse),
            })

            # save persistence predictions too
            np.savez(
                "results/preds_raw_persistence.npz",
                y_test=y_test,
                y_pred=y_p_test,
            )

        # 2) Ridge
        ridge_model, _, _ = train_ridge(X_train_s, y_train, X_val_s, y_val)
        mae, rmse = test_ridge(ridge_model, X_test_s, y_test)
        perf_records.append({
            "setting": setting_name,
            "model": "ridge",
            "mae": float(mae),
            "rmse": float(rmse),
        })

        # save ridge predictions for this setting
        s_test, w, f = X_test_s.shape
        Xte = X_test_s.reshape(s_test, w * f)  # <-- flatten window
        y_ridge_pred = ridge_model.predict(Xte).ravel()

        np.savez(
            f"results/preds_{setting_name}_ridge.npz",
            y_test=y_test,
            y_pred=y_ridge_pred,
        )

        # 3) LSTM
        lstm_model, _ = train_lstm(X_train_s, y_train, X_val_s, y_val)
        mae, rmse = test_lstm(lstm_model, X_test_s, y_test)
        perf_records.append({
            "setting": setting_name,
            "model": "lstm",
            "mae": float(mae),
            "rmse": float(rmse),
        })

        # NEW: save LSTM predictions for this setting
        y_lstm_pred = lstm_model.predict(X_test_s, verbose=0).ravel()
        np.savez(
            f"results/preds_{setting_name}_lstm.npz",
            y_test=y_test,
            y_pred=y_lstm_pred,
        )

    df_perf = pd.DataFrame(perf_records)
    df_perf.to_csv("results/model_performance.csv", index=False)
    print("Saved results/model_performance.csv")


if __name__ == "__main__":
    main()
