# run_experiments.py

import numpy as np
from pprint import pprint
import os
import pandas as pd   

from preprocess import load_dataset
from decompose import run_all_decompositions
from windows import tensor_to_timeseries, make_sliding_windows, time_split
from models import (
    persistence_baseline,
    train_ridge, test_ridge,
    train_mlp, test_mlp,
    train_lstm, test_lstm,
    train_cnn, test_cnn,
)
from config import WINDOW, HORIZON


def build_windows_from_tensor(X, Y):
    X_ts, y_ts = tensor_to_timeseries(X, Y)
    X_win, y_win = make_sliding_windows(X_ts, y_ts, WINDOW, HORIZON)
    return X_win, y_win


def evaluate_setting(name, X_win, y_win):
    print(f"\n=== Setting: {name} ===")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = time_split(X_win, y_win)

    # Baseline
    baseline_mae, baseline_rmse = persistence_baseline(X_test, y_test)
    print("Persistence baseline  MAE/RMSE:", baseline_mae, baseline_rmse)

    # Ridge
    ridge, val_mae, val_rmse = train_ridge(X_train, y_train, X_val, y_val)
    print("Ridge val MAE/RMSE:", val_mae, val_rmse)
    ridge_mae, ridge_rmse = test_ridge(ridge, X_test, y_test)
    print("Ridge test MAE/RMSE:", ridge_mae, ridge_rmse)

    # MLP
    mlp, val_mae, val_rmse = train_mlp(X_train, y_train, X_val, y_val)
    print("MLP val MAE/RMSE:", val_mae, val_rmse)
    mlp_mae, mlp_rmse = test_mlp(mlp, X_test, y_test)
    print("MLP test MAE/RMSE:", mlp_mae, mlp_rmse)

    # LSTM
    lstm, _ = train_lstm(X_train, y_train, X_val, y_val)
    lstm_mae, lstm_rmse = test_lstm(lstm, X_test, y_test)
    print("LSTM test MAE/RMSE:", lstm_mae, lstm_rmse)

    # CNN
    cnn, _ = train_cnn(X_train, y_train, X_val, y_val)
    cnn_mae, cnn_rmse = test_cnn(cnn, X_test, y_test)
    print("CNN test MAE/RMSE:", cnn_mae, cnn_rmse)

    return {
        "baseline": (baseline_mae, baseline_rmse),
        "ridge": (ridge_mae, ridge_rmse),
        "mlp": (mlp_mae, mlp_rmse),
        "lstm": (lstm_mae, lstm_rmse),
        "cnn": (cnn_mae, cnn_rmse),
    }


def main():
    # 1) load raw tensor
    X_raw, Y, dates, mean, std = load_dataset()
    print("Raw X shape:", X_raw.shape)

    # 2) decompose
    cp_res, tucker_res = run_all_decompositions(X_raw)

    print("\nCP reconstruction errors:")
    for r, (_, e) in cp_res.items():
        print("  rank", r, "->", e)

    print("\nTucker reconstruction errors:")
    for ranks, (_, e) in tucker_res.items():
        print("  ranks", ranks, "->", e)

    # 3) windows for raw data
    X_raw_win, y_raw = build_windows_from_tensor(X_raw, Y)
    results = {}
    results["raw"] = evaluate_setting("raw", X_raw_win, y_raw)

    # 4) windows for CP
    for r, (X_cp, err) in cp_res.items():
        X_win, y = build_windows_from_tensor(X_cp, Y)
        name = f"cp_rank_{r}"
        results[name] = evaluate_setting(name, X_win, y)

    # 5) windows for Tucker
    for ranks, (X_tucker, err) in tucker_res.items():
        X_win, y = build_windows_from_tensor(X_tucker, Y)
        name = f"tucker_{ranks[0]}_{ranks[1]}_{ranks[2]}"
        results[name] = evaluate_setting(name, X_win, y)

    print("\n=== Summary (test MAE/RMSE) ===")
    pprint(results)

    # ---------- SAVE NUMERIC RESULTS ----------

    os.makedirs("results", exist_ok=True)

    # 5a) Save decomposition summary
    decomp_records = []
    for r, (_, err) in cp_res.items():
        decomp_records.append({
            "method": "cp",
            "rank": r,
            "rank1": r,
            "rank2": None,
            "rank3": None,
            "rel_error": float(err),
        })

    for (r1, r2, r3), (_, err) in tucker_res.items():
        decomp_records.append({
            "method": "tucker",
            "rank": None,
            "rank1": r1,
            "rank2": r2,
            "rank3": r3,
            "rel_error": float(err),
        })

    df_decomp = pd.DataFrame(decomp_records)
    df_decomp.to_csv("results/decomposition_summary.csv", index=False)
    print("Saved results/decomposition_summary.csv")

    # 5b) Save model performance summary
    perf_records = []
    for setting_name, model_dict in results.items():
        for model_name, (mae, rmse) in model_dict.items():
            perf_records.append({
                "setting": setting_name,
                "model": model_name,
                "mae": float(mae),
                "rmse": float(rmse),
            })

    df_perf = pd.DataFrame(perf_records)
    df_perf.to_csv("results/model_performance.csv", index=False)
    print("Saved results/model_performance.csv")


if __name__ == "__main__":
    main()
