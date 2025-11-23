import os
import numpy as np
import matplotlib.pyplot as plt

from preprocess import load_dataset
from decompose import cp_decompose, tucker_decompose
from windows import tensor_to_timeseries, make_sliding_windows, time_split
from models import (
    persistence_baseline,
    train_ridge,
    train_mlp,
    train_lstm,
    train_cnn,
)
from config import WINDOW, HORIZON, CP_RANKS, TUCKER_RANKS


# ----------------- Utility: dirs ----------------- #

RESULTS_DIR = "results"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots", "demo")


def ensure_dirs():
    os.makedirs(PLOTS_DIR, exist_ok=True)


# ----------------- Part 1: Decomposition vs original ----------------- #

def plot_decomposition_example():
    """
    Take one example day, run CP + Tucker on that single day tensor,
    and plot original vs reconstructed for one feature.
    """

    print("Loading dataset for decomposition demo...")
    X, Y, dates, mean, std = load_dataset()  # X: (D, T, F), normalized
    D, T, F = X.shape
    print(f"X shape: {X.shape}, Y shape: {Y.shape}")

    # Choose which day and feature to visualize
    example_day_idx = min(3, D - 1)   # e.g. 4th day or last if fewer days
    feature_idx = 0                   # 0 = first feature (e.g. wind speed)

    X_day = X[example_day_idx:example_day_idx + 1]  # shape (1, T, F)
    print(f"Using day index {example_day_idx}, date = {dates[example_day_idx]}")

    # Choose ranks for demo (take some "middle" ones from config)
    cp_rank_demo = CP_RANKS[len(CP_RANKS) // 2]          # e.g. rank 10
    tucker_rank_demo = TUCKER_RANKS[len(TUCKER_RANKS) // 2]  # e.g. (20, 40, 3)

    print(f"Running CP decomposition on single day with rank={cp_rank_demo}...")
    X_cp_rec, cp_err = cp_decompose(X_day, rank=cp_rank_demo)

    print(f"Running Tucker decomposition on single day with ranks={tucker_rank_demo}...")
    X_tk_rec, tk_err = tucker_decompose(X_day, ranks=tucker_rank_demo)

    # Denormalize for plotting (use same mean/std)
    # mean/std were computed over all days, but apply to this day as well
    X_day_denorm = X_day * std + mean
    X_cp_denorm = X_cp_rec * std + mean
    X_tk_denorm = X_tk_rec * std + mean

    orig = X_day_denorm[0, :, feature_idx]
    cp_rec = X_cp_denorm[0, :, feature_idx]
    tk_rec = X_tk_denorm[0, :, feature_idx]

    t_axis = np.arange(T)

    plt.figure(figsize=(10, 4))
    plt.plot(t_axis, orig, label="Original", linewidth=2)
    plt.plot(t_axis, cp_rec, label=f"CP rank={cp_rank_demo}", linestyle="--")
    plt.plot(t_axis, tk_rec, label=f"Tucker {tucker_rank_demo}", linestyle=":")
    plt.xlabel("Time slot (10 min)")
    plt.ylabel(f"Feature {feature_idx} (denormalized units)")
    plt.title(f"Original vs CP/Tucker reconstruction\nDay {dates[example_day_idx]} | CP err={cp_err:.3f}, Tucker err={tk_err:.3f}")
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(PLOTS_DIR, "decomposition_example_day.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print("Saved:", out_path)


# ----------------- Part 2: Forecasting predictions vs truth ----------------- #

def plot_forecasting_example():
    """
    Train all models on the raw tensor (no decomposition),
    then plot true vs predicted on a slice of the test set.
    """

    print("Loading dataset for forecasting demo...")
    X, Y, dates, mean, std = load_dataset()  # X: (D, T, F), normalized
    print(f"X shape: {X.shape}, Y shape: {Y.shape}")

    # Build sliding windows on raw data
    from windows import tensor_to_timeseries, make_sliding_windows, time_split

    X_ts, y_ts = tensor_to_timeseries(X, Y)
    X_win, y_win = make_sliding_windows(X_ts, y_ts, WINDOW, HORIZON)
    print(f"Windowed data: X_win={X_win.shape}, y_win={y_win.shape}")

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = time_split(X_win, y_win)
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # -------- Baseline (persistence) -------- #

    print("Computing persistence baseline...")
    # Persistence baseline: predict next value as last feature of the window
    y_pred_baseline = X_test[:, -1, -1]

    # -------- Ridge -------- #

    print("Training Ridge...")
    ridge_model, ridge_val_mae, ridge_val_rmse = train_ridge(X_train, y_train, X_val, y_val)
    print(f"Ridge val MAE={ridge_val_mae:.3f}, RMSE={ridge_val_rmse:.3f}")
    s_test, w, f = X_test.shape
    X_test_flat = X_test.reshape(s_test, w * f)
    y_pred_ridge = ridge_model.predict(X_test_flat)

    # -------- MLP -------- #

    print("Training MLP...")
    mlp_model, mlp_val_mae, mlp_val_rmse = train_mlp(X_train, y_train, X_val, y_val)
    print(f"MLP val MAE={mlp_val_mae:.3f}, RMSE={mlp_val_rmse:.3f}")
    y_pred_mlp = mlp_model.predict(X_test_flat)

    # -------- LSTM -------- #

    print("Training LSTM (reduced epochs for demo)...")
    lstm_model, _ = train_lstm(
        X_train, y_train, X_val, y_val,
        epochs=40,       # less than full experiments to keep it faster
        batch_size=64,
    )
    y_pred_lstm = lstm_model.predict(X_test, verbose=0).ravel()

    # -------- CNN -------- #

    print("Training CNN (reduced epochs for demo)...")
    cnn_model, _ = train_cnn(
        X_train, y_train, X_val, y_val,
        epochs=30,
        batch_size=64,
    )
    y_pred_cnn = cnn_model.predict(X_test, verbose=0).ravel()

    # -------- Choose a slice of the test set to visualize -------- #

    n_test = len(y_test)
    print(f"Total test samples: {n_test}")

    # Take a contiguous slice from the start of test set
    slice_len = min(200, n_test)
    start_idx = 0
    end_idx = start_idx + slice_len

    y_true_slice = y_test[start_idx:end_idx]
    x_axis = np.arange(slice_len)

    y_baseline_slice = y_pred_baseline[start_idx:end_idx]
    y_ridge_slice = y_pred_ridge[start_idx:end_idx]
    y_mlp_slice = y_pred_mlp[start_idx:end_idx]
    y_lstm_slice = y_pred_lstm[start_idx:end_idx]
    y_cnn_slice = y_pred_cnn[start_idx:end_idx]

    plt.figure(figsize=(12, 5))
    plt.plot(x_axis, y_true_slice, label="True power", linewidth=2)
    plt.plot(x_axis, y_baseline_slice, label="Persistence", alpha=0.7)
    plt.plot(x_axis, y_ridge_slice, label="Ridge", alpha=0.7)
    plt.plot(x_axis, y_mlp_slice, label="MLP", alpha=0.7)
    plt.plot(x_axis, y_lstm_slice, label="LSTM", alpha=0.7)
    plt.plot(x_axis, y_cnn_slice, label="CNN", alpha=0.7)

    plt.xlabel("Test sample index (chronological)")
    plt.ylabel("LV ActivePower (kW)")
    plt.title("Forecasting: True vs model predictions (test slice)")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()

    out_path = os.path.join(PLOTS_DIR, "forecasting_predictions_test_slice.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print("Saved:", out_path)


# ----------------- Main ----------------- #

def main():
    ensure_dirs()
    plot_decomposition_example()
    plot_forecasting_example()


if __name__ == "__main__":
    main()
