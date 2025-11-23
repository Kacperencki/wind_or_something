import os
import numpy as np
import matplotlib.pyplot as plt

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


# -------------------------------------------------------------------
# Settings
# -------------------------------------------------------------------

RESULTS_DIR = "results"
PLOTS_ROOT = os.path.join(RESULTS_DIR, "plots", "predictions_per_model")

# how many consecutive test points to visualize
PLOT_SAMPLES = 200


def ensure_dirs():
    os.makedirs(PLOTS_ROOT, exist_ok=True)


def build_windows_from_tensor(X, Y):
    """
    X: (D, T, F)
    Y: (D, T)
    → X_win: (N, W, F), y_win: (N,)
    """
    X_ts, y_ts = tensor_to_timeseries(X, Y)
    X_win, y_win = make_sliding_windows(X_ts, y_ts, WINDOW, HORIZON)
    return X_win, y_win


def train_and_plot_setting(setting_name, X_tensor, Y):
    """
    For a given tensor X_tensor (raw or decomposed):
      1) build sliding windows
      2) split into train/val/test
      3) train baseline, Ridge, MLP, LSTM, CNN
      4) compute test predictions
      5) save per-model plots + one combined plot
    """
    print(f"\n=== Setting: {setting_name} ===")

    # 1) windows
    X_win, y_win = build_windows_from_tensor(X_tensor, Y)
    print("  Windowed:", X_win.shape, y_win.shape)

    # 2) split
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = time_split(X_win, y_win)
    print("  Train/Val/Test:", X_train.shape, X_val.shape, X_test.shape)

    # ----------------------------------------------------------------
    # BASELINE (no training)
    # ----------------------------------------------------------------
    print("  Baseline (persistence)...")
    baseline_mae, baseline_rmse = persistence_baseline(X_test, y_test)
    print(f"    Baseline MAE={baseline_mae:.2f}, RMSE={baseline_rmse:.2f}")
    y_pred_baseline = X_test[:, -1, -1]  # last feature as next target

    # ----------------------------------------------------------------
    # RIDGE
    # ----------------------------------------------------------------
    print("  Training Ridge...")
    ridge_model, ridge_val_mae, ridge_val_rmse = train_ridge(X_train, y_train, X_val, y_val)
    print(f"    Ridge val MAE={ridge_val_mae:.2f}, RMSE={ridge_val_rmse:.2f}")
    ridge_mae, ridge_rmse = test_ridge(ridge_model, X_test, y_test)
    print(f"    Ridge test MAE={ridge_mae:.2f}, RMSE={ridge_rmse:.2f}")

    s_test, w, f = X_test.shape
    X_test_flat = X_test.reshape(s_test, w * f)
    y_pred_ridge = ridge_model.predict(X_test_flat)

    # ----------------------------------------------------------------
    # MLP
    # ----------------------------------------------------------------
    print("  Training MLP...")
    mlp_model, mlp_val_mae, mlp_val_rmse = train_mlp(X_train, y_train, X_val, y_val)
    print(f"    MLP val MAE={mlp_val_mae:.2f}, RMSE={mlp_val_rmse:.2f}")
    mlp_mae, mlp_rmse = test_mlp(mlp_model, X_test, y_test)
    print(f"    MLP test MAE={mlp_mae:.2f}, RMSE={mlp_rmse:.2f}")

    y_pred_mlp = mlp_model.predict(X_test_flat)

    # ----------------------------------------------------------------
    # LSTM
    # ----------------------------------------------------------------
    print("  Training LSTM (default epochs, early stopping)...")
    lstm_model, _ = train_lstm(X_train, y_train, X_val, y_val)
    lstm_mae, lstm_rmse = test_lstm(lstm_model, X_test, y_test)
    print(f"    LSTM test MAE={lstm_mae:.2f}, RMSE={lstm_rmse:.2f}")

    y_pred_lstm = lstm_model.predict(X_test, verbose=0).ravel()

    # ----------------------------------------------------------------
    # CNN
    # ----------------------------------------------------------------
    print("  Training CNN (default epochs, early stopping)...")
    cnn_model, _ = train_cnn(X_train, y_train, X_val, y_val)
    cnn_mae, cnn_rmse = test_cnn(cnn_model, X_test, y_test)
    print(f"    CNN test MAE={cnn_mae:.2f}, RMSE={cnn_rmse:.2f}")

    y_pred_cnn = cnn_model.predict(X_test, verbose=0).ravel()

    # ----------------------------------------------------------------
    # Slice for plotting
    # ----------------------------------------------------------------
    n_test = len(y_test)
    n_plot = min(PLOT_SAMPLES, n_test)
    start_idx = 0
    end_idx = start_idx + n_plot

    x_axis = np.arange(n_plot)
    y_true_slice = y_test[start_idx:end_idx]

    preds = {
        "baseline": y_pred_baseline[start_idx:end_idx],
        "ridge":    y_pred_ridge[start_idx:end_idx],
        "mlp":      y_pred_mlp[start_idx:end_idx],
        "lstm":     y_pred_lstm[start_idx:end_idx],
        "cnn":      y_pred_cnn[start_idx:end_idx],
    }

    # ----------------------------------------------------------------
    # Create directory for this setting
    # ----------------------------------------------------------------
    setting_dir = os.path.join(PLOTS_ROOT, setting_name)
    os.makedirs(setting_dir, exist_ok=True)

    # ----------------------------------------------------------------
    # Per-model plots
    # ----------------------------------------------------------------
    for model_name, y_pred_slice in preds.items():
        plt.figure(figsize=(10, 4))
        plt.plot(x_axis, y_true_slice, label="True", linewidth=2)
        plt.plot(x_axis, y_pred_slice, label=model_name.upper(), linewidth=1.5, alpha=0.8)

        plt.xlabel("Test sample index (chronological)")
        plt.ylabel("LV ActivePower (kW)")
        plt.title(f"{setting_name} – {model_name.upper()} vs True")
        plt.legend()
        plt.tight_layout()

        out_path = os.path.join(setting_dir, f"{model_name}.png")
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"  Saved {model_name} plot:", out_path)

    # ----------------------------------------------------------------
    # Combined plot (all models + true)
    # ----------------------------------------------------------------
    plt.figure(figsize=(12, 5))
    plt.plot(x_axis, y_true_slice, label="True", linewidth=2)

    # Style: thinner lines + alpha to reduce clutter
    for model_name, y_pred_slice in preds.items():
        if model_name == "lstm":
            plt.plot(x_axis, y_pred_slice, label="LSTM", linewidth=1.8, alpha=0.9)
        else:
            plt.plot(x_axis, y_pred_slice, label=model_name.upper(), linewidth=1.2, alpha=0.7)

    plt.xlabel("Test sample index (chronological)")
    plt.ylabel("LV ActivePower (kW)")
    plt.title(f"{setting_name} – all models vs True")
    plt.legend(fontsize=8)
    plt.tight_layout()

    combined_path = os.path.join(setting_dir, "all_models.png")
    plt.savefig(combined_path, dpi=200)
    plt.close()
    print("  Saved combined plot:", combined_path)


def main():
    ensure_dirs()

    # 1) load raw tensor
    print("Loading dataset...")
    X_raw, Y, dates, mean, std = load_dataset()
    print("Raw X shape:", X_raw.shape, "Y shape:", Y.shape)

    # 2) run CP + Tucker decompositions
    print("\nRunning CP + Tucker decompositions...")
    cp_res, tucker_res = run_all_decompositions(X_raw)

    # collect all settings
    settings = {"raw": X_raw}

    for r, (X_cp, err) in cp_res.items():
        name = f"cp_rank_{r}"
        print(f"Adding setting {name}, rel_error={err:.4f}")
        settings[name] = X_cp

    for (r1, r2, r3), (X_tk, err) in tucker_res.items():
        name = f"tucker_{r1}_{r2}_{r3}"
        print(f"Adding setting {name}, rel_error={err:.4f}")
        settings[name] = X_tk

    # 3) train + plot for each setting
    for name, X_tensor in settings.items():
        train_and_plot_setting(name, X_tensor, Y)


if __name__ == "__main__":
    main()
