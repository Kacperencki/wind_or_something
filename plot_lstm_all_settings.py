import os
import numpy as np
import matplotlib.pyplot as plt

from preprocess import load_dataset
from decompose import run_all_decompositions
from windows import tensor_to_timeseries, make_sliding_windows, time_split
from models import train_lstm
from config import WINDOW, HORIZON

# -------------------------------------------------------------------
# Settings
# -------------------------------------------------------------------

RESULTS_DIR = "results"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots", "lstm_predictions")

PLOT_SAMPLES = 200    # how many test points to show on plot
LSTM_EPOCHS = 100      # reduce if you want it faster (e.g. 15)
LSTM_BATCH = 64       # can adjust if memory issues


def ensure_dirs():
    os.makedirs(PLOTS_DIR, exist_ok=True)


def build_windows_from_tensor(X, Y):
    """
    X: (D, T, F)
    Y: (D, T)
    → X_win: (N, W, F), y_win: (N,)
    """
    X_ts, y_ts = tensor_to_timeseries(X, Y)
    X_win, y_win = make_sliding_windows(X_ts, y_ts, WINDOW, HORIZON)
    return X_win, y_win


def plot_lstm_for_setting(setting_name, X_tensor, Y):
    """
    For a given tensor X_tensor (raw or decomposed),
    train ONE LSTM and plot predictions vs true on a slice of test set.
    """
    print(f"\n=== Setting: {setting_name} ===")

    X_win, y_win = build_windows_from_tensor(X_tensor, Y)
    print("  Windowed:", X_win.shape, y_win.shape)

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = time_split(X_win, y_win)
    print("  Train/Val/Test:", X_train.shape, X_val.shape, X_test.shape)

    # Train LSTM
    print(f"  Training LSTM: epochs={LSTM_EPOCHS}, batch_size={LSTM_BATCH}...")
    lstm_model, _ = train_lstm(
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=LSTM_EPOCHS,
        batch_size=LSTM_BATCH,
    )

    # Predict on test
    y_pred = lstm_model.predict(X_test, verbose=0).ravel()

    # Select a slice to plot
    n_test = len(y_test)
    n_plot = min(PLOT_SAMPLES, n_test)
    start_idx = 0
    end_idx = start_idx + n_plot

    y_true_slice = y_test[start_idx:end_idx]
    y_pred_slice = y_pred[start_idx:end_idx]
    x_axis = np.arange(n_plot)

    # Plot
    plt.figure(figsize=(12, 5))
    plt.plot(x_axis, y_true_slice, label="True", linewidth=2)
    plt.plot(x_axis, y_pred_slice, label="LSTM", alpha=0.8)

    plt.xlabel("Test sample index (chronological)")
    plt.ylabel("LV ActivePower (kW)")
    plt.title(f"{setting_name}: true vs LSTM prediction (test slice)")
    plt.legend()
    plt.tight_layout()

    fname = f"{setting_name}_lstm_predictions.png"
    out_path = os.path.join(PLOTS_DIR, fname)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print("  Saved plot:", out_path)


def main():
    ensure_dirs()

    # 1) Load raw tensor
    print("Loading dataset...")
    X_raw, Y, dates, mean, std = load_dataset()
    print("Raw X:", X_raw.shape, "Y:", Y.shape)

    # 2) Run all decompositions on full tensor
    print("\nRunning CP + Tucker decompositions...")
    cp_res, tucker_res = run_all_decompositions(X_raw)

    # Collect all settings: name -> X_tensor
    settings = {"raw": X_raw}

    for r, (X_cp, err) in cp_res.items():
        name = f"cp_rank_{r}"
        print(f"Adding setting {name}, rel_error={err:.4f}")
        settings[name] = X_cp

    for (r1, r2, r3), (X_tk, err) in tucker_res.items():
        name = f"tucker_{r1}_{r2}_{r3}"
        print(f"Adding setting {name}, rel_error={err:.4f}")
        settings[name] = X_tk

    # 3) For each setting, train one LSTM and plot predictions
    for name, X_tensor in settings.items():
        plot_lstm_for_setting(name, X_tensor, Y)


if __name__ == "__main__":
    main()
