"""
viz.py - Visualization utilities for the wind-turbine tensor decomposition project.

Assumptions:
- `run.py` has already been executed.
- It produced at least:
    results/model_performance.csv
- Optionally:
    results/decomposition_summary.csv
    results/preds_<setting>_<model>.npz with arrays:
        - y_test
        - y_pred
"""

import os
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


RESULTS_DIR = "results"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NICE_LABELS = {
    # RAW
    ("raw", "persistence"): "Persistence (y(t-1))",
    ("raw", "ridge"):       "Raw + Ridge",
    ("raw", "lstm"):        "Raw + LSTM",

    # PCA
    ("pca_k3", "ridge"): "PCA(k=3) + Ridge",
    ("pca_k3", "lstm"):  "PCA(k=3) + LSTM",
    ("pca_k5", "ridge"): "PCA(k=5) + Ridge",
    ("pca_k5", "lstm"):  "PCA(k=5) + LSTM",

    # CP
    ("cp_latent_r12", "ridge"): "CP(r=12) + Ridge",
    ("cp_latent_r12", "lstm"):  "CP(r=12) + LSTM",

    # Tucker m3
    ("tucker_latent_m3_50_24_5", "ridge"): "Tucker m3 (50,24,5) + Ridge",
    ("tucker_latent_m3_50_24_5", "lstm"):  "Tucker m3 (50,24,5) + LSTM",
    ("tucker_latent_m3_80_24_7", "ridge"): "Tucker m3 (80,24,7) + Ridge",
    ("tucker_latent_m3_80_24_7", "lstm"):  "Tucker m3 (80,24,7) + LSTM",

    # Tucker core
    ("tucker_latent_core_50_24_5", "ridge"): "Tucker core (50,24,5) + Ridge",
    ("tucker_latent_core_50_24_5", "lstm"):  "Tucker core (50,24,5) + LSTM",
    ("tucker_latent_core_80_24_7", "ridge"): "Tucker core (80,24,7) + Ridge",
    ("tucker_latent_core_80_24_7", "lstm"):  "Tucker core (80,24,7) + LSTM",
}


def pretty_label(setting: str, model: str) -> str:
    key = (setting, model)
    if key in NICE_LABELS:
        return NICE_LABELS[key]
    return f"{setting} + {model}"


def _check_file(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Expected file not found: {path}")


# ---------------------------------------------------------------------------
# 1. Decomposition quality: CP vs Tucker reconstruction error
# ---------------------------------------------------------------------------

def plot_decomposition_errors(
    path: str = os.path.join(RESULTS_DIR, "decomposition_summary.csv")
) -> None:
    _check_file(path)
    df = pd.read_csv(path)

    df_cp = df[df["method"] == "cp"].copy()
    df_tk = df[df["method"] == "tucker"].copy()

    plt.figure()
    if not df_cp.empty:
        plt.plot(df_cp["rank"], df_cp["rel_error"], marker="o", label="CP (rank R)")
    if not df_tk.empty:
        plt.plot(df_tk["rank3"], df_tk["rel_error"], marker="s", label="Tucker (R_F)")

    plt.xlabel("Rank (R for CP, R_F for Tucker)")
    plt.ylabel("Relative reconstruction error")
    plt.title("CP vs Tucker reconstruction error on daily tensor")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 2. Global model performance: bar charts of RMSE / MAE
# ---------------------------------------------------------------------------

def plot_model_performance(
    path: str = os.path.join(RESULTS_DIR, "model_performance.csv"),
    metric: str = "rmse",
    model_filter: Optional[str] = None,
    sort_ascending: bool = True,
) -> None:
    """
    Bar chart of performance for all (setting, model) pairs.

    metric : "rmse" or "mae"
    model_filter : if given (e.g. "lstm" or "ridge"), only that model type is shown.
    """
    _check_file(path)
    df = pd.read_csv(path)

    if metric not in {"rmse", "mae"}:
        raise ValueError("metric must be 'rmse' or 'mae'")

    if model_filter is not None:
        df = df[df["model"] == model_filter].copy()

    df["label"] = [pretty_label(s, m) for s, m in zip(df["setting"], df["model"])]
    df = df.sort_values(metric, ascending=sort_ascending)

    plt.figure(figsize=(10, 4))
    plt.bar(df["label"], df[metric])
    plt.ylabel(metric.upper())
    plt.title(f"Test {metric.upper()} for all models"
              + ("" if model_filter is None else f" ({model_filter})"))
    plt.xticks(rotation=30, ha="right")

    if "persistence" in df["model"].values:
        pers_vals = df[df["model"] == "persistence"][metric]
        if not pers_vals.empty:
            val = float(pers_vals.iloc[0])
            plt.axhline(val, linestyle="--", label="Persistence")
            plt.legend()

    plt.tight_layout()
    plt.show()


def plot_main_models_rmse(
    path: str = os.path.join(RESULTS_DIR, "model_performance.csv")
) -> None:
    """
    Focused RMSE comparison of the most important LSTM models:
    - Persistence
    - Raw + LSTM
    - PCA(k=5) + LSTM
    - CP(r=12) + LSTM
    - Tucker core (50,24,5) + LSTM
    - Tucker core (80,24,7) + LSTM
    """
    _check_file(path)
    df = pd.read_csv(path)

    keep = [
        ("raw", "persistence"),
        ("raw", "lstm"),
        ("pca_k5", "lstm"),
        ("cp_latent_r12", "lstm"),
        ("tucker_latent_core_50_24_5", "lstm"),
        ("tucker_latent_core_80_24_7", "lstm"),
    ]

    rows = []
    for setting, model in keep:
        row = df[(df["setting"] == setting) & (df["model"] == model)]
        if not row.empty:
            r = row.iloc[0]
            rows.append({
                "label": pretty_label(setting, model),
                "rmse": r["rmse"],
            })

    plot_df = pd.DataFrame(rows)
    plot_df = plot_df.sort_values("rmse", ascending=True)

    plt.figure(figsize=(10, 4))
    plt.bar(plot_df["label"], plot_df["rmse"])
    plt.ylabel("RMSE")
    plt.title("Forecast performance of main LSTM models (test RMSE)")
    plt.xticks(rotation=25, ha="right")

    pers = df[(df["setting"] == "raw") & (df["model"] == "persistence")]
    if not pers.empty:
        rmse_pers = pers["rmse"].iloc[0]
        plt.axhline(rmse_pers, linestyle="--", label="Persistence")
        plt.legend()
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 3. True vs predicted: time-series windows
# ---------------------------------------------------------------------------

def load_predictions(
    setting: str,
    model: str = "lstm",
    results_dir: str = RESULTS_DIR,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load y_test and y_pred from results/preds_<setting>_<model>.npz.

    Works for model in {"persistence", "ridge", "lstm"} provided
    run.py saved files with this naming convention.
    """
    fname = f"preds_{setting}_{model}.npz"
    path = os.path.join(results_dir, fname)
    _check_file(path)
    data = np.load(path)
    y_test = data["y_test"].ravel()
    y_pred = data["y_pred"].ravel()
    return y_test, y_pred


def plot_timeseries_window(
    settings: List[str],
    model: str = "lstm",
    start: int = 0,
    length: int = 400,
    results_dir: str = RESULTS_DIR,
) -> None:
    """
    Plot a window of the test series for several representations
    using the same model type (ridge / lstm / persistence).

    settings : list of setting names (e.g. ["raw", "tucker_latent_core_80_24_7"])
    model    : "lstm", "ridge", or "persistence"
    """
    if not settings:
        raise ValueError("settings list must not be empty")

    y_test, y_pred_first = load_predictions(settings[0], model, results_dir)
    end = min(start + length, len(y_test))
    t = np.arange(end - start)

    plt.figure(figsize=(10, 4))
    plt.plot(t, y_test[start:end], label="True")
    plt.plot(t, y_pred_first[start:end], label=pretty_label(settings[0], model))

    for setting in settings[1:]:
        _, y_pred = load_predictions(setting, model, results_dir)
        plt.plot(t, y_pred[start:end], label=pretty_label(setting, model))

    plt.xlabel("Time index (test, 10-min steps)")
    plt.ylabel("Power output")
    plt.title(f"Forecasts on test set (example window) – model={model}")
    plt.legend()
    plt.tight_layout()
    plt.show()

def save_timeseries_per_model(
    settings: List[str],
    model: str = "lstm",
    start: int = 0,
    length: int = 400,
    results_dir: str = RESULTS_DIR,
    out_dir: str = "figures",
) -> None:
    """
    For each setting, create a separate PNG with:
        - true y_test
        - predictions of (setting, model)

    Example:
        save_timeseries_per_model(
            ["raw", "tucker_latent_core_80_24_7"],
            model="lstm",
            start=200,
            length=400,
        )
    """
    os.makedirs(out_dir, exist_ok=True)

    for setting in settings:
        y_test, y_pred = load_predictions(setting, model, results_dir)
        end = min(start + length, len(y_test))
        t = np.arange(end - start)

        plt.figure(figsize=(10, 3))
        plt.plot(t, y_test[start:end], label="True")
        plt.plot(t, y_pred[start:end], label=pretty_label(setting, model))

        plt.xlabel("Time index (test, 10-min steps)")
        plt.ylabel("Power output")
        plt.title(f"{pretty_label(setting, model)} – time series (test window)")
        plt.legend()
        plt.tight_layout()

        fname = os.path.join(out_dir, f"ts_{setting}_{model}.png")
        plt.savefig(fname, dpi=300, bbox_inches="tight")
        plt.close()


# ---------------------------------------------------------------------------
# 4. True vs predicted: scatter plots (calibration)
# ---------------------------------------------------------------------------

def plot_scatter_true_vs_pred(
    settings: List[str],
    model: str = "lstm",
    max_points: Optional[int] = 5000,
    results_dir: str = RESULTS_DIR,
) -> None:
    """
    Scatter: y_true vs y_pred for several representations using same model.
    settings : list of setting names (e.g. ["raw", "tucker_latent_core_80_24_7"])
    model    : "lstm", "ridge", or "persistence"
    """
    if not settings:
        raise ValueError("settings list must not be empty")

    y_test, _ = load_predictions(settings[0], model, results_dir)
    n = len(y_test)

    if max_points is not None and max_points < n:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, size=max_points, replace=False)
    else:
        idx = np.arange(n)

    plt.figure(figsize=(6, 6))

    for setting in settings:
        y_true, y_pred = load_predictions(setting, model, results_dir)
        y_true_s = y_true[idx]
        y_pred_s = y_pred[idx]
        plt.scatter(y_true_s, y_pred_s, alpha=0.3, label=pretty_label(setting, model))

    min_v = float(y_test.min())
    max_v = float(y_test.max())
    plt.plot([min_v, max_v], [min_v, max_v])

    plt.xlabel("True power")
    plt.ylabel("Predicted power")
    plt.title(f"True vs predicted power on test set – model={model}")
    plt.legend()
    plt.tight_layout()
    plt.show()



def save_scatter_per_model(
    settings: List[str],
    model: str = "lstm",
    max_points: Optional[int] = 5000,
    results_dir: str = RESULTS_DIR,
    out_dir: str = "figures",
) -> None:
    """
    For each setting, create a separate PNG:
        x = true, y = predicted for (setting, model)
    """
    os.makedirs(out_dir, exist_ok=True)

    for setting in settings:
        y_true, y_pred = load_predictions(setting, model, results_dir)
        n = len(y_true)

        if max_points is not None and max_points < n:
            rng = np.random.default_rng(0)
            idx = rng.choice(n, size=max_points, replace=False)
            y_true_s = y_true[idx]
            y_pred_s = y_pred[idx]
        else:
            y_true_s = y_true
            y_pred_s = y_pred

        plt.figure(figsize=(5, 5))
        plt.scatter(y_true_s, y_pred_s, alpha=0.3)
        min_v = float(min(y_true_s.min(), y_pred_s.min()))
        max_v = float(max(y_true_s.max(), y_pred_s.max()))
        plt.plot([min_v, max_v], [min_v, max_v])

        plt.xlabel("True power")
        plt.ylabel("Predicted power")
        plt.title(f"{pretty_label(setting, model)} – true vs predicted")
        plt.tight_layout()

        fname = os.path.join(out_dir, f"scatter_{setting}_{model}.png")
        plt.savefig(fname, dpi=300, bbox_inches="tight")
        plt.close()

# ---------------------------------------------------------------------------
# 5. Convenience main (you can also call functions manually from a notebook)
# ---------------------------------------------------------------------------

def main():
    # 1) Decomposition errors
    try:
        plot_decomposition_errors()
    except FileNotFoundError:
        print("decomposition_summary.csv not found; skipping decomposition plot.")

    # 2) Global performance
    try:
        # all models
        plot_model_performance(metric="rmse", model_filter=None)
        # only LSTM
        plot_model_performance(metric="rmse", model_filter="lstm")
        # only Ridge
        plot_model_performance(metric="rmse", model_filter="ridge")
        # focused main LSTM models
        plot_main_models_rmse()
    except FileNotFoundError:
        print("model_performance.csv not found; skipping performance plots.")

    # 3) Save per-model time series (true vs each model), as images
    try:
        settings_important = [
            "raw",
            "pca_k5",
            "cp_latent_r12",
            "tucker_latent_core_50_24_5",
            "tucker_latent_core_80_24_7",
        ]

        # LSTM versions
        save_timeseries_per_model(
            settings_important,
            model="lstm",
            start=200,
            length=400,
            out_dir="figures_lstm_ts",
        )

        # Ridge versions
        save_timeseries_per_model(
            settings_important,
            model="ridge",
            start=200,
            length=400,
            out_dir="figures_ridge_ts",
        )

        # Persistence: only defined for 'raw'
        save_timeseries_per_model(
            ["raw"],
            model="persistence",
            start=200,
            length=400,
            out_dir="figures_persistence_ts",
        )

    except FileNotFoundError:
        print("Prediction .npz files not found; skipping per-model time-series export.")

    # 4) Save per-model scatter plots (true vs predicted), as images
    try:
        settings_important = [
            "raw",
            "pca_k5",
            "cp_latent_r12",
            "tucker_latent_core_50_24_5",
            "tucker_latent_core_80_24_7",
        ]

        save_scatter_per_model(
            settings_important,
            model="lstm",
            max_points=4000,
            out_dir="figures_lstm_scatter",
        )

        save_scatter_per_model(
            settings_important,
            model="ridge",
            max_points=4000,
            out_dir="figures_ridge_scatter",
        )

        save_scatter_per_model(
            ["raw"],
            model="persistence",
            max_points=4000,
            out_dir="figures_persistence_scatter",
        )

    except FileNotFoundError:
        print("Prediction .npz files not found; skipping per-model scatter export.")

if __name__ == "__main__":
    main()
