# analyze_results.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


RESULTS_DIR = "results"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")


def ensure_dirs():
    os.makedirs(PLOTS_DIR, exist_ok=True)


def plot_cp_error_vs_rank():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "decomposition_summary.csv"))
    df_cp = df[df["method"] == "cp"].copy()
    df_cp = df_cp.sort_values("rank")

    plt.figure()
    plt.plot(df_cp["rank"], df_cp["rel_error"], marker="o")
    plt.xlabel("CP rank R")
    plt.ylabel("Relative reconstruction error")
    plt.title("CP reconstruction error vs rank")
    plt.grid(True)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "cp_error_vs_rank.png")
    plt.savefig(path, dpi=200)
    plt.close()
    print("Saved", path)


def plot_tucker_error_vs_rank():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "decomposition_summary.csv"))
    df_tk = df[df["method"] == "tucker"].copy()

    # Use rank1 (day rank) as x-axis; annotate others
    df_tk = df_tk.sort_values("rank1")

    plt.figure()
    x = np.arange(len(df_tk))
    plt.plot(x, df_tk["rel_error"], marker="o")
    labels = [f"({r1},{r2},{r3})" for r1, r2, r3 in zip(df_tk["rank1"], df_tk["rank2"], df_tk["rank3"])]
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Relative reconstruction error")
    plt.title("Tucker reconstruction error vs ranks")
    plt.grid(True)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "tucker_error_vs_rank.png")
    plt.savefig(path, dpi=200)
    plt.close()
    print("Saved", path)


def plot_model_mae_by_setting(model_name: str):
    df = pd.read_csv(os.path.join(RESULTS_DIR, "model_performance.csv"))
    df_m = df[df["model"] == model_name].copy()

    # Sort by MAE (ascending = best first)
    df_m = df_m.sort_values("mae")

    plt.figure(figsize=(10, 4))
    bars = plt.bar(df_m["setting"], df_m["mae"], color="steelblue")

    # Annotate values on top
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.1f}",
            ha="center",
            va="bottom",
            fontsize=8
        )

    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Test MAE")
    plt.title(f"{model_name.upper()} test MAE by setting (sorted best → worst)")
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, f"{model_name}_mae_by_setting.png")
    plt.savefig(path, dpi=200)
    plt.close()
    print("Saved", path)


def main():
    ensure_dirs()
    plot_cp_error_vs_rank()
    plot_tucker_error_vs_rank()
    for m in ["baseline", "ridge", "mlp", "lstm", "cnn"]:
        plot_model_mae_by_setting(m)


if __name__ == "__main__":
    main()
