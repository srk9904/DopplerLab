import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.data.dataset import PATH_TO_ID
from src.features.extraction import MAX_SPEED_MPS

def plot_results(df, model_names, colors, fig_dir, speed_lim, dist_lim,
                 dist_bins, bin_labels, max_dist):
    """Shared plotting routine — same graphs as notebook benchmark."""
    os.makedirs(fig_dir, exist_ok=True)
    path_names = ["straight", "parabola", "bezier"]

    # 1. Confusion matrices
    fig, axes = plt.subplots(1, len(model_names), figsize=(6 * len(model_names), 5))
    if len(model_names) == 1: axes = [axes]
    fig.suptitle("Path Classification — Confusion Matrices", fontsize=13)
    for ax, mname in zip(axes, model_names):
        sub = df[df["model"] == mname]
        conf = np.zeros((3, 3), dtype=int)
        for _, row in sub.iterrows():
            conf[PATH_TO_ID[row["path_gt"]], PATH_TO_ID[row["path_pred"]]] += 1
        acc = 100 * np.trace(conf) / conf.sum()
        row_totals = conf.sum(axis=1)
        col_totals = conf.sum(axis=0)
        ax.imshow(conf, cmap="Blues", vmin=0)
        ax.set_xticks(range(3))
        ax.set_xticklabels([f"{c}\n(n={col_totals[i]})" for i, c in enumerate(path_names)], rotation=20, fontsize=9)
        ax.set_yticks(range(3))
        ax.set_yticklabels([f"{c} (n={row_totals[i]})" for i, c in enumerate(path_names)], fontsize=9)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"{mname}   acc = {acc:.1f}%")
        for r in range(3):
            for c in range(3):
                ax.text(c, r, str(conf[r, c]), ha="center", va="center", fontsize=12,
                        color="white" if conf[r, c] > conf.max() * 0.6 else "black")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "confusion_matrices.png"), dpi=150)
    plt.close()

    # 2. Speed scatter
    fig, axes = plt.subplots(1, len(model_names), figsize=(6 * len(model_names), 4.5))
    if len(model_names) == 1: axes = [axes]
    fig.suptitle("Speed Estimation — GT vs Predicted", fontsize=13)
    for ax, mname in zip(axes, model_names):
        sub = df[df["model"] == mname]
        ax.scatter(sub["speed_gt"], sub["speed_pred"], s=10, alpha=0.45, color=colors[mname])
        ax.plot(speed_lim, speed_lim, "k--", linewidth=1, label="y = x")
        ax.set_xlim(speed_lim); ax.set_ylim(speed_lim)
        ax.set_xlabel("GT speed (m/s)"); ax.set_ylabel("Pred speed (m/s)")
        mae = sub["speed_err"].mean()
        ax.set_title(f"{mname}   MAE = {mae:.2f} m/s")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "speed_scatter.png"), dpi=150)
    plt.close()

    # 3. Distance scatter
    fig, axes = plt.subplots(1, len(model_names), figsize=(6 * len(model_names), 4.5))
    if len(model_names) == 1: axes = [axes]
    fig.suptitle("Distance Estimation — GT vs Predicted", fontsize=13)
    for ax, mname in zip(axes, model_names):
        sub = df[df["model"] == mname]
        ax.scatter(sub["dist_gt"], sub["dist_pred"], s=10, alpha=0.45, color=colors[mname])
        ax.plot(dist_lim, dist_lim, "k--", linewidth=1, label="y = x")
        ax.set_xlim(dist_lim); ax.set_ylim(dist_lim)
        ax.set_xlabel("GT distance (m)"); ax.set_ylabel("Pred distance (m)")
        mae = sub["dist_err"].mean()
        ax.set_title(f"{mname}   MAE = {mae:.2f} m")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "distance_scatter.png"), dpi=150)
    plt.close()

    # 4. MAE comparison
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    fig.suptitle("Mean Absolute Error", fontsize=13)
    for ax, (col, unit) in zip(axes, [("speed_err", "Speed MAE (m/s)"), ("dist_err", "Distance MAE (m)")]):
        vals = [df[df["model"] == m][col].mean() for m in model_names]
        ax.bar(model_names, vals, color=[colors.get(m, 'steelblue') for m in model_names])
        ax.set_ylabel(unit)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "mae_comparison.png"), dpi=150)
    plt.close()
