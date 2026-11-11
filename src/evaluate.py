"""Metrics, fit-condition labelling and plots."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def fit_condition(r2_train: float, r2_test: float) -> str:
    if r2_train < 0.6 and r2_test < 0.6:
        return "Underfitting"
    if r2_train - r2_test > 0.15:
        return "Overfitting"
    return "Best Fitting"


def evaluate(name, y_train, pred_train, y_test, pred_test) -> dict:
    r2_tr = r2_score(y_train, pred_train)
    r2_te = r2_score(y_test, pred_test)
    mse_te = mean_squared_error(y_test, pred_test)
    return {
        "Model": name,
        "Train_R2": round(r2_tr, 4),
        "Test_R2": round(r2_te, 4),
        "Test_MAE": round(mean_absolute_error(y_test, pred_test), 4),
        "Test_MSE": round(mse_te, 4),
        "Test_RMSE": round(float(np.sqrt(mse_te)), 4),
        "Condition": fit_condition(r2_tr, r2_te),
    }


def parity_plot(name, y_train, pred_train, y_test, pred_test, out_path: Path):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(pred_train, y_train, alpha=0.75, label=f"{name} train")
    ax.scatter(pred_test, y_test, alpha=0.75, marker="s", label=f"{name} test")
    lo = min(np.min(y_train), np.min(y_test), np.min(pred_train), np.min(pred_test))
    hi = max(np.max(y_train), np.max(y_test), np.max(pred_train), np.max(pred_test))
    ax.plot([lo, hi], [lo, hi], "k-", lw=2, label="y = x")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{name} - Predicted vs Actual")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def comparison_plot(results_df, out_path: Path):
    colors = {"Best Fitting": "#2ecc71", "Overfitting": "#e74c3c", "Underfitting": "#f39c12"}
    df = results_df.sort_values("Test_R2")
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(df.index, df["Test_R2"], color=[colors[c] for c in df["Condition"]], edgecolor="k")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Test R²")
    ax.set_title("Test R² across regression models (colour = fit condition)")
    for i, v in enumerate(df["Test_R2"]):
        ax.text(max(v, 0) + 0.01, i, f"{v:.4f}", va="center", fontsize=8)
    from matplotlib.patches import Patch

    ax.legend(handles=[Patch(color=c, label=k) for k, c in colors.items()], loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
