"""
plots.py
--------
Reusable Matplotlib figures used in both notebooks and reports.
All functions save to FIGURES_DIR (config.py) when ``save=True``.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve

from stroke_prediction.config import FIGURES_DIR


def pr_auc_bar(
    results_df: pd.DataFrame,
    title: str = "PR-AUC Comparison",
    save: bool = True,
    filename: str = "pr_auc_comparison.png",
) -> plt.Figure:
    """
    Horizontal bar chart ranking models by PR-AUC.

    Parameters
    ----------
    results_df : DataFrame with columns ['model', 'test_pr_auc'].
    """
    df  = results_df.sort_values("test_pr_auc", ascending=True)
    fig, ax = plt.subplots(figsize=(9, max(4, len(df) * 0.55)))

    colors = plt.cm.Blues(np.linspace(0.35, 0.85, len(df)))
    bars   = ax.barh(df["model"], df["test_pr_auc"], color=colors, edgecolor="white")

    for bar, val in zip(bars, df["test_pr_auc"]):
        ax.text(
            bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", fontsize=9,
        )

    ax.set_xlabel("PR-AUC", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlim(0, df["test_pr_auc"].max() * 1.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    if save:
        fig.savefig(FIGURES_DIR / filename, dpi=150, bbox_inches="tight")
    return fig


def calibration_curves(
    models: dict[str, tuple],
    y_test: np.ndarray | pd.Series,
    title: str = "Calibration: Before vs After Isotonic Calibration",
    save: bool = True,
    filename: str = "calibration_before_after.png",
) -> plt.Figure:
    """
    Side-by-side calibration curve panels.

    Parameters
    ----------
    models : {model_name: (proba_before, proba_after)}
    y_test : true binary labels
    """
    n    = len(models)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, (name, (before, after)) in zip(axes, models.items()):
        for probs, label, ls in [
            (before, "Before calibration", "--"),
            (after,  "After calibration",  "-"),
        ]:
            frac_pos, mean_pred = calibration_curve(y_test, probs, n_bins=10)
            ax.plot(mean_pred, frac_pos, marker="o", linestyle=ls, label=label)

        ax.plot([0, 1], [0, 1], "k:", alpha=0.5, label="Perfect calibration")
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.set_xlabel("Mean predicted probability", fontsize=9)
        ax.set_ylabel("Fraction of positives", fontsize=9)
        ax.legend(fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.suptitle(title, fontsize=12, fontweight="bold", y=1.03)
    plt.tight_layout()

    if save:
        fig.savefig(FIGURES_DIR / filename, dpi=150, bbox_inches="tight")
    return fig


def feature_importance_bar(
    feature_names: list[str],
    importances: np.ndarray,
    top_n: int = 15,
    title: str = "Top Feature Importances",
    save: bool = True,
    filename: str = "feature_importance.png",
) -> plt.Figure:
    """Horizontal bar chart for feature importances or coefficients."""
    idx = np.argsort(np.abs(importances))[-top_n:]
    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.45)))

    vals   = importances[idx]
    colors = ["#E53935" if v > 0 else "#1E88E5" for v in vals]
    ax.barh([feature_names[i] for i in idx], vals, color=colors, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Coefficient value", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    if save:
        fig.savefig(FIGURES_DIR / filename, dpi=150, bbox_inches="tight")
    return fig
