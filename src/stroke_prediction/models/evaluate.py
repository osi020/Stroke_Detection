"""
evaluate.py
-----------
Threshold selection and evaluation utilities shared between the
modeling notebook and any downstream reporting code.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ── Threshold selection ───────────────────────────────────────────────────────

def select_threshold_f2(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray,
    step: float = 0.01,
) -> float:
    """
    Sweep thresholds and return the one that maximises F2-score (β=2).

    WHY F2 instead of F1
    --------------------
    With ~5 % positive class, F1-based threshold search tends to select
    high thresholds (0.76–0.81) that maximise precision at the cost of
    recall collapsing to 10–16 %.  In stroke screening, a missed stroke
    (false negative) is far more costly than a false alarm.  F2 weights
    recall twice as heavily as precision, yielding clinically sensible
    thresholds in the 0.10–0.50 range.

    Parameters
    ----------
    y_true  : ground-truth binary labels.
    y_proba : predicted probabilities for the positive class.
    step    : granularity of the threshold sweep.

    Returns
    -------
    Best threshold (rounded to 2 decimal places).
    """
    best_thr, best_f2 = 0.5, 0.0
    for thr in np.arange(0.05, 0.95, step):
        y_pred = (y_proba >= thr).astype(int)
        score  = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
        if score > best_f2:
            best_f2, best_thr = score, thr
    return round(float(best_thr), 2)


# ── Full evaluation at a given threshold ─────────────────────────────────────

def evaluate_at_threshold(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray,
    threshold: float,
    model_name: str = "model",
) -> dict:
    """
    Return a dictionary of evaluation metrics at a fixed decision threshold.

    Metrics
    -------
    accuracy, precision, recall, f1, f2 (β=2), roc_auc, pr_auc
    """
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "model":          model_name,
        "best_threshold": threshold,
        "test_accuracy":  accuracy_score(y_true, y_pred),
        "test_precision": precision_score(y_true, y_pred, zero_division=0),
        "test_recall":    recall_score(y_true, y_pred, zero_division=0),
        "test_f1":        f1_score(y_true, y_pred, zero_division=0),
        "test_f2":        fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        "test_roc_auc":   roc_auc_score(y_true, y_proba),
        "test_pr_auc":    average_precision_score(y_true, y_proba),
    }


def evaluate_with_f2_threshold(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray,
    model_name: str = "model",
) -> dict:
    """
    Select the best F2-threshold automatically, then evaluate.
    Convenience wrapper around select_threshold_f2 + evaluate_at_threshold.
    """
    thr = select_threshold_f2(y_true, y_proba)
    return evaluate_at_threshold(y_true, y_proba, thr, model_name)


# ── Benchmark helper ──────────────────────────────────────────────────────────

def benchmark_models(
    models: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """
    Evaluate multiple fitted pipelines and return a ranked DataFrame.

    Parameters
    ----------
    models : {name: fitted_pipeline}
    X_test, y_test : held-out test set

    Returns
    -------
    DataFrame sorted by PR-AUC descending.
    """
    rows = []
    for name, pipe in models.items():
        y_proba = pipe.predict_proba(X_test)[:, 1]
        rows.append(evaluate_with_f2_threshold(y_test, y_proba, name))
    return (
        pd.DataFrame(rows)
        .sort_values("test_pr_auc", ascending=False)
        .reset_index(drop=True)
    )
