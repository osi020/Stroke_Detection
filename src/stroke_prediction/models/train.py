"""
train.py
--------
Defines the canonical model pipelines used across notebooks and the app.
All pipelines are built from the shared preprocessor in
``stroke_prediction.data.preprocessing``.
"""

from __future__ import annotations

import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

from stroke_prediction.config import RANDOM_STATE
from stroke_prediction.data.preprocessing import build_preprocessor, get_feature_groups


def build_standard_pipeline(X_ref: pd.DataFrame) -> ImbPipeline:
    """
    LogReg_Balanced — class-weighted Logistic Regression.

    F2-optimised decision threshold: 0.72
    Holdout recall 68 % | precision 22 % | PR-AUC 0.257

    Parameters
    ----------
    X_ref : any DataFrame with the same schema as X_train (used to infer
            numerical / categorical column groups).
    """
    num_cols, cat_cols = get_feature_groups(X_ref)
    preprocessor = build_preprocessor(num_cols, cat_cols)
    return ImbPipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(
            max_iter=3_000,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        )),
    ])


def build_sensitive_pipeline(
    standard_pipeline: ImbPipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> CalibratedClassifierCV:
    """
    LogReg_Balanced_calibrated — isotonic calibration of the standard model.

    Produces well-calibrated probabilities that reflect the true ~5 % base rate.
    F2-optimised decision threshold: 0.14
    Holdout recall 72 % | precision 20 % | PR-AUC 0.256

    Parameters
    ----------
    standard_pipeline : already **fitted** ImbPipeline from build_standard_pipeline().
    X_train, y_train  : training data (calibration uses 3-fold CV internally).
    """
    cal = CalibratedClassifierCV(standard_pipeline, method="isotonic", cv=3)
    cal.fit(X_train, y_train)
    return cal


def load_or_train(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Convenience function: train both pipelines and return them.

    Returns
    -------
    standard_pipe  : fitted ImbPipeline  (LogReg_Balanced)
    sensitive_pipe : fitted CalibratedClassifierCV
    """
    standard = build_standard_pipeline(X_train)
    standard.fit(X_train, y_train)
    sensitive = build_sensitive_pipeline(standard, X_train, y_train)
    return standard, sensitive
