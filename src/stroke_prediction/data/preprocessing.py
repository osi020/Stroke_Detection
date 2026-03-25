"""
preprocessing.py
----------------
All data-loading and feature-engineering logic in one place.
Both the Streamlit app and the notebooks import from here so the
preprocessing is never duplicated or out of sync.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from stroke_prediction.config import DATA_RAW, RANDOM_STATE, TEST_SIZE


# ── Raw data ──────────────────────────────────────────────────────────────────

def load_raw() -> pd.DataFrame:
    """Load the original CSV as-is."""
    path = DATA_RAW / "healthcare-dataset-stroke-data.csv"
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply minimal, reproducible cleaning steps:

    1. Drop the ``id`` column (patient identifier, not predictive).
    2. Remove the ``'Other'`` gender category (3 rows — too rare to model).

    Returns a fresh copy; the original DataFrame is not modified.
    """
    df = df.copy()
    if "id" in df.columns:
        df.drop(columns=["id"], inplace=True)
    df = df[df["gender"] != "Other"].reset_index(drop=True)
    return df


# ── Feature groups ────────────────────────────────────────────────────────────

def get_feature_groups(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Return (numerical_cols, categorical_cols) inferred from dtypes.

    Numerical  : int64, float64
    Categorical: object, category, bool
    """
    num = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    return num, cat


# ── Sklearn preprocessing pipeline ───────────────────────────────────────────

def build_preprocessor(
    num_cols: list[str],
    cat_cols: list[str],
) -> ColumnTransformer:
    """
    Return a fitted-ready ColumnTransformer:

    Numerical pipeline
        1. Median imputation  (handles missing BMI values)
        2. log1p transform    (reduces right-skew in age & glucose)
        3. StandardScaler

    Categorical pipeline
        OneHotEncoder with ``handle_unknown='ignore'``
        (safely handles unseen categories at inference time)
    """
    num_pipe = Pipeline([
        ("imputer",       SimpleImputer(strategy="median")),
        ("log_transform", FunctionTransformer(np.log1p, validate=False)),
        ("scaler",        StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        sparse_threshold=0.0,
    )


# ── Train / test split ────────────────────────────────────────────────────────

def get_splits(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Stratified train/test split (80/20).

    Parameters
    ----------
    df : cleaned DataFrame that still contains the ``stroke`` target column.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=["stroke"])
    y = df["stroke"]
    return train_test_split(
        X, y,
        test_size=RANDOM_STATE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


# ── Convenience: one-liner load → split ──────────────────────────────────────

def load_and_split() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load raw data, clean it, and return train/test splits."""
    df = clean(load_raw())
    X = df.drop(columns=["stroke"])
    y = df["stroke"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


# ── Inference helper ──────────────────────────────────────────────────────────

def build_inference_row(
    gender: str,
    age: float,
    hypertension: int,
    heart_disease: int,
    ever_married: str,
    work_type: str,
    residence_type: str,
    avg_glucose_level: float,
    bmi: float | None,
    smoking_status: str,
) -> pd.DataFrame:
    """
    Build a single-row DataFrame suitable for model.predict_proba().
    Pass ``bmi=None`` when the value is unknown (imputed as median).
    """
    return pd.DataFrame([{
        "gender":            gender,
        "age":               float(age),
        "hypertension":      int(hypertension),
        "heart_disease":     int(heart_disease),
        "ever_married":      ever_married,
        "work_type":         work_type,
        "Residence_type":    residence_type,
        "avg_glucose_level": float(avg_glucose_level),
        "bmi":               float(bmi) if bmi else np.nan,
        "smoking_status":    smoking_status,
    }])
