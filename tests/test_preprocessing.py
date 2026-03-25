"""
Basic smoke tests for the preprocessing pipeline.
Run with:  python -m pytest tests/ -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest

from stroke_prediction.data.preprocessing import (
    build_inference_row,
    build_preprocessor,
    clean,
    get_feature_groups,
    load_raw,
)


@pytest.fixture(scope="module")
def raw_df():
    return load_raw()


@pytest.fixture(scope="module")
def clean_df(raw_df):
    return clean(raw_df)


def test_load_raw_shape(raw_df):
    assert raw_df.shape[0] > 5_000
    assert "stroke" in raw_df.columns


def test_clean_drops_id(clean_df):
    assert "id" not in clean_df.columns


def test_clean_removes_other_gender(clean_df):
    assert "Other" not in clean_df["gender"].unique()


def test_feature_groups(clean_df):
    X = clean_df.drop(columns=["stroke"])
    num, cat = get_feature_groups(X)
    assert "age" in num
    assert "bmi" in num
    assert "gender" in cat
    assert "smoking_status" in cat


def test_preprocessor_output_shape(clean_df):
    X = clean_df.drop(columns=["stroke"])
    num, cat = get_feature_groups(X)
    pre = build_preprocessor(num, cat)
    out = pre.fit_transform(X)
    assert out.shape[0] == len(X)
    assert out.shape[1] > len(num)   # one-hot expands cat columns


def test_preprocessor_handles_missing_bmi(clean_df):
    X = clean_df.drop(columns=["stroke"]).copy()
    X.loc[X.index[0], "bmi"] = np.nan
    num, cat = get_feature_groups(X)
    pre = build_preprocessor(num, cat)
    out = pre.fit_transform(X)
    assert not np.isnan(out).any()


def test_inference_row_schema(clean_df):
    X_ref = clean_df.drop(columns=["stroke"])
    row = build_inference_row(
        gender="Male", age=65, hypertension=1, heart_disease=0,
        ever_married="Yes", work_type="Private", residence_type="Urban",
        avg_glucose_level=150.0, bmi=28.5, smoking_status="never smoked",
    )
    assert set(row.columns) == set(X_ref.columns)


def test_inference_row_missing_bmi():
    row = build_inference_row(
        gender="Female", age=50, hypertension=0, heart_disease=0,
        ever_married="No", work_type="Self-employed", residence_type="Rural",
        avg_glucose_level=90.0, bmi=None, smoking_status="smokes",
    )
    assert np.isnan(row["bmi"].iloc[0])
