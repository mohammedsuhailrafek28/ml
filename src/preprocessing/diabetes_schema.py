"""Diabetes (Pima Indians) feature semantics — single source of truth.

The Pima CSV encodes clinically-impossible measurements as ``0`` (a glucose,
blood pressure, skin fold, insulin or BMI of exactly zero is not survivable).
The previous implementation converted those zeros to NaN *in train_all.py,
before the pipeline* — so the persisted model never learned that step and, at
inference, a submitted ``0`` was scaled as a genuine very-low value (a
train/serve mismatch and a form of leakage).

This module fixes that by defining one zero-as-missing set and a
pipeline-internal transformer used by both ``src/training/train_diabetes.py``
and the API model registry, so training and inference behave identically.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

DIABETES_FEATURES: tuple[str, ...] = (
    "pregnancies", "glucose", "blood_pressure", "skin_thickness",
    "insulin", "bmi", "diabetes_pedigree", "age",
)

# Columns where a literal 0 is physiologically impossible => treat as missing.
# pregnancies 0 (nulliparous) and age are genuine values and are NOT included.
DIABETES_ZERO_AS_MISSING: tuple[str, ...] = (
    "glucose", "blood_pressure", "skin_thickness", "insulin", "bmi",
)

# Generous sanity bounds — NOT diagnostic cut-offs. 0 is accepted for the
# zero-as-missing columns (it means "not provided" and the pipeline imputes it).
DIABETES_RANGES: dict[str, tuple[float, float]] = {
    "pregnancies": (0, 20),          # data 0-17
    "glucose": (0, 400),             # mg/dL, 2-hr OGTT plasma glucose; data 0-199
    "blood_pressure": (0, 200),      # mm Hg, diastolic; data 0-122
    "skin_thickness": (0, 110),      # mm, triceps skin fold; data 0-99
    "insulin": (0, 1200),            # mu U/ml, 2-hr serum insulin; data 0-846
    "bmi": (0, 100),                 # kg/m^2; data 0-67.1
    "diabetes_pedigree": (0.0, 3.0), # pedigree function; data 0.078-2.42
    "age": (18, 120),                # years; data 21-81
}

DIABETES_INTEGER_FIELDS: set[str] = {"pregnancies", "age"}

DIABETES_LABELS: dict[str, str] = {
    "pregnancies": "Number of pregnancies",
    "glucose": "Plasma glucose (2-hr OGTT)",
    "blood_pressure": "Diastolic blood pressure",
    "skin_thickness": "Triceps skin fold thickness",
    "insulin": "2-hour serum insulin",
    "bmi": "Body mass index",
    "diabetes_pedigree": "Diabetes pedigree function",
    "age": "Age",
}

DIABETES_UNITS: dict[str, str] = {
    "pregnancies": "count",
    "glucose": "mg/dL",
    "blood_pressure": "mm Hg",
    "skin_thickness": "mm",
    "insulin": "mu U/ml",
    "bmi": "kg/m²",
    "diabetes_pedigree": "score",
    "age": "years",
}


class ZeroToNaN(BaseEstimator, TransformerMixin):
    """Replace exact zeros with NaN in the configured columns.

    Stateless and deterministic (``fit`` is a no-op), so it is safe anywhere in
    a Pipeline and serialises cleanly with joblib. Accepts a pandas DataFrame
    (the training/inference path always passes named columns) and returns a
    DataFrame so the downstream ColumnTransformer keeps working by name.
    """

    def __init__(self, columns=DIABETES_ZERO_AS_MISSING):
        self.columns = tuple(columns)

    def fit(self, X, y=None):  # noqa: D401 - sklearn API
        return self

    def transform(self, X):
        frame = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        cols = [c for c in self.columns if c in frame.columns]
        frame[cols] = frame[cols].replace(0, np.nan)
        return frame

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features if input_features is not None else [], dtype=object)


# Engineered interaction terms evaluated in PHASE 11. Kept here (not in the
# trainer) so that, IF the dev-only CV shows they help, the fitted pipeline that
# references them still deserialises. They are computed deterministically from
# the raw columns, identically at train and inference time.
DIABETES_INTERACTIONS: tuple[str, ...] = (
    "glucose_x_bmi", "bmi_x_age", "glucose_over_insulin",
)


class AddInteractions(BaseEstimator, TransformerMixin):
    """Append deterministic interaction columns to the raw feature frame."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        frame = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        g = pd.to_numeric(frame["glucose"], errors="coerce")
        b = pd.to_numeric(frame["bmi"], errors="coerce")
        a = pd.to_numeric(frame["age"], errors="coerce")
        ins = pd.to_numeric(frame["insulin"], errors="coerce")
        frame["glucose_x_bmi"] = g * b
        frame["bmi_x_age"] = b * a
        frame["glucose_over_insulin"] = g / ins.replace(0, np.nan)
        return frame

    def get_feature_names_out(self, input_features=None):
        base = list(input_features) if input_features is not None else []
        return np.asarray(base + list(DIABETES_INTERACTIONS), dtype=object)
