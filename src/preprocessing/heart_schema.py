"""Heart (UCI Cleveland) feature semantics — single source of truth.

The Cleveland database encodes several *categorical* variables with integers
(``cp`` 1-4, ``thal`` 3/6/7, ``restecg`` 0/1/2 ...). They must NOT be treated as
continuous measurements, and — critically — training and inference must feed the
model the *identical* representation. The previous implementation let ``ca`` /
``thal`` reach a one-hot encoder as raw floats during training but as JSON
ints/strings at serve time, so ``handle_unknown="ignore"`` silently produced an
all-zero encoding for those columns instead of raising.

This module fixes that by defining one canonical representation (string tokens)
and one normaliser used by both ``src/training/train_heart.py`` and the API
model registry.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Continuous / integer measurements -> numeric pipeline (impute + scale).
HEART_NUMERIC: tuple[str, ...] = ("age", "trestbps", "chol", "thalach", "oldpeak")

# Integer-coded categories -> categorical pipeline (impute + one-hot).
HEART_CATEGORICAL: tuple[str, ...] = (
    "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal",
)

# Canonical accepted tokens per categorical feature (post-normalisation strings).
HEART_CATEGORIES: dict[str, list[str]] = {
    "sex": ["0", "1"],                 # 0 = female, 1 = male
    "cp": ["1", "2", "3", "4"],        # chest pain type
    "fbs": ["0", "1"],                 # fasting blood sugar > 120 mg/dl
    "restecg": ["0", "1", "2"],        # resting ECG result
    "exang": ["0", "1"],              # exercise-induced angina
    "slope": ["1", "2", "3"],         # slope of peak exercise ST segment
    "ca": ["0", "1", "2", "3"],       # major vessels coloured by fluoroscopy
    "thal": ["3", "6", "7"],          # 3 = normal, 6 = fixed defect, 7 = reversible defect
}

# Generous supersets of the Cleveland spread — sanity bounds, NOT clinical cut-offs.
HEART_NUMERIC_RANGES: dict[str, tuple[float, float]] = {
    "age": (18, 100),        # data: 29-77
    "trestbps": (80, 220),   # resting BP mm Hg; data: 94-200
    "chol": (100, 600),      # serum cholesterol mg/dl; data: 126-564
    "thalach": (60, 220),    # max heart rate bpm; data: 71-202
    "oldpeak": (0.0, 8.0),   # ST depression; data: 0.0-6.2
}

# Every categorical here is imputable by the pipeline; ``ca`` and ``thal`` are the
# only ones actually missing in the Cleveland file (4 and 2 rows) but any
# categorical may be omitted and will be filled with the training-fold mode.
HEART_OPTIONAL: set[str] = {"ca", "thal"}

# Human-readable option labels for the UI (value token -> label).
HEART_VALUE_LABELS: dict[str, dict[str, str]] = {
    "sex": {"0": "Female", "1": "Male"},
    "cp": {
        "1": "Typical angina",
        "2": "Atypical angina",
        "3": "Non-anginal pain",
        "4": "Asymptomatic",
    },
    "fbs": {"0": "120 mg/dl or below", "1": "Above 120 mg/dl"},
    "restecg": {
        "0": "Normal",
        "1": "ST-T wave abnormality",
        "2": "Probable/definite left ventricular hypertrophy",
    },
    "exang": {"0": "No", "1": "Yes"},
    "slope": {"1": "Upsloping", "2": "Flat", "3": "Downsloping"},
    "ca": {"0": "0 vessels", "1": "1 vessel", "2": "2 vessels", "3": "3 vessels"},
    "thal": {"3": "Normal", "6": "Fixed defect", "7": "Reversible defect"},
}


def _token(value) -> object:
    """Normalise a single raw categorical value to its canonical string token.

    Accepts ``3``, ``3.0``, ``"3"``, ``"3.0"`` -> ``"3"``. Leaves missing as NaN.
    Non-integer-looking values are returned stringified so the caller's
    membership check against HEART_CATEGORIES can reject them cleanly.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if isinstance(value, str):
        value = value.strip()
        if value == "" or value.lower() in {"nan", "none"}:
            return np.nan
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if f != f:  # NaN
        return np.nan
    if float(f).is_integer():
        return str(int(f))
    return str(value)


def normalize_categoricals(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with every HEART_CATEGORICAL column as canonical tokens."""
    out = frame.copy()
    for col in HEART_CATEGORICAL:
        if col in out.columns:
            out[col] = out[col].map(_token).astype("object")
    return out


def normalize_measurements(values: dict) -> dict:
    """Normalise an inbound API measurement dict (categoricals -> tokens)."""
    out = dict(values)
    for col in HEART_CATEGORICAL:
        if col in out and out[col] is not None and out[col] != "":
            out[col] = _token(out[col])
    return out
