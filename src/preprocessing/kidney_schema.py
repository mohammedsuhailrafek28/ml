"""Chronic Kidney Disease (UCI) feature semantics - single source of truth.

The UCI CKD file is notoriously messy: `?` for missing, stray tabs / spaces in
categorical cells (`"\tno"`, `" yes"`), and a tab in the target (`"ckd\t"`).
Numeric columns are read as strings when a `?` is present.

This module defines one canonical representation and the normalisers used by
`src/training/train_kidney.py`, the API model registry and the tests, so
training and inference feed the model the identical values.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

# Continuous / integer measurements -> numeric pipeline (impute + scale).
KIDNEY_NUMERIC: tuple[str, ...] = (
    "age", "bp", "sg", "al", "su", "bgr", "bu", "sc", "sod", "pot",
    "hemo", "pcv", "wc", "rc",
)

# yes/no & normal/abnormal style fields -> categorical pipeline (impute + one-hot).
KIDNEY_CATEGORICAL: tuple[str, ...] = (
    "rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane",
)

KIDNEY_FEATURES: tuple[str, ...] = KIDNEY_NUMERIC + KIDNEY_CATEGORICAL

KIDNEY_CATEGORIES: dict[str, list[str]] = {
    "rbc": ["normal", "abnormal"],
    "pc": ["normal", "abnormal"],
    "pcc": ["present", "notpresent"],
    "ba": ["present", "notpresent"],
    "htn": ["yes", "no"],
    "dm": ["yes", "no"],
    "cad": ["yes", "no"],
    "appet": ["good", "poor"],
    "pe": ["yes", "no"],
    "ane": ["yes", "no"],
}

# Generous supersets of the observed spread - sanity bounds, NOT clinical cut-offs.
KIDNEY_NUMERIC_RANGES: dict[str, tuple[float, float]] = {
    "age": (1, 110),        # years; data 2-90
    "bp": (30, 200),        # diastolic BP mm Hg; data 50-180
    "sg": (1.0, 1.05),      # urine specific gravity; data 1.005-1.025
    "al": (0, 5),           # albumin (0-5 scale)
    "su": (0, 5),           # sugar (0-5 scale)
    "bgr": (20, 600),       # random blood glucose mg/dL; data 22-490
    "bu": (1, 450),         # blood urea mg/dL; data 1.5-391
    "sc": (0.1, 90),        # serum creatinine mg/dL; data 0.4-76
    "sod": (4, 170),        # sodium mEq/L; data 4.5-163 (4.5 is a known dataset artefact)
    "pot": (2, 50),         # potassium mEq/L; data 2.5-47
    "hemo": (3, 20),        # haemoglobin g/dL; data 3.1-17.8
    "pcv": (9, 60),         # packed cell volume %; data 9-54
    "wc": (2000, 30000),    # white blood cell count /cumm; data 2200-26400
    "rc": (2, 8),           # red blood cell count millions/cmm; data 2.1-8
}

# Every CKD feature is imputable by the pipeline. The UCI file is heavily
# incomplete (rbc ~38% missing, rc ~33%, wc ~27%), so the API treats every
# feature as optional: omit it or send null and the pipeline imputes.
KIDNEY_OPTIONAL: set[str] = set(KIDNEY_FEATURES)

KIDNEY_LABELS: dict[str, str] = {
    "age": "Age", "bp": "Diastolic blood pressure",
    "sg": "Urine specific gravity", "al": "Urine albumin", "su": "Urine sugar",
    "rbc": "Red blood cells (urine microscopy)", "pc": "Pus cells (urine microscopy)",
    "pcc": "Pus cell clumps", "ba": "Bacteria (urine microscopy)",
    "bgr": "Random blood glucose", "bu": "Blood urea", "sc": "Serum creatinine",
    "sod": "Sodium", "pot": "Potassium", "hemo": "Haemoglobin",
    "pcv": "Packed cell volume", "wc": "White blood cell count",
    "rc": "Red blood cell count", "htn": "Hypertension",
    "dm": "Diabetes mellitus", "cad": "Coronary artery disease",
    "appet": "Appetite", "pe": "Pedal edema", "ane": "Anaemia",
}

KIDNEY_UNITS: dict[str, str] = {
    "age": "years", "bp": "mm Hg", "sg": "ratio", "al": "0-5 scale", "su": "0-5 scale",
    "bgr": "mg/dL", "bu": "mg/dL", "sc": "mg/dL", "sod": "mEq/L", "pot": "mEq/L",
    "hemo": "g/dL", "pcv": "%", "wc": "cells/cumm", "rc": "millions/cmm",
}

# Human-readable option labels for the UI (token -> label).
KIDNEY_VALUE_LABELS: dict[str, dict[str, str]] = {
    "rbc": {"normal": "Normal", "abnormal": "Abnormal"},
    "pc": {"normal": "Normal", "abnormal": "Abnormal"},
    "pcc": {"present": "Present", "notpresent": "Not present"},
    "ba": {"present": "Present", "notpresent": "Not present"},
    "htn": {"yes": "Yes", "no": "No"},
    "dm": {"yes": "Yes", "no": "No"},
    "cad": {"yes": "Yes", "no": "No"},
    "appet": {"good": "Good", "poor": "Poor"},
    "pe": {"yes": "Yes", "no": "No"},
    "ane": {"yes": "Yes", "no": "No"},
}


def _clean_token(value) -> object:
    """Lower-case and strip everything except a-z (kills stray tabs/spaces)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    s = re.sub(r"[^a-z]", "", str(value).strip().lower())
    return s if s else np.nan


def normalize_target(series: pd.Series) -> pd.Series:
    """`ckd` / `ckd\\t` / ` ckd ` -> 1, `notckd` -> 0, else NaN."""
    return series.map(_clean_token).map({"ckd": 1, "notckd": 0})


def normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Coerce numeric columns to float and categorical columns to clean tokens."""
    out = frame.copy()
    for col in KIDNEY_NUMERIC:
        if col in out.columns:
            out[col] = pd.to_numeric(
                out[col].replace({"?": np.nan, "": np.nan}), errors="coerce"
            )
    for col in KIDNEY_CATEGORICAL:
        if col in out.columns:
            out[col] = out[col].map(_clean_token).astype("object")
    return out


def normalize_measurements(values: dict) -> dict:
    """Normalise an inbound API measurement dict the same way as the trainer.

    Numeric cells become float (or None for blank / '?'); categorical cells
    become clean tokens (or None). None => the pipeline imputes.
    """
    out = dict(values)
    for col in KIDNEY_NUMERIC:
        if col in out:
            v = out[col]
            if v in (None, "", "?") or (isinstance(v, str) and v.strip() in ("", "?")):
                out[col] = None
            else:
                try:
                    out[col] = float(v)
                except (TypeError, ValueError):
                    pass  # leave as-is; validation will report it
    for col in KIDNEY_CATEGORICAL:
        if col in out:
            tok = _clean_token(out[col])
            out[col] = None if (isinstance(tok, float) and np.isnan(tok)) else tok
    return out
