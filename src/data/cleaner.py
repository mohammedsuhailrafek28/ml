"""Auditable, disease-aware data cleaning."""
from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
from src.utils.config import DiseaseConfig


def clean_dataset(frame: pd.DataFrame, config: DiseaseConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    data = frame.copy()
    before_shape = list(data.shape)
    original_missing = data.isna().sum().to_dict()
    for col in data.select_dtypes(include="object"):
        data[col] = data[col].astype("string").str.strip().str.lower()
        data[col] = data[col].replace({"?": pd.NA, "": pd.NA, "\t?": pd.NA})
    for col in config.numeric:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    for col in config.zero_as_missing:
        data.loc[data[col] == 0, col] = np.nan
    duplicate_count = int(data.duplicated().sum())
    data = data.drop_duplicates().reset_index(drop=True)
    data = data.dropna(subset=[config.target])
    data[config.target] = pd.to_numeric(data[config.target], errors="raise").astype(int)
    invalid_counts: dict[str, int] = {}
    for col in config.numeric:
        invalid = data[col] < 0
        invalid_counts[col] = int(invalid.sum())
        data.loc[invalid, col] = np.nan
    summary = {
        "disease": config.key,
        "shape_before": before_shape,
        "shape_after": list(data.shape),
        "duplicates_removed": duplicate_count,
        "missing_before": original_missing,
        "missing_after_rules": data.isna().sum().to_dict(),
        "invalid_negative_values_set_missing": invalid_counts,
        "missing_strategy": "Median for numeric and most-frequent for categorical, fitted on training folds only.",
        "outlier_strategy": "Retained. Tree models are robust; scaled linear/SVM models are compared without deleting plausible extremes.",
    }
    return data, summary

