"""Structural validation for medical tabular datasets."""
from __future__ import annotations

import pandas as pd
from src.utils.config import DiseaseConfig


def validate_schema(frame: pd.DataFrame, config: DiseaseConfig) -> None:
    missing = sorted(set(config.features + (config.target,)) - set(frame.columns))
    if missing:
        raise ValueError(f"{config.title}: missing required columns: {missing}")
    if frame[config.target].dropna().nunique() != 2:
        raise ValueError(f"{config.title}: target must contain exactly two classes")

