"""Parkinson's (UCI voice) feature semantics - single source of truth.

The dataset is 195 sustained-phonation recordings from 32 subjects (~6 each).
`name` encodes the subject (`phon_R01_S01_3` -> subject `phon_R01_S01`); it is a
GROUPING key for subject-disjoint splitting and must NEVER be a model feature.
`status` is the subject-level label (1 = Parkinson's, 0 = control).

All 22 predictors are pre-computed acoustic voice biomarkers - the kind produced
by voice-analysis software (e.g. Praat), not values a person can type from
memory. The UI states this explicitly.
"""
from __future__ import annotations

import re

import pandas as pd

SUBJECT_ID_COLUMN = "name"
TARGET = "status"
SUBJECT_ID_REGEX = r"^(phon_R\d+_S\d+)"

# 22 production features, in dataset column order.
PARKINSONS_FEATURES: tuple[str, ...] = (
    "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)",
    "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP",
    "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5",
    "MDVP:APQ", "Shimmer:DDA",
    "NHR", "HNR",
    "RPDE", "DFA", "spread1", "spread2", "D2", "PPE",
)

PARKINSONS_LABELS: dict[str, str] = {
    "MDVP:Fo(Hz)": "Average vocal fundamental frequency",
    "MDVP:Fhi(Hz)": "Maximum vocal fundamental frequency",
    "MDVP:Flo(Hz)": "Minimum vocal fundamental frequency",
    "MDVP:Jitter(%)": "Frequency variation (jitter, %)",
    "MDVP:Jitter(Abs)": "Absolute frequency variation (jitter)",
    "MDVP:RAP": "Relative amplitude perturbation (jitter)",
    "MDVP:PPQ": "Five-point period perturbation quotient (jitter)",
    "Jitter:DDP": "Average absolute difference of period differences (jitter)",
    "MDVP:Shimmer": "Amplitude variation (shimmer)",
    "MDVP:Shimmer(dB)": "Amplitude variation (shimmer, dB)",
    "Shimmer:APQ3": "Three-point amplitude perturbation quotient (shimmer)",
    "Shimmer:APQ5": "Five-point amplitude perturbation quotient (shimmer)",
    "MDVP:APQ": "Eleven-point amplitude perturbation quotient (shimmer)",
    "Shimmer:DDA": "Average absolute difference between amplitude differences",
    "NHR": "Noise-to-harmonics ratio",
    "HNR": "Harmonics-to-noise ratio",
    "RPDE": "Recurrence period density entropy (nonlinear dynamics)",
    "DFA": "Detrended fluctuation analysis (signal fractal scaling)",
    "spread1": "Nonlinear fundamental-frequency variation measure 1",
    "spread2": "Nonlinear fundamental-frequency variation measure 2",
    "D2": "Correlation dimension (nonlinear dynamics)",
    "PPE": "Pitch period entropy",
}

PARKINSONS_UNITS: dict[str, str] = {
    "MDVP:Fo(Hz)": "Hz", "MDVP:Fhi(Hz)": "Hz", "MDVP:Flo(Hz)": "Hz",
    "MDVP:Jitter(%)": "%", "MDVP:Jitter(Abs)": "s", "HNR": "dB",
    "MDVP:Shimmer(dB)": "dB",
}

PARKINSONS_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Fundamental frequency", ("MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)")),
    ("Jitter (frequency variation)",
     ("MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP")),
    ("Shimmer (amplitude variation)",
     ("MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5",
      "MDVP:APQ", "Shimmer:DDA")),
    ("Noise / harmonic measures", ("NHR", "HNR")),
    ("Nonlinear voice measures",
     ("RPDE", "DFA", "spread1", "spread2", "D2", "PPE")),
)

# Generous supersets of the observed spread - sanity bounds, NOT diagnostic cut-offs.
PARKINSONS_RANGES: dict[str, tuple[float, float]] = {
    "MDVP:Fo(Hz)": (50, 300), "MDVP:Fhi(Hz)": (60, 650), "MDVP:Flo(Hz)": (40, 300),
    "MDVP:Jitter(%)": (0.0, 0.1), "MDVP:Jitter(Abs)": (0.0, 0.001),
    "MDVP:RAP": (0.0, 0.06), "MDVP:PPQ": (0.0, 0.06), "Jitter:DDP": (0.0, 0.2),
    "MDVP:Shimmer": (0.0, 0.3), "MDVP:Shimmer(dB)": (0.0, 3.0),
    "Shimmer:APQ3": (0.0, 0.2), "Shimmer:APQ5": (0.0, 0.2), "MDVP:APQ": (0.0, 0.3),
    "Shimmer:DDA": (0.0, 0.5), "NHR": (0.0, 1.0), "HNR": (0.0, 45.0),
    "RPDE": (0.0, 1.0), "DFA": (0.4, 1.0), "spread1": (-10.0, 0.0),
    "spread2": (0.0, 1.0), "D2": (0.5, 5.0), "PPE": (0.0, 1.0),
}

# PHASE 11: exact / near-collinear measures. Reduced set keeps one representative
# per acoustic family plus every non-redundant measure.
PARKINSONS_REDUCED_FEATURES: tuple[str, ...] = (
    "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)",
    "MDVP:Jitter(%)", "MDVP:Jitter(Abs)",           # drop RAP/PPQ/DDP (|r|>0.95 with Jitter%)
    "MDVP:Shimmer(dB)", "MDVP:APQ",                  # drop Shimmer/APQ3/APQ5/DDA (|r|>0.95)
    "NHR", "HNR",
    "RPDE", "DFA", "spread1", "spread2", "D2", "PPE",
)


def subject_id(name) -> str:
    """`phon_R01_S07_4` -> `phon_R01_S07`. Falls back to the whole string."""
    if name is None:
        return ""
    m = re.match(SUBJECT_ID_REGEX, str(name))
    return m.group(1) if m else str(name)


def subject_ids(names: pd.Series) -> pd.Series:
    return names.map(subject_id)
