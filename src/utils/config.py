"""Shared paths, feature definitions, and reproducibility settings."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RANDOM_STATE = 42
TEST_SIZE = 0.2


@dataclass(frozen=True)
class DiseaseConfig:
    key: str
    title: str
    target: str
    positive_label: str
    features: tuple[str, ...]
    categorical: tuple[str, ...] = ()
    zero_as_missing: tuple[str, ...] = ()

    @property
    def numeric(self) -> tuple[str, ...]:
        return tuple(f for f in self.features if f not in self.categorical)

    @property
    def raw_path(self) -> Path:
        return ROOT / "datasets" / "raw" / f"{self.key}.csv"

    @property
    def processed_path(self) -> Path:
        return ROOT / "datasets" / "processed" / f"{self.key}_clean.csv"

    @property
    def model_dir(self) -> Path:
        return ROOT / "models" / self.key


DISEASES = {
    "liver": DiseaseConfig(
        "liver", "Liver disease", "Selector", "higher-risk pattern",
        ("Age", "Gender", "TB", "DB", "Alkphos", "Sgpt", "Sgot", "TP", "ALB", "A/G Ratio"), ("Gender",),
    ),
    "heart": DiseaseConfig(
        "heart", "Heart disease", "target", "presence of heart disease pattern",
        ("age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"),
        ("sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"),
    ),
    "diabetes": DiseaseConfig(
        "diabetes", "Diabetes", "target", "diabetes pattern",
        ("pregnancies", "glucose", "blood_pressure", "skin_thickness", "insulin",
         "bmi", "diabetes_pedigree", "age"), (),
        ("glucose", "blood_pressure", "skin_thickness", "insulin", "bmi"),
    ),
    "kidney": DiseaseConfig(
        "kidney", "Chronic kidney disease", "class", "CKD pattern",
        ("age","bp","sg","al","su","rbc","pc","pcc","ba","bgr","bu","sc","sod","pot","hemo","pcv","wc","rc","htn","dm","cad","appet","pe","ane"),
        ("rbc","pc","pcc","ba","htn","dm","cad","appet","pe","ane"),
    ),
    "parkinsons": DiseaseConfig(
        "parkinsons", "Parkinson's disease", "status", "Parkinson's voice pattern",
        ("MDVP:Fo(Hz)","MDVP:Fhi(Hz)","MDVP:Flo(Hz)","MDVP:Jitter(%)","MDVP:Jitter(Abs)","MDVP:RAP","MDVP:PPQ","Jitter:DDP","MDVP:Shimmer","MDVP:Shimmer(dB)","Shimmer:APQ3","Shimmer:APQ5","MDVP:APQ","Shimmer:DDA","NHR","HNR","RPDE","DFA","spread1","spread2","D2","PPE"),
    ),
}

DISCLAIMER = (
    "Educational and research use only. This system estimates patterns in public "
    "datasets; it does not provide a medical diagnosis. Consult a qualified "
    "healthcare professional for medical advice."
)
