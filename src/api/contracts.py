"""Authoritative, user-facing disease input contracts.

Field order follows the persisted release manifest's ``ordered_active_features``.
The metadata attached to each Pydantic field is consumed by OpenAPI and the
frontend contract generator. Bounds are sanity checks, not clinical limits.
"""
from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, create_model, model_validator

from src.model_release import load_release_manifest
from src.preprocessing.diabetes_schema import DIABETES_INTEGER_FIELDS, DIABETES_RANGES
from src.preprocessing.heart_schema import HEART_CATEGORIES, HEART_NUMERIC_RANGES, HEART_OPTIONAL, HEART_VALUE_LABELS, _token
from src.preprocessing.kidney_schema import KIDNEY_CATEGORICAL, KIDNEY_CATEGORIES, KIDNEY_FEATURES, KIDNEY_NUMERIC_RANGES, KIDNEY_UNITS, _clean_token
from src.preprocessing.parkinsons_schema import PARKINSONS_FEATURES, PARKINSONS_LABELS, PARKINSONS_RANGES, PARKINSONS_REDUCED_FEATURES, PARKINSONS_UNITS
from src.api.services.model_registry import LIVER_OPTIONAL, LIVER_RANGES
from src.utils.config import DISEASES

_GROUPS: dict[str, dict[str, str]] = {
    "liver": {"Age": "Patient information", "Gender": "Patient information", "TB": "Bilirubin measurements", "DB": "Bilirubin measurements", "Alkphos": "Liver enzymes", "Sgpt": "Liver enzymes", "Sgot": "Liver enzymes", "TP": "Protein measurements", "ALB": "Protein measurements", "A/G Ratio": "Protein measurements"},
    "diabetes": {"pregnancies": "Patient information", "age": "Patient information", "glucose": "Glucose and blood pressure", "blood_pressure": "Glucose and blood pressure", "skin_thickness": "Body measurements", "insulin": "Body measurements", "bmi": "Body measurements", "diabetes_pedigree": "Risk history"},
    "heart": {"age": "Patient information", "sex": "Patient information", "cp": "Symptoms", "trestbps": "Vitals", "chol": "Laboratory measurements", "fbs": "Laboratory measurements", "restecg": "Cardiac tests", "thalach": "Vitals", "exang": "Symptoms", "oldpeak": "Cardiac tests", "slope": "Cardiac tests", "ca": "Cardiac tests", "thal": "Cardiac tests"},
    "kidney": {"age": "Patient information", "bp": "Vitals", "sg": "Urine measurements", "al": "Urine measurements", "su": "Urine measurements", "rbc": "Urine microscopy", "pc": "Urine microscopy", "pcc": "Urine microscopy", "ba": "Urine microscopy", "bgr": "Blood measurements", "bu": "Blood measurements", "sc": "Blood measurements", "sod": "Blood measurements", "pot": "Blood measurements", "hemo": "Blood measurements", "pcv": "Blood measurements", "wc": "Blood measurements", "rc": "Blood measurements", "htn": "Medical history", "dm": "Medical history", "cad": "Medical history", "appet": "Symptoms", "pe": "Symptoms", "ane": "Medical history"},
    "parkinsons": {key: group for group, keys in (("Fundamental frequency", ("MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)")), ("Jitter (frequency variation)", ("MDVP:Jitter(%)", "MDVP:Jitter(Abs)")), ("Shimmer (amplitude variation)", ("MDVP:Shimmer(dB)", "MDVP:APQ")), ("Noise / harmonic measures", ("NHR", "HNR")), ("Nonlinear voice measures", ("RPDE", "DFA", "spread1", "spread2", "D2", "PPE"))) for key in keys},
}

_LABELS = {
    "liver": {"Age": ("Age", "Age in years."), "Gender": ("Gender", "Select the category used in the source dataset."), "TB": ("Total bilirubin", "Total bilirubin measurement."), "DB": ("Direct bilirubin", "Direct bilirubin measurement."), "Alkphos": ("Alkaline phosphatase", "Alkaline phosphatase measurement."), "Sgpt": ("Alanine aminotransferase (ALT/SGPT)", "ALT/SGPT enzyme measurement."), "Sgot": ("Aspartate aminotransferase (AST/SGOT)", "AST/SGOT enzyme measurement."), "TP": ("Total proteins", "Total serum protein measurement."), "ALB": ("Albumin", "Serum albumin measurement."), "A/G Ratio": ("Albumin / globulin ratio", "Optional; leave unavailable to use the persisted pipeline's imputation.")},
    "diabetes": {"pregnancies": ("Number of pregnancies", "Enter the count; zero is a valid value."), "glucose": ("Plasma glucose (2-hour OGTT)", "Zero represents a source dataset missing-value code."), "blood_pressure": ("Diastolic blood pressure", "Zero represents a source dataset missing-value code."), "skin_thickness": ("Triceps skinfold thickness", "Zero represents a source dataset missing-value code."), "insulin": ("Two-hour serum insulin", "Zero represents a source dataset missing-value code."), "bmi": ("Body mass index", "Zero represents a source dataset missing-value code."), "diabetes_pedigree": ("Diabetes pedigree function", "Source-dataset pedigree score."), "age": ("Age", "Age in years.")},
    "heart": {"age": ("Age", "Age in years."), "sex": ("Sex category", "Source dataset code: 0 female, 1 male."), "cp": ("Chest pain type", "Choose the source dataset category."), "trestbps": ("Resting blood pressure", "Resting blood pressure measurement."), "chol": ("Serum cholesterol", "Serum cholesterol measurement."), "fbs": ("Fasting blood sugar > 120 mg/dL", "Source dataset category."), "restecg": ("Resting ECG result", "Choose the source dataset category."), "thalach": ("Maximum heart rate", "Maximum heart rate achieved."), "exang": ("Exercise-induced angina", "Source dataset category."), "oldpeak": ("ST depression", "ST depression induced by exercise relative to rest."), "slope": ("Peak exercise ST slope", "Choose the source dataset category."), "ca": ("Major vessels", "Optional; if unavailable, the persisted pipeline imputes it."), "thal": ("Thalassemia category", "Optional; if unavailable, the persisted pipeline imputes it.")},
    "kidney": {"age": ("Age", "Age in years."), "bp": ("Diastolic blood pressure", "Blood pressure measurement."), "sg": ("Urine specific gravity", "Urine specific gravity."), "al": ("Urine albumin", "Dataset scale from 0 to 5."), "su": ("Urine sugar", "Dataset scale from 0 to 5."), "rbc": ("Red blood cells (urine microscopy)", "Choose normal or abnormal."), "pc": ("Pus cells (urine microscopy)", "Choose normal or abnormal."), "pcc": ("Pus cell clumps", "Choose present or not present."), "ba": ("Bacteria (urine microscopy)", "Choose present or not present."), "bgr": ("Random blood glucose", "Blood glucose measurement."), "bu": ("Blood urea", "Blood urea measurement."), "sc": ("Serum creatinine", "Serum creatinine measurement."), "sod": ("Sodium", "Sodium measurement."), "pot": ("Potassium", "Potassium measurement."), "hemo": ("Haemoglobin", "Haemoglobin measurement."), "pcv": ("Packed cell volume", "Packed cell volume percentage."), "wc": ("White blood cell count", "White blood cell count."), "rc": ("Red blood cell count", "Red blood cell count."), "htn": ("Hypertension", "Choose yes or no."), "dm": ("Diabetes mellitus", "Choose yes or no."), "cad": ("Coronary artery disease", "Choose yes or no."), "appet": ("Appetite", "Choose good or poor."), "pe": ("Pedal edema", "Choose yes or no."), "ane": ("Anaemia", "Choose yes or no.")},
}
for _key in PARKINSONS_REDUCED_FEATURES:
    _LABELS["parkinsons"] = _LABELS.get("parkinsons", {})
    _LABELS["parkinsons"][_key] = (PARKINSONS_LABELS[_key], "Pre-computed voice-analysis biomarker. Medical AI Suite does not record or process audio.")


def _field_specs() -> dict[str, list[dict]]:
    manifest = load_release_manifest()
    output: dict[str, list[dict]] = {}
    liver_units = {"Age": "years", "TB": "mg/dL", "DB": "mg/dL", "Alkphos": "IU/L", "Sgpt": "IU/L", "Sgot": "IU/L", "TP": "g/dL", "ALB": "g/dL", "A/G Ratio": "ratio"}
    diabetes_units = {"pregnancies": "count", "glucose": "mg/dL", "blood_pressure": "mm Hg", "skin_thickness": "mm", "insulin": "mu U/ml", "bmi": "kg/m²", "diabetes_pedigree": "score", "age": "years"}
    heart_units = {"age": "years", "trestbps": "mm Hg", "chol": "mg/dL", "thalach": "bpm", "oldpeak": "mm"}
    kidney_integer = {"age", "bp", "al", "su", "pcv", "wc"}
    for disease, release in manifest["releases"].items():
        ordered = release["ordered_active_features"]
        if disease == "liver": ranges, units, integer = LIVER_RANGES, liver_units, {"Age", "Alkphos", "Sgpt", "Sgot"}
        elif disease == "diabetes": ranges, units, integer = DIABETES_RANGES, diabetes_units, DIABETES_INTEGER_FIELDS
        elif disease == "heart": ranges, units, integer = HEART_NUMERIC_RANGES, heart_units, {"age", "trestbps", "chol", "thalach"}
        elif disease == "kidney": ranges, units, integer = KIDNEY_NUMERIC_RANGES, KIDNEY_UNITS, kidney_integer
        else: ranges, units, integer = PARKINSONS_RANGES, PARKINSONS_UNITS, set()
        fields = []
        for position, name in enumerate(ordered):
            label, help_text = _LABELS.get(disease, {}).get(name, (name, f"Enter {name}."))
            categories = None
            unit = units.get(name)
            if disease == "liver" and name == "Gender": categories = [{"value": x, "label": x} for x in ("Female", "Male")]
            elif disease == "heart" and name in HEART_CATEGORIES: categories = [{"value": v, "label": HEART_VALUE_LABELS[name][v]} for v in HEART_CATEGORIES[name]]
            elif disease == "kidney" and name in KIDNEY_CATEGORIES: categories = [{"value": v, "label": v.replace("notpresent", "Not present").title()} for v in KIDNEY_CATEGORIES[name]]
            optional = name in LIVER_OPTIONAL if disease == "liver" else name in HEART_OPTIONAL if disease == "heart" else disease == "kidney"
            bounds = ranges.get(name)
            fields.append({"name": name, "label": label, "help": help_text, "unit": unit, "kind": "categorical" if categories else "integer" if name in integer else "number", "required": not optional, "nullable": optional, "min": bounds[0] if bounds else None, "max": bounds[1] if bounds else None, "step": 1 if name in integer else 0.000001 if disease == "parkinsons" else 0.01, "options": categories, "group": _GROUPS[disease].get(name, "Measurements"), "display_order": position, "control": "select" if categories else "number"})
        output[disease] = fields
    return output


def _normalize_heart_category(value):
    # The historical API accepted numeric code tokens but never booleans.
    return value if isinstance(value, bool) else _token(value)


def _normalize_measurements(disease: str, values):
    if not isinstance(values, dict):
        return values
    normalized = dict(values)
    specs = {field["name"]: field for field in FIELD_SPECS[disease]}
    if disease == "kidney":
        for name in KIDNEY_FEATURES:
            if name not in specs:
                lo, hi = (None, None)
                specs[name] = {"name": name, "kind": "categorical" if name in KIDNEY_CATEGORICAL else "integer" if name in {"age", "bp", "al", "su", "pcv", "wc"} else "number", "nullable": True, "min": lo, "max": hi}
    if disease == "parkinsons":
        for name in PARKINSONS_FEATURES:
            if name not in specs:
                specs[name] = {"name": name, "kind": "number", "nullable": True, "min": None, "max": None}
    for name, field in specs.items():
        if name not in normalized:
            continue
        value = normalized[name]
        if field["nullable"] and (value == "" or value == "?"):
            normalized[name] = None
            continue
        if value is None:
            continue
        if field["kind"] == "categorical":
            if disease == "heart":
                normalized[name] = _normalize_heart_category(value)
            elif disease == "kidney":
                normalized[name] = _clean_token(value)
            continue
        if isinstance(value, bool):
            continue
        if disease == "kidney" and isinstance(value, str):
            try:
                value = float(value)
            except ValueError:
                continue
        if field["kind"] == "integer" and isinstance(value, float) and value.is_integer():
            value = int(value)
        normalized[name] = value
    return normalized


class _ContractBase(BaseModel):
    """Shared pre-validation normalization with standard JSON Schema fields."""

    model_config = ConfigDict(extra="forbid")
    _contract_disease: ClassVar[str]

    @model_validator(mode="before")
    @classmethod
    def normalize_contract_values(cls, value):
        return _normalize_measurements(cls._contract_disease, value)


FIELD_SPECS = _field_specs()
DISEASES_ALLOWED = tuple(FIELD_SPECS)


def _make_measurement_model(disease: str) -> type[BaseModel]:
    fields = {}
    for item in FIELD_SPECS[disease]:
        extras = {"ui": {k: item[k] for k in ("label", "help", "unit", "group", "display_order", "control", "kind", "required", "nullable", "min", "max", "step", "options")}}
        field_options = {"title": item["label"], "description": item["help"], "json_schema_extra": extras}
        if item["kind"] != "categorical":
            field_options.update(ge=item["min"], le=item["max"], allow_inf_nan=False)
        if item["kind"] == "categorical":
            values = tuple(option["value"] for option in item["options"])
            annotation = Literal[values]  # type: ignore[valid-type]
        else:
            base = StrictInt if item["kind"] == "integer" else StrictFloat
            annotation = base
        if item["nullable"]:
            annotation = annotation | None
            field_options["default"] = None
        fields[item["name"]] = (annotation, Field(**field_options))
    if disease == "parkinsons":
        # The persisted 15-field feature set is what the wizard displays. Accept
        # the seven known redundant source columns for legacy API callers; they
        # remain optional and are validated, then ignored by the fitted model.
        active = set(PARKINSONS_REDUCED_FEATURES)
        for name in PARKINSONS_FEATURES:
            if name in active:
                continue
            lo, hi = PARKINSONS_RANGES[name]
            label = PARKINSONS_LABELS[name]
            annotation = StrictFloat | None
            unit = PARKINSONS_UNITS.get(name)
            help_text = "Legacy optional biomarker; not used by the released model."
            position = len(fields)
            ui = {"label": label, "help": help_text, "unit": unit, "group": "Compatibility fields", "display_order": position, "control": "number", "kind": "number", "required": False, "nullable": True, "min": lo, "max": hi, "step": 0.000001, "options": None, "display": False}
            fields[name] = (annotation, Field(default=None, title=label, description=help_text, ge=lo, le=hi, allow_inf_nan=False, json_schema_extra={"ui": ui}))
    if disease == "kidney":
        # Preserve accepted optional source columns omitted by the persisted
        # 14-feature pipeline while keeping the assessment wizard on active inputs.
        active = {item["name"] for item in FIELD_SPECS[disease]}
        for name in KIDNEY_FEATURES:
            if name in active:
                continue
            unit = KIDNEY_UNITS.get(name)
            if name in KIDNEY_CATEGORICAL:
                options = tuple(KIDNEY_CATEGORIES[name])
                annotation = Literal[options] | None  # type: ignore[valid-type]
            else:
                lo, hi = KIDNEY_NUMERIC_RANGES[name]
                integer = name in {"age", "bp", "al", "su", "pcv", "wc"}
                base = StrictInt if integer else StrictFloat
                annotation = base | None
            label = name
            help_text = "Legacy optional kidney input; accepted for compatibility and imputed or ignored when inactive."
            lo, hi = KIDNEY_NUMERIC_RANGES[name] if name not in KIDNEY_CATEGORICAL else (None, None)
            options = [{"value": token, "label": token} for token in KIDNEY_CATEGORIES[name]] if name in KIDNEY_CATEGORICAL else None
            integer = name in {"age", "bp", "al", "su", "pcv", "wc"}
            position = len(fields)
            ui = {"label": label, "help": help_text, "unit": unit, "group": "Compatibility fields", "display_order": position, "control": "select" if options else "number", "kind": "categorical" if options else "integer" if integer else "number", "required": False, "nullable": True, "min": lo, "max": hi, "step": 1 if integer else 0.01, "options": options, "display": False}
            field_options = {"default": None, "title": label, "description": help_text, "json_schema_extra": {"ui": ui}}
            if lo is not None:
                field_options.update(ge=lo, le=hi, allow_inf_nan=False)
            fields[name] = (annotation, Field(**field_options))
    model = create_model(f"{disease.title()}Measurements", __base__=_ContractBase, **fields)
    model._contract_disease = disease
    return model


MEASUREMENT_MODELS = {disease: _make_measurement_model(disease) for disease in DISEASES_ALLOWED}


def validate_measurements(disease: str, values: dict) -> dict:
    """Validate and normalize one input mapping without changing feature order."""
    if disease not in MEASUREMENT_MODELS:
        raise ValueError("Unknown disease")
    normalized = MEASUREMENT_MODELS[disease].model_validate(values).model_dump()
    if disease == "parkinsons":
        normalized = {name: normalized[name] for name in PARKINSONS_REDUCED_FEATURES}
    return normalized


def contract_document() -> dict:
    """Stable generator input for frontend contracts."""
    return {"diseases": {disease: {"name": DISEASES[disease].title, "features": FIELD_SPECS[disease]} for disease in DISEASES_ALLOWED}}
