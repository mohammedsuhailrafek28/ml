import json
import pandas as pd
from joblib import load
from src.utils.config import DISEASES
from src.preprocessing.heart_schema import (
    HEART_CATEGORICAL,
    HEART_CATEGORIES,
    HEART_NUMERIC_RANGES,
    HEART_OPTIONAL,
    normalize_measurements,
    _token,
)
from src.preprocessing.diabetes_schema import (
    DIABETES_FEATURES,
    DIABETES_INTEGER_FIELDS,
    DIABETES_RANGES,
)
from src.preprocessing.kidney_schema import (
    KIDNEY_CATEGORICAL,
    KIDNEY_CATEGORIES,
    KIDNEY_NUMERIC,
    KIDNEY_NUMERIC_RANGES,
    _clean_token,
    normalize_measurements as normalize_kidney_measurements,
)
from src.preprocessing.parkinsons_schema import (
    PARKINSONS_FEATURES,
    PARKINSONS_RANGES,
    PARKINSONS_REDUCED_FEATURES,
)

# Plausible input ranges for validation. Values outside these are rejected with a
# 422 so obviously wrong entries (typos, wrong units) do not reach the model.
# Ranges are generous supersets of the ILPD observed spread, not clinical limits.
LIVER_RANGES = {
    "Age": (1, 120),
    "TB": (0.0, 100.0),        # total bilirubin, mg/dL
    "DB": (0.0, 60.0),         # direct bilirubin, mg/dL
    "Alkphos": (10, 3000),     # alkaline phosphatase, IU/L
    "Sgpt": (1, 3000),         # ALT, IU/L
    "Sgot": (1, 6000),         # AST, IU/L
    "TP": (1.0, 12.0),         # total proteins, g/dL
    "ALB": (0.5, 7.0),         # albumin, g/dL
    "A/G Ratio": (0.0, 5.0),
}
LIVER_GENDER = {"Male", "Female"}
# A/G Ratio is the only feature allowed to be missing (imputed by the pipeline).
LIVER_OPTIONAL = {"A/G Ratio"}


class Registry:
    def __init__(self):
        self.models = {}

    def _config(self, k):
        if k not in DISEASES:
            raise KeyError("Unknown disease")
        return DISEASES[k]

    def _dir(self, k):
        # Honour MODEL_ROOT (src.api.settings) with the repo `models/` dir as the
        # default, so deployments can mount artifacts elsewhere without code changes.
        try:
            from src.api.settings import get_settings

            return get_settings().model_root / k
        except Exception:  # noqa: BLE001 - never break inference over config import
            return self._config(k).model_dir

    def _model(self, k):
        if k not in self.models:
            self.models[k] = load(self._dir(k) / f"{k}_pipeline.joblib")
        return self.models[k]

    def _meta_file(self, k):
        p = self._dir(k) / "metadata.json"
        return json.loads(p.read_text()) if p.exists() else {}

    def available(self):
        return [k for k in DISEASES
                if (self._dir(k) / f"{k}_pipeline.joblib").exists()]

    def threshold(self, k):
        return float(self._meta_file(k).get("operating_threshold",
               self._meta_file(k).get("threshold", 0.5)))

    def active_features(self, k):
        """Ordered raw request features consumed by the persisted pipeline."""
        return list(self._meta_file(k).get("active_features", self._config(k).features))

    def metadata(self, k):
        c = self._config(k)
        m = json.loads((self._dir(k) / "metrics.json").read_text())
        meta = self._meta_file(k)
        out = {
            "slug": k,
            "name": c.title,
            "dataset": c.raw_path.name,
            "selected_model": m["selected_model"],
            "input_features": list(c.features),
            "target": c.target,
        }
        # Extra, non-breaking context for the UI / report when the trainer wrote it.
        for key in ("dataset_source", "dataset_url", "dataset_sha256",
                    "operating_threshold", "holdout_metrics", "cv_metrics",
                    "limitations", "disclaimer", "permutation_importance_holdout",
                    "rows_used", "random_seed", "categorical_values",
                    "numeric_features", "categorical_features"):
            if key in meta:
                out[key] = meta[key]
        return out

    def catalog(self):
        return [self.metadata(k) for k in self.available()]

    def _validate_liver(self, values):
        missing = [f for f in DISEASES["liver"].features
                   if f not in LIVER_OPTIONAL
                   and (f not in values or values[f] is None or values[f] == "")]
        if missing:
            raise ValueError(f"Missing required fields: {sorted(missing)}")
        gender = values.get("Gender")
        if gender not in LIVER_GENDER:
            raise ValueError(f"Gender must be one of {sorted(LIVER_GENDER)}")
        for field, (lo, hi) in LIVER_RANGES.items():
            v = values.get(field)
            if v is None or v == "":
                continue
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                raise ValueError(f"{field} must be a number")
            if not (lo <= v <= hi):
                raise ValueError(f"{field} must be between {lo} and {hi}")

    def _validate_heart(self, values):
        # Required = every feature except the ones the pipeline can impute.
        missing = [f for f in DISEASES["heart"].features
                   if f not in HEART_OPTIONAL
                   and (f not in values or values[f] is None or values[f] == "")]
        if missing:
            raise ValueError(f"Missing required fields: {sorted(missing)}")
        for field, (lo, hi) in HEART_NUMERIC_RANGES.items():
            v = values.get(field)
            if v is None or v == "":
                raise ValueError(f"{field} is required")
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                raise ValueError(f"{field} must be a number")
            if not (lo <= v <= hi):
                raise ValueError(f"{field} must be between {lo} and {hi}")
        for field in HEART_CATEGORICAL:
            v = values.get(field)
            if v is None or v == "":
                if field in HEART_OPTIONAL:
                    continue
                raise ValueError(f"{field} is required")
            token = _token(v)
            if token not in HEART_CATEGORIES[field]:
                raise ValueError(
                    f"{field} must be one of {HEART_CATEGORIES[field]} "
                    f"(got {v!r})"
                )

    def _validate_parkinsons(self, values):
        # Inputs are pre-computed voice biomarkers. The production model uses the
        # 15 non-redundant measures; those are required. The 7 near-duplicate
        # measures (RAP/PPQ/DDP/Shimmer/APQ3/APQ5/DDA) are accepted if sent.
        for bad in ("name", "status"):
            if bad in values:
                raise ValueError(f"{bad!r} is an identifier/label, not a model input")
        required = list(PARKINSONS_REDUCED_FEATURES)
        missing = [f for f in required
                   if f not in values or values[f] is None or values[f] == ""]
        if missing:
            raise ValueError(f"Missing required voice biomarkers: {sorted(missing)}")
        for field in PARKINSONS_FEATURES:
            if field not in values or values[field] in (None, ""):
                continue
            v = values[field]
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                raise ValueError(f"{field} must be a number")
            lo, hi = PARKINSONS_RANGES[field]
            if not (lo <= v <= hi):
                raise ValueError(f"{field} must be between {lo} and {hi}")

    def _validate_kidney(self, values):
        # Every CKD feature is optional (the UCI file is heavily incomplete and the
        # pipeline imputes). We only reject values that are present but wrong.
        for field, v in values.items():
            if v is None or v == "":
                continue
            if field in KIDNEY_NUMERIC:
                if isinstance(v, bool):
                    raise ValueError(f"{field} must be a number")
                try:
                    num = float(v)  # tolerate "44" from clients that send strings
                except (TypeError, ValueError):
                    raise ValueError(f"{field} must be a number")
                lo, hi = KIDNEY_NUMERIC_RANGES[field]
                if not (lo <= num <= hi):
                    raise ValueError(f"{field} must be between {lo} and {hi}")
            elif field in KIDNEY_CATEGORICAL:
                token = _clean_token(v)
                if token not in KIDNEY_CATEGORIES[field]:
                    raise ValueError(
                        f"{field} must be one of {KIDNEY_CATEGORIES[field]} (got {v!r})"
                    )

    def _validate_diabetes(self, values):
        missing = [f for f in DIABETES_FEATURES
                   if f not in values or values[f] is None or values[f] == ""]
        if missing:
            raise ValueError(f"Missing required fields: {sorted(missing)}")
        for field in DIABETES_FEATURES:
            v = values[field]
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                raise ValueError(f"{field} must be a number")
            if v < 0:
                raise ValueError(f"{field} cannot be negative")
            if field in DIABETES_INTEGER_FIELDS and float(v) != int(v):
                raise ValueError(f"{field} must be a whole number")
            lo, hi = DIABETES_RANGES[field]
            if not (lo <= v <= hi):
                raise ValueError(f"{field} must be between {lo} and {hi}")
        # glucose / blood_pressure / skin_thickness / insulin / bmi may be 0:
        # that is the dataset's "not measured" placeholder and the pipeline's
        # ZeroToNaN step converts it to a NaN that median-imputation then fills.

    def predict(self, k, values):
        c = self._config(k)
        unknown = set(values) - set(c.features)
        if unknown:
            raise ValueError(f"Unexpected fields: {sorted(unknown)}")
        if k == "liver":
            self._validate_liver(values)
        elif k == "heart":
            self._validate_heart(values)
            values = normalize_measurements(values)
        elif k == "diabetes":
            self._validate_diabetes(values)
        elif k == "kidney":
            self._validate_kidney(values)
            values = normalize_kidney_measurements(values)
        elif k == "parkinsons":
            self._validate_parkinsons(values)
        frame = pd.DataFrame([{f: values.get(f) for f in c.features}])
        active = self._meta_file(k).get("active_features")
        if active:
            # Some trainers (kidney, parkinsons) select a feature subset. The
            # kidney pipeline selects by name via ColumnTransformer, but the
            # parkinsons pipeline is a bare imputer+scaler+model, so we must hand
            # it exactly the columns it was fitted on, in order.
            frame = frame[[f for f in active if f in frame.columns]]
        model = self._model(k)
        thr = self.threshold(k)
        prob = (float(model.predict_proba(frame)[0, 1])
                if hasattr(model, "predict_proba") else None)
        if k in ("liver", "heart", "diabetes", "kidney", "parkinsons") and prob is not None:
            pred = int(prob >= thr)
        else:
            pred = int(model.predict(frame)[0])
        meta = self._meta_file(k)
        factors = [
            {"feature": d["feature"], "importance": round(d["importance_mean"], 4)}
            for d in meta.get("permutation_importance_holdout", [])[:5]
        ]
        return {
            "disease": k,
            "prediction": pred,
            "label": "Higher-risk pattern detected" if pred else "Lower-risk pattern detected",
            "probability": prob,
            "threshold": thr,
            "selectedModel": self.metadata(k)["selected_model"],
            "modelMetrics": meta.get("holdout_metrics"),
            "topFactors": factors,
            "limitations": meta.get("limitations", ["Educational model only", "Not clinically validated"]),
            "disclaimer": meta.get("disclaimer", "This prediction is not a medical diagnosis."),
        }


registry = Registry()
