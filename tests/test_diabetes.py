"""Diabetes (Pima Indians) module: ML pipeline, API contract, leakage checks.

Extra attention to the zero-as-missing contract: clinically-impossible zeros in
glucose / blood_pressure / skin_thickness / insulin / bmi must be turned into
NaN *inside* the persisted pipeline, so training and inference behave identically.
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.model_selection import train_test_split

from src.api.main import app
from src.api.services.model_registry import registry
from src.model_release import training_metadata_dataset_sha256
from src.preprocessing.diabetes_schema import ZeroToNaN
from src.utils.config import DISEASES, RANDOM_STATE, TEST_SIZE

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
CFG = DISEASES["diabetes"]
MODEL_DIR = ROOT / "models" / "diabetes"
VALID = {
    "pregnancies": 6, "glucose": 148, "blood_pressure": 72, "skin_thickness": 35,
    "insulin": 0, "bmi": 33.6, "diabetes_pedigree": 0.627, "age": 50,
}
client = TestClient(app)


def _binary_frame():
    raw = pd.read_csv(CFG.raw_path).dropna(subset=[CFG.target])
    frame = raw.loc[~raw.duplicated(keep="first")].reset_index(drop=True)
    X = frame[list(CFG.features)]
    y = pd.to_numeric(frame[CFG.target], errors="coerce").astype(int)
    return X, y


# --------------------------- ML pipeline ---------------------------------------
def test_dataset_loads_and_pima_shape():
    raw = pd.read_csv(CFG.raw_path)
    assert raw.shape == (768, 9)
    assert set(pd.to_numeric(raw[CFG.target]).unique()) == {0, 1}


def test_pipeline_has_zero_to_nan_first():
    pipe = registry._model("diabetes")
    assert list(pipe.named_steps)[:2] == ["zero_to_nan", "interactions"] or \
           list(pipe.named_steps)[0] == "zero_to_nan"
    assert "model" in pipe.named_steps


def test_zero_to_nan_transformer_semantics():
    z = ZeroToNaN()
    df = pd.DataFrame([{**VALID}])
    out = z.transform(df)
    assert np.isnan(out.loc[0, "insulin"])       # 0 -> NaN (missing placeholder)
    assert out.loc[0, "pregnancies"] == 6         # genuine value untouched
    df0 = pd.DataFrame([{**VALID, "pregnancies": 0, "glucose": 0}])
    out0 = z.transform(df0)
    assert out0.loc[0, "pregnancies"] == 0        # nulliparous stays 0
    assert np.isnan(out0.loc[0, "glucose"])       # impossible 0 -> NaN


def test_feature_schema_and_target_not_leaked():
    assert CFG.target not in CFG.features
    names = json.loads((MODEL_DIR / "feature_names.json").read_text())
    assert not any("target" in n for n in names)
    assert any("glucose" in n for n in names)


def test_prediction_communication_contract():
    out = registry.predict("diabetes", VALID)
    assert out["prediction"] in (0, 1)
    assert 0.0 <= out["model_score"] <= 1.0
    assert out["model_identifier"].startswith("diabetes:")
    assert out["decision_threshold"] == 0.5
    assert out["score_type"] == "uncalibrated_model_score"


def test_threshold_result_follows_score():
    out = registry.predict("diabetes", VALID)
    expected = "at_or_above" if out["model_score"] >= out["decision_threshold"] else "below"
    assert out["threshold_result"] == expected


def test_metadata_is_complete():
    meta = json.loads((MODEL_DIR / "metadata.json").read_text())
    for key in ("dataset_source", "dataset_sha256", "split_strategy", "random_seed",
                "algorithms_evaluated", "hyperparameter_search", "selected_algorithm",
                "cv_metrics", "holdout_metrics", "operating_threshold",
                "permutation_importance_holdout", "limitations", "disclaimer",
                "training_timestamp", "zero_as_missing_columns", "feature_engineering"):
        assert key in meta, f"metadata missing {key}"
    assert meta["dataset_sha256"] == \
        training_metadata_dataset_sha256(CFG.raw_path)
    assert meta["zero_as_missing_columns"] == \
        ["glucose", "blood_pressure", "skin_thickness", "insulin", "bmi"]


def test_feature_engineering_experiment_recorded():
    meta = json.loads((MODEL_DIR / "metadata.json").read_text())
    fe = meta["feature_engineering"]
    assert "probes" in fe and "mean_gain" in fe and "adopted" in fe
    # engineered_features list must agree with the adopt decision
    assert bool(meta["engineered_features"]) == bool(fe["adopted"])


# --------------------------- API contract -------------------------------------
def test_api_valid_request_succeeds():
    r = client.post("/api/v1/predictions/diabetes", json={"measurements": VALID})
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["model_score"] <= 1.0
    assert body["model_identifier"]


def test_api_missing_field_rejected():
    bad = {k: v for k, v in VALID.items() if k != "bmi"}
    r = client.post("/api/v1/predictions/diabetes", json={"measurements": bad})
    assert r.status_code == 422
    assert "bmi" in r.json()["detail"]


def test_api_unknown_field_rejected():
    r = client.post("/api/v1/predictions/diabetes",
                    json={"measurements": {**VALID, "hba1c": 5.5}})
    assert r.status_code == 422
    assert "hba1c" in r.json()["detail"]


def test_api_wrong_datatype_rejected():
    r = client.post("/api/v1/predictions/diabetes",
                    json={"measurements": {**VALID, "glucose": "high"}})
    assert r.status_code == 422


def test_api_negative_value_rejected():
    r = client.post("/api/v1/predictions/diabetes",
                    json={"measurements": {**VALID, "age": -3}})
    assert r.status_code == 422


def test_api_out_of_range_rejected():
    r = client.post("/api/v1/predictions/diabetes",
                    json={"measurements": {**VALID, "bmi": 500}})
    assert r.status_code == 422


def test_api_fractional_pregnancies_rejected():
    r = client.post("/api/v1/predictions/diabetes",
                    json={"measurements": {**VALID, "pregnancies": 2.5}})
    assert r.status_code == 422


def test_api_zero_glucose_is_accepted_and_treated_as_missing():
    """0 is the dataset's 'not measured' placeholder -> accepted, imputed."""
    r0 = client.post("/api/v1/predictions/diabetes",
                     json={"measurements": {**VALID, "glucose": 0}})
    assert r0.status_code == 200
    p_missing = r0.json()["model_score"]
    p_real_low = registry.predict("diabetes", {**VALID, "glucose": 1})["model_score"]
    assert p_missing != p_real_low, "glucose=0 must not be scaled as a real value"


def test_api_metadata_endpoint():
    r = client.get("/api/v1/diseases/diabetes")
    assert r.status_code == 200
    body = r.json()
    assert body["dataset"] == "diabetes.csv"
    assert body["selected_model"]
    assert "holdout_metrics" in body


# --------------------------- leakage / integrity -----------------------------
def test_no_exact_duplicate_rows_span_the_split():
    X, y = _binary_frame()
    assert X.assign(_t=y).duplicated().sum() == 0
    Xtr, Xte, _, _ = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    assert len(Xtr.merge(Xte.drop_duplicates(), how="inner")) == 0


def test_model_selection_not_on_holdout():
    meta = json.loads((MODEL_DIR / "metadata.json").read_text())
    assert meta["model_selection"]["rule"].lower().startswith("max dev")
    assert "dev" in meta["split_strategy"].lower()


def test_holdout_auc_matches_metadata_within_tolerance():
    from sklearn.metrics import roc_auc_score
    X, y = _binary_frame()
    _, Xte, _, yte = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    proba = registry._model("diabetes").predict_proba(Xte)[:, 1]
    recorded = json.loads((MODEL_DIR / "metrics.json").read_text())["holdout"]["roc_auc"]
    assert abs(roc_auc_score(yte, proba) - recorded) < 1e-6


# --------------------------- integration -------------------------------------
def test_frontend_style_payload_round_trip():
    form_payload = {
        "pregnancies": 1, "glucose": 89, "blood_pressure": 66, "skin_thickness": 23,
        "insulin": 94, "bmi": 28.1, "diabetes_pedigree": 0.167, "age": 21,
    }
    r = client.post("/api/v1/predictions/diabetes", json={"measurements": form_payload})
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["model_score"] <= 1.0
    assert body["threshold_result"] in ("at_or_above", "below")
    assert body["decision_threshold"] == 0.5
