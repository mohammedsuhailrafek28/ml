"""Chronic Kidney Disease (UCI) module: ML pipeline, API contract, leakage checks.

The historical model scored 1.00 on every metric; these tests lock in the
safeguards that show the (still very high) new score is genuine separability and
not leakage: id-proxy excluded, categorical tokens normalised, no duplicates
across the split, model chosen on dev only, multi-seed stability recorded.
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
from src.preprocessing.kidney_schema import normalize_frame, normalize_target
from src.utils.config import DISEASES, RANDOM_STATE, TEST_SIZE

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
CFG = DISEASES["kidney"]
MODEL_DIR = ROOT / "models" / "kidney"
# A realistic CKD-positive record (subset of fields; rest imputed).
VALID = {
    "age": 48, "bp": 80, "sg": 1.02, "al": 1, "bgr": 121, "bu": 36, "sc": 1.2,
    "hemo": 15.4, "pcv": 44, "rc": 5.2, "htn": "yes", "dm": "yes",
    "appet": "good", "ane": "no",
}
client = TestClient(app)


def _binary_frame():
    raw = pd.read_csv(CFG.raw_path)
    frame = raw.drop(columns=[c for c in ("id",) if c in raw.columns])
    frame = normalize_frame(frame)
    y = normalize_target(raw[CFG.target])
    keep = y.notna()
    frame, y = frame.loc[keep].reset_index(drop=True), y.loc[keep].astype(int).reset_index(drop=True)
    return frame[list(CFG.features)], y


# --------------------------- ML pipeline ---------------------------------------
def test_pipeline_artifact_exists_and_loads():
    pipe = registry._model("kidney")
    assert set(pipe.named_steps) == {"preprocessor", "model"}


def test_dataset_loads_and_ckd_shape():
    raw = pd.read_csv(CFG.raw_path)
    assert raw.shape == (400, 26)
    assert "id" in raw.columns  # present in the file...
    assert "id" not in CFG.features  # ...but never a feature


def test_target_normalisation_handles_trailing_tab():
    raw = pd.read_csv(CFG.raw_path)
    assert (raw[CFG.target].astype(str).str.contains("\t")).any()  # 'ckd\t' rows exist
    y = normalize_target(raw[CFG.target])
    assert set(y.dropna().unique()) == {0, 1}
    assert y.isna().sum() == 0


def test_categorical_tokens_are_clean_in_feature_names():
    names = json.loads((MODEL_DIR / "feature_names.json").read_text())
    assert not any("\t" in n or "  " in n for n in names)
    assert not any(n.endswith("_ yes") or "dm_\t" in n for n in names)


def test_prediction_and_probability():
    out = registry.predict("kidney", VALID)
    assert out["prediction"] in (0, 1)
    assert 0.0 <= out["probability"] <= 1.0
    assert out["selectedModel"] == json.loads((MODEL_DIR / "metrics.json").read_text())["selected_model"]
    assert out["threshold"] == 0.5


def test_prediction_label_follows_threshold():
    out = registry.predict("kidney", VALID)
    expected = "Higher-risk pattern detected" if out["probability"] >= out["threshold"] else "Lower-risk pattern detected"
    assert out["label"] == expected


def test_messy_categorical_tokens_normalise():
    base = registry.predict("kidney", VALID)["probability"]
    messy = registry.predict("kidney", {**VALID, "dm": "\tyes", "htn": " YES ", "appet": "Good"})["probability"]
    assert base == pytest.approx(messy)


def test_top_factors_are_real_features():
    out = registry.predict("kidney", VALID)
    assert out["topFactors"]
    for f in out["topFactors"]:
        assert f["feature"] in CFG.features


def test_metadata_is_complete_and_records_investigations():
    meta = json.loads((MODEL_DIR / "metadata.json").read_text())
    for key in ("dataset_source", "dataset_sha256", "split_strategy", "random_seed",
                "algorithms_evaluated", "hyperparameter_search", "selected_algorithm",
                "cv_metrics", "holdout_metrics", "operating_threshold",
                "permutation_importance_holdout", "limitations", "disclaimer",
                "training_timestamp", "feature_selection_experiment",
                "missingness_analysis", "multi_seed_stability_dev",
                "perfect_score_investigation", "active_features"):
        assert key in meta, f"metadata missing {key}"
    assert meta["dataset_sha256"] == \
        __import__("hashlib").sha256(CFG.raw_path.read_bytes()).hexdigest()
    st = meta["multi_seed_stability_dev"]
    assert set(st["seeds"]) == {21, 42, 84, 123, 2026}
    assert st["min"] <= st["mean"] <= st["max"]


def test_missingness_analysis_flags_the_collection_artefact():
    meta = json.loads((MODEL_DIR / "metadata.json").read_text())
    ma = meta["missingness_analysis"]
    # CKD rows have many more missing labs than non-CKD; count-missing predicts target.
    assert ma["count_missing_features_auc_vs_target"] > 0.7
    assert ma["mean_missing_features_positive"] > ma["mean_missing_features_negative"]
    assert "do NOT add missing-indicator" in ma["note"]


def test_perfect_score_investigation_present_when_score_is_near_perfect():
    meta = json.loads((MODEL_DIR / "metadata.json").read_text())
    hold = meta["holdout_metrics"]
    psi = meta["perfect_score_investigation"]
    if hold["roc_auc"] >= 0.999 or hold["accuracy"] == 1.0:
        assert psi is not None
        assert psi["target_excluded_from_features"] is True
        assert psi["id_proxy_excluded"] is True
        assert psi["all_preprocessing_inside_pipeline"] is True
        assert psi["exact_duplicates_across_partitions"] == 0
        assert "separab" in psi["verdict"].lower()


# --------------------------- API contract -------------------------------------
def test_api_valid_request_succeeds():
    r = client.post("/api/v1/predictions/kidney", json={"measurements": VALID})
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["probability"] <= 1.0


def test_api_request_with_missing_optional_values_ok():
    r = client.post("/api/v1/predictions/kidney",
                    json={"measurements": {"hemo": 9.1, "sc": 4.2, "sg": 1.01, "al": 3}})
    assert r.status_code == 200
    assert 0.0 <= r.json()["probability"] <= 1.0


def test_api_explicit_null_optional_ok():
    r = client.post("/api/v1/predictions/kidney",
                    json={"measurements": {**VALID, "rc": None, "pcv": None}})
    assert r.status_code == 200


def test_api_unknown_field_rejected():
    r = client.post("/api/v1/predictions/kidney",
                    json={"measurements": {**VALID, "egfr": 55}})
    assert r.status_code == 422
    assert "egfr" in r.json()["detail"]


def test_api_wrong_datatype_rejected():
    r = client.post("/api/v1/predictions/kidney",
                    json={"measurements": {**VALID, "hemo": "low"}})
    assert r.status_code == 422


def test_api_invalid_category_rejected():
    r = client.post("/api/v1/predictions/kidney",
                    json={"measurements": {**VALID, "htn": "sometimes"}})
    assert r.status_code == 422
    assert "htn" in r.json()["detail"]


def test_api_impossible_numeric_rejected():
    r = client.post("/api/v1/predictions/kidney",
                    json={"measurements": {**VALID, "hemo": 250}})
    assert r.status_code == 422


def test_api_categorical_whitespace_is_normalised_not_rejected():
    r = client.post("/api/v1/predictions/kidney",
                    json={"measurements": {**VALID, "dm": "\tno", "cad": " no"}})
    assert r.status_code == 200


def test_api_metadata_endpoint():
    r = client.get("/api/v1/diseases/kidney")
    assert r.status_code == 200
    body = r.json()
    assert body["dataset"] == "kidney.csv"
    assert body["selected_model"]
    assert "holdout_metrics" in body


# --------------------------- leakage / integrity -----------------------------
def test_id_is_a_target_proxy_and_is_excluded():
    raw = pd.read_csv(CFG.raw_path)
    y = normalize_target(raw[CFG.target])
    # rows are sorted by class: every id < 250 is ckd, every id >= 250 is notckd
    assert (y[raw["id"] < 250] == 1).all()
    assert (y[raw["id"] >= 250] == 0).all()
    assert "id" not in CFG.features
    assert "id" not in json.loads((MODEL_DIR / "feature_names.json").read_text())
    assert not any("id" == n.split("__")[-1] for n in json.loads((MODEL_DIR / "feature_names.json").read_text()))


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
    meta = json.loads((MODEL_DIR / "metadata.json").read_text())
    active = meta["active_features"]
    X, y = _binary_frame()
    _, Xte, _, yte = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    proba = registry._model("kidney").predict_proba(Xte[active])[:, 1]
    recorded = meta["holdout_metrics"]["roc_auc"]
    assert abs(roc_auc_score(yte, proba) - recorded) < 1e-6


# --------------------------- integration -------------------------------------
def test_frontend_style_payload_round_trip():
    form_payload = {
        "age": 60, "bp": 90, "sg": 1.01, "al": 3, "bgr": 250, "bu": 90, "sc": 4.5,
        "hemo": 9.2, "pcv": 28, "rc": 3.1, "htn": "yes", "dm": "yes",
        "appet": "poor", "ane": "yes",
    }
    r = client.post("/api/v1/predictions/kidney", json={"measurements": form_payload})
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["probability"] <= 1.0
    assert body["label"] in ("Higher-risk pattern detected", "Lower-risk pattern detected")
    assert body["threshold"] == 0.5
