"""Liver (ILPD) module: ML pipeline, API contract, and leakage checks."""
import json
import warnings
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.model_selection import train_test_split

from src.api.main import app
from src.api.services.model_registry import registry
from src.utils.config import DISEASES, RANDOM_STATE, TEST_SIZE

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
CFG = DISEASES["liver"]
MODEL_DIR = ROOT / "models" / "liver"
VALID = {
    "Age": 45, "Gender": "Male", "TB": 1.0, "DB": 0.3, "Alkphos": 190,
    "Sgpt": 25, "Sgot": 30, "TP": 6.8, "ALB": 3.3, "A/G Ratio": 0.9,
}
client = TestClient(app)


# --------------------------- ML pipeline ---------------------------------------
def test_pipeline_artifact_exists_and_loads():
    pipe = registry._model("liver")
    assert set(pipe.named_steps) == {"preprocessor", "model"}


def test_feature_schema_matches_config():
    names = json.loads((MODEL_DIR / "feature_names.json").read_text())
    # every raw feature is represented in the fitted preprocessor output
    assert any("Age" in n for n in names)
    assert any("Gender" in n for n in names)
    assert CFG.target not in CFG.features


def test_prediction_communication_contract():
    out = registry.predict("liver", VALID)
    assert out["prediction"] in (0, 1)
    assert 0.0 <= out["model_score"] <= 1.0
    assert out["model_identifier"].startswith("liver:")
    assert out["decision_threshold"] == 0.5
    assert out["score_type"] == "uncalibrated_model_score"


def test_threshold_result_follows_score():
    out = registry.predict("liver", VALID)
    expected = "at_or_above" if out["model_score"] >= out["decision_threshold"] else "below"
    assert out["threshold_result"] == expected


def test_metadata_is_complete():
    meta = json.loads((MODEL_DIR / "metadata.json").read_text())
    for key in ("dataset_source", "dataset_sha256", "split_strategy", "random_seed",
                "algorithms_evaluated", "hyperparameter_search", "selected_algorithm",
                "cv_metrics", "holdout_metrics", "operating_threshold",
                "permutation_importance_holdout", "limitations", "disclaimer",
                "training_timestamp"):
        assert key in meta, f"metadata missing {key}"
    assert meta["dataset_sha256"] == \
        __import__("hashlib").sha256(CFG.raw_path.read_bytes()).hexdigest()


# --------------------------- API contract -------------------------------------
def test_api_valid_request_succeeds():
    r = client.post("/api/v1/predictions/liver", json={"measurements": VALID})
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["model_score"] <= 1.0
    assert body["model_identifier"]


def test_api_missing_field_rejected():
    bad = {k: v for k, v in VALID.items() if k != "Sgpt"}
    r = client.post("/api/v1/predictions/liver", json={"measurements": bad})
    assert r.status_code == 422
    assert "Sgpt" in r.json()["detail"]


def test_api_optional_ag_ratio_allowed_missing():
    bad = {k: v for k, v in VALID.items() if k != "A/G Ratio"}
    r = client.post("/api/v1/predictions/liver", json={"measurements": bad})
    assert r.status_code == 200


def test_api_wrong_datatype_rejected():
    r = client.post("/api/v1/predictions/liver",
                    json={"measurements": {**VALID, "TB": "high"}})
    assert r.status_code == 422


def test_api_invalid_category_rejected():
    r = client.post("/api/v1/predictions/liver",
                    json={"measurements": {**VALID, "Gender": "Other"}})
    assert r.status_code == 422
    assert "Gender" in r.json()["detail"]


def test_api_out_of_range_rejected():
    r = client.post("/api/v1/predictions/liver",
                    json={"measurements": {**VALID, "Age": 500}})
    assert r.status_code == 422


def test_api_unknown_field_rejected():
    r = client.post("/api/v1/predictions/liver",
                    json={"measurements": {**VALID, "cholesterol": 5}})
    assert r.status_code == 422
    assert "cholesterol" in r.json()["detail"]


def test_api_metadata_endpoint():
    r = client.get("/api/v1/diseases/liver")
    assert r.status_code == 200
    body = r.json()
    assert body["dataset"] == "liver.csv"
    assert body["selected_model"]
    assert "holdout_metrics" in body


# --------------------------- leakage / integrity -----------------------------
def test_no_exact_duplicate_rows_span_the_split():
    raw = pd.read_csv(CFG.raw_path).dropna(subset=[CFG.target])
    frame = raw.loc[~raw.duplicated(keep="first")].reset_index(drop=True)
    assert frame.duplicated().sum() == 0
    X, y = frame[list(CFG.features)], frame[CFG.target]
    Xtr, Xte, _, _ = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    merged = Xtr.merge(Xte.drop_duplicates(), how="inner")
    assert len(merged) == 0, "identical rows appear in both train and test"


def test_target_not_leaked_into_features():
    assert CFG.target not in CFG.features
    pipe = registry._model("liver")
    assert "Selector" not in json.loads((MODEL_DIR / "feature_names.json").read_text())


def test_model_selection_not_on_holdout():
    meta = json.loads((MODEL_DIR / "metadata.json").read_text())
    assert "dev" in meta["split_strategy"].lower() or "cv" in meta["split_strategy"].lower()
    assert meta["model_selection"]["rule"].lower().startswith("max dev")


def test_holdout_auc_matches_metadata_within_tolerance():
    """Retrain-free check: the persisted pipeline reproduces the recorded holdout AUC."""
    from sklearn.metrics import roc_auc_score
    raw = pd.read_csv(CFG.raw_path).dropna(subset=[CFG.target])
    frame = raw.loc[~raw.duplicated(keep="first")].reset_index(drop=True)
    X = frame[list(CFG.features)]
    y = frame[CFG.target].astype(str).str.strip().map({"1": 1, "2": 0}).astype(int)
    _, Xte, _, yte = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    proba = registry._model("liver").predict_proba(Xte)[:, 1]
    recorded = json.loads((MODEL_DIR / "metrics.json").read_text())["holdout"]["roc_auc"]
    assert abs(roc_auc_score(yte, proba) - recorded) < 1e-6
