"""Heart (UCI Cleveland) module: ML pipeline, API contract, and leakage checks.

Extra attention to ``ca`` and ``thal`` -- previously a train/serve mismatch,
because they reached the one-hot encoder as raw floats in training but as JSON
ints/strings at inference (silently encoded as all-zeros instead of raising).
"""
import json
import warnings
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.model_selection import train_test_split

from src.api.main import app
from src.model_release import training_metadata_dataset_sha256
from src.api.services.model_registry import registry
from src.preprocessing.heart_schema import normalize_categoricals
from src.utils.config import DISEASES, RANDOM_STATE, TEST_SIZE

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
CFG = DISEASES["heart"]
MODEL_DIR = ROOT / "models" / "heart"
VALID = {
    "age": 63, "sex": "1", "cp": "1", "trestbps": 145, "chol": 233, "fbs": "1",
    "restecg": "2", "thalach": 150, "exang": "0", "oldpeak": 2.3, "slope": "3",
    "ca": "0", "thal": "6",
}
client = TestClient(app)


def _load_binary_frame():
    raw = pd.read_csv(CFG.raw_path, na_values=["?", "NA", "N/A", ""]).dropna(subset=[CFG.target])
    frame = normalize_categoricals(raw)
    frame = frame.loc[~frame.duplicated(keep="first")].reset_index(drop=True)
    X = frame[list(CFG.features)]
    y = (pd.to_numeric(frame[CFG.target], errors="coerce") > 0).astype(int)
    return X, y


# --------------------------- ML pipeline ---------------------------------------
def test_pipeline_artifact_exists_and_loads():
    pipe = registry._model("heart")
    assert set(pipe.named_steps) == {"preprocessor", "model"}


def test_dataset_loads_and_is_cleveland_shape():
    raw = pd.read_csv(CFG.raw_path, na_values=["?"])
    assert raw.shape == (303, 14)
    assert list(raw.columns)[-1] == CFG.target


def test_feature_schema_matches_config():
    names = json.loads((MODEL_DIR / "feature_names.json").read_text())
    assert any("age" in n for n in names)
    assert any("thal" in n for n in names)
    assert any("ca" in n for n in names)
    assert CFG.target not in CFG.features


def test_prediction_communication_contract():
    out = registry.predict("heart", VALID)
    assert out["prediction"] in (0, 1)
    assert 0.0 <= out["model_score"] <= 1.0
    assert out["model_identifier"].startswith("heart:")
    assert out["decision_threshold"] == 0.5
    assert out["score_type"] == "uncalibrated_model_score"


def test_threshold_result_follows_score():
    out = registry.predict("heart", VALID)
    expected = "at_or_above" if out["model_score"] >= out["decision_threshold"] else "below"
    assert out["threshold_result"] == expected


def test_metadata_is_complete():
    meta = json.loads((MODEL_DIR / "metadata.json").read_text())
    for key in ("dataset_source", "dataset_sha256", "split_strategy", "random_seed",
                "algorithms_evaluated", "hyperparameter_search", "selected_algorithm",
                "cv_metrics", "holdout_metrics", "operating_threshold",
                "permutation_importance_holdout", "limitations", "disclaimer",
                "training_timestamp", "categorical_values"):
        assert key in meta, f"metadata missing {key}"
    assert meta["dataset_sha256"] == \
        training_metadata_dataset_sha256(CFG.raw_path)
    assert meta["target_mapping"] == {"0": 0, "1-4": 1}


# --------------------------- API contract -------------------------------------
def test_api_valid_request_succeeds():
    r = client.post("/api/v1/predictions/heart", json={"measurements": VALID})
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["model_score"] <= 1.0
    assert body["model_identifier"]


def test_api_missing_field_rejected():
    bad = {k: v for k, v in VALID.items() if k != "trestbps"}
    r = client.post("/api/v1/predictions/heart", json={"measurements": bad})
    assert r.status_code == 422
    assert "trestbps" in r.json()["detail"]


def test_api_optional_ca_thal_allowed_missing():
    bad = {k: v for k, v in VALID.items() if k not in ("ca", "thal")}
    r = client.post("/api/v1/predictions/heart", json={"measurements": bad})
    assert r.status_code == 200


def test_api_wrong_datatype_rejected():
    r = client.post("/api/v1/predictions/heart",
                    json={"measurements": {**VALID, "chol": "high"}})
    assert r.status_code == 422


def test_api_invalid_cp_category_rejected():
    r = client.post("/api/v1/predictions/heart",
                    json={"measurements": {**VALID, "cp": 9}})
    assert r.status_code == 422
    assert "cp" in r.json()["detail"]


def test_api_invalid_thal_category_rejected():
    # 2 is a Kaggle-encoding value; the raw Cleveland schema only allows 3/6/7.
    r = client.post("/api/v1/predictions/heart",
                    json={"measurements": {**VALID, "thal": 2}})
    assert r.status_code == 422
    assert "thal" in r.json()["detail"]


def test_api_out_of_range_rejected():
    r = client.post("/api/v1/predictions/heart",
                    json={"measurements": {**VALID, "age": 500}})
    assert r.status_code == 422


def test_api_unknown_field_rejected():
    r = client.post("/api/v1/predictions/heart",
                    json={"measurements": {**VALID, "smoker": 1}})
    assert r.status_code == 422
    assert "smoker" in r.json()["detail"]


def test_api_metadata_endpoint():
    r = client.get("/api/v1/diseases/heart")
    assert r.status_code == 200
    body = r.json()
    assert body["dataset"] == "heart.csv"
    assert body["selected_model"]
    assert "holdout_metrics" in body
    assert "categorical_values" in body


# --------------------------- ca / thal encoding (train == serve) --------------
def test_ca_thal_string_int_float_are_equivalent():
    base = registry.predict("heart", VALID)["model_score"]
    as_int = registry.predict("heart", {**VALID, "ca": 0, "thal": 6})["model_score"]
    as_float = registry.predict("heart", {**VALID, "ca": 0.0, "thal": 6.0})["model_score"]
    as_floatstr = registry.predict("heart", {**VALID, "ca": "0.0", "thal": "6.0"})["model_score"]
    assert base == pytest.approx(as_int) == pytest.approx(as_float) == pytest.approx(as_floatstr)


def test_changing_ca_changes_model_score():
    """A different (valid) ca value must actually flow through the encoder."""
    p0 = registry.predict("heart", {**VALID, "ca": "0"})["model_score"]
    p3 = registry.predict("heart", {**VALID, "ca": "3"})["model_score"]
    assert p0 != p3, "ca is being swallowed (all-zero one-hot) instead of encoded"


def test_feature_names_have_expected_ca_thal_levels():
    names = json.loads((MODEL_DIR / "feature_names.json").read_text())
    for lvl in ("0", "1", "2", "3"):
        assert any(n.endswith(f"ca_{lvl}") for n in names)
    for lvl in ("3", "6", "7"):
        assert any(n.endswith(f"thal_{lvl}") for n in names)
    assert not any("ca_?" in n or "thal_?" in n for n in names)


# --------------------------- leakage / integrity -----------------------------
def test_no_exact_duplicate_rows_span_the_split():
    X, y = _load_binary_frame()
    assert X.assign(_t=y).duplicated().sum() == 0
    Xtr, Xte, _, _ = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    merged = Xtr.merge(Xte.drop_duplicates(), how="inner")
    assert len(merged) == 0, "identical rows appear in both train and test"


def test_target_not_leaked_into_features():
    assert CFG.target not in CFG.features
    assert "target" not in json.loads((MODEL_DIR / "feature_names.json").read_text())


def test_model_selection_not_on_holdout():
    meta = json.loads((MODEL_DIR / "metadata.json").read_text())
    assert "dev" in meta["split_strategy"].lower() or "cv" in meta["split_strategy"].lower()
    assert meta["model_selection"]["rule"].lower().startswith("max dev")


def test_holdout_auc_matches_metadata_within_tolerance():
    from sklearn.metrics import roc_auc_score
    X, y = _load_binary_frame()
    _, Xte, _, yte = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    proba = registry._model("heart").predict_proba(Xte)[:, 1]
    recorded = json.loads((MODEL_DIR / "metrics.json").read_text())["holdout"]["roc_auc"]
    assert abs(roc_auc_score(yte, proba) - recorded) < 1e-6


# --------------------------- integration -------------------------------------
def test_frontend_style_payload_round_trip():
    """Categoricals sent as strings (as the Next.js form does) must predict."""
    form_payload = {
        "age": 55, "sex": "1", "cp": "4", "trestbps": 130, "chol": 246, "fbs": "0",
        "restecg": "2", "thalach": 150, "exang": "1", "oldpeak": 1.0, "slope": "2",
        "ca": "1", "thal": "7",
    }
    r = client.post("/api/v1/predictions/heart", json={"measurements": form_payload})
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["model_score"] <= 1.0
    assert body["threshold_result"] in ("at_or_above", "below")
    assert body["decision_threshold"] == 0.5
