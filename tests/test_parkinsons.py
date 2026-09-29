"""Parkinson's (UCI voice) module: subject-aware ML, API contract, leakage checks.

The dataset has ~6 recordings per subject, so the central risk is subject
leakage. These tests lock in: correct subject-id extraction, zero group overlap
between DEV and HOLDOUT and inside every CV fold, `name`/`status` never features,
and model selection on DEV group-CV rather than the holdout.
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from src.api.main import app
from src.api.services.model_registry import registry
from src.model_release import training_metadata_dataset_sha256
from src.preprocessing.parkinsons_schema import (
    PARKINSONS_FEATURES,
    PARKINSONS_REDUCED_FEATURES,
    subject_id,
    subject_ids,
)
from src.utils.config import DISEASES, RANDOM_STATE, TEST_SIZE

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
CFG = DISEASES["parkinsons"]
MODEL_DIR = ROOT / "models" / "parkinsons"
FULL = {
    "MDVP:Fo(Hz)": 119.992, "MDVP:Fhi(Hz)": 157.302, "MDVP:Flo(Hz)": 74.997,
    "MDVP:Jitter(%)": 0.00784, "MDVP:Jitter(Abs)": 7e-05, "MDVP:RAP": 0.0037,
    "MDVP:PPQ": 0.00554, "Jitter:DDP": 0.01109, "MDVP:Shimmer": 0.04374,
    "MDVP:Shimmer(dB)": 0.426, "Shimmer:APQ3": 0.02182, "Shimmer:APQ5": 0.0313,
    "MDVP:APQ": 0.02971, "Shimmer:DDA": 0.06545, "NHR": 0.02211, "HNR": 21.033,
    "RPDE": 0.414783, "DFA": 0.815285, "spread1": -4.813031, "spread2": 0.266482,
    "D2": 2.301442, "PPE": 0.284654,
}
REDUCED = {k: FULL[k] for k in PARKINSONS_REDUCED_FEATURES}
client = TestClient(app)


def _load():
    raw = pd.read_csv(CFG.raw_path)
    X = raw[list(PARKINSONS_FEATURES)]
    y = pd.to_numeric(raw["status"]).astype(int)
    g = subject_ids(raw["name"])
    return raw, X, y, g


# --------------------------- subject leakage ---------------------------------
def test_subject_id_extractor():
    assert subject_id("phon_R01_S07_4") == "phon_R01_S07"
    assert subject_id("phon_R27_S33_6") == "phon_R27_S33"
    # recordings of one subject collapse to one group; different subjects differ
    assert subject_id("phon_R01_S07_1") == subject_id("phon_R01_S07_6")
    assert subject_id("phon_R01_S07_1") != subject_id("phon_R01_S08_1")


def test_group_extraction_covers_all_rows_uniquely():
    raw, _, y, g = _load()
    assert g.notna().all() and (g != "").all()
    assert g.nunique() == 32
    # every subject has a single status label
    assert raw.assign(_g=g, _y=y).groupby("_g")["_y"].nunique().max() == 1


def test_name_and_status_absent_from_features():
    assert "name" not in CFG.features
    assert "status" not in CFG.features
    names = json.loads((MODEL_DIR / "feature_names.json").read_text())
    assert "name" not in names and "status" not in names


def test_dev_holdout_have_zero_subject_overlap():
    _, X, y, g = _load()
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    dev, hold = next(gss.split(X, y, g))
    assert set(g.iloc[dev]) & set(g.iloc[hold]) == set()
    assert g.iloc[hold].nunique() >= 2
    assert y.iloc[hold].nunique() == 2  # both classes present in holdout


def test_every_cv_fold_has_zero_subject_overlap():
    _, X, y, g = _load()
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    dev, _ = next(gss.split(X, y, g))
    Xd, yd, gd = X.iloc[dev], y.iloc[dev], g.iloc[dev]
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for tr, va in sgkf.split(Xd, yd, gd):
        assert set(gd.iloc[tr]) & set(gd.iloc[va]) == set()


# --------------------------- ML pipeline ------------------------------------
def test_pipeline_artifact_loads():
    pipe = registry._model("parkinsons")
    assert list(pipe.named_steps) == ["imputer", "scaler", "model"]


def test_prediction_communication_contract():
    out = registry.predict("parkinsons", REDUCED)
    assert out["prediction"] in (0, 1)
    assert 0.0 <= out["model_score"] <= 1.0
    assert out["model_identifier"].startswith("parkinsons:")
    assert out["decision_threshold"] == 0.5
    assert out["score_type"] == "uncalibrated_model_score"
    assert out["release_status"] == "experimental"
    assert "pre-computed voice biomarkers" in " ".join(out["limitations"])
    assert "195 recordings from 32 subjects (8 controls)" in " ".join(out["limitations"])
    assert "ROC-AUC 0.586 and specificity 0.00" in " ".join(out["limitations"])
    assert "not a probability" in out["disclaimer"]


def test_full_and_reduced_payload_agree():
    a = registry.predict("parkinsons", FULL)["model_score"]
    b = registry.predict("parkinsons", REDUCED)["model_score"]
    assert a == pytest.approx(b)  # model only consumes the active feature subset


def test_metadata_records_subject_aware_methodology():
    meta = json.loads((MODEL_DIR / "metadata.json").read_text())
    for key in ("dataset_source", "dataset_sha256", "unique_subjects",
                "target_distribution_by_subject", "subject_id_rule",
                "split_strategy", "group_cv_strategy", "dev_holdout_subjects",
                "naive_vs_subject_aware", "feature_experiment",
                "multi_seed_group_stability_dev", "model_selection",
                "holdout_metrics", "permutation_importance_holdout",
                "limitations", "disclaimer", "active_features"):
        assert key in meta, f"metadata missing {key}"
    assert meta["unique_subjects"] == 32
    assert meta["identifier_excluded"] == ["name", "status"]
    assert "GroupShuffleSplit" in meta["split_strategy"]
    assert "StratifiedGroupKFold" in meta["group_cv_strategy"]
    # the stale "no group split implemented" claim must be gone
    assert "no group split" not in json.dumps(meta).lower()
    assert meta["dataset_sha256"] == \
        training_metadata_dataset_sha256(CFG.raw_path)


def test_naive_vs_subject_aware_gap_is_recorded_and_large():
    meta = json.loads((MODEL_DIR / "metadata.json").read_text())
    nv = meta["naive_vs_subject_aware"]
    assert nv["naive_row_split_holdout_roc_auc"] > nv["subject_aware_split_holdout_roc_auc"]
    assert nv["inflation"] > 0.15  # recording-level leakage inflates a lot here


def test_multi_seed_stability_generated():
    meta = json.loads((MODEL_DIR / "metadata.json").read_text())
    st = meta["multi_seed_group_stability_dev"]
    assert set(st["seeds"]) == {21, 42, 84, 123, 2026}
    assert st["valid_seeds"] >= 3
    assert not np.isnan(st["mean"])
    assert st["min"] <= st["mean"] <= st["max"]


def test_model_selection_not_on_holdout():
    meta = json.loads((MODEL_DIR / "metadata.json").read_text())
    assert meta["model_selection"]["rule"].lower().startswith("max dev")
    assert "dev" in meta["split_strategy"].lower()


def test_holdout_auc_matches_metadata_within_tolerance():
    from sklearn.metrics import roc_auc_score
    meta = json.loads((MODEL_DIR / "metadata.json").read_text())
    active = meta["active_features"]
    _, X, y, g = _load()
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    _, hold = next(gss.split(X, y, g))
    proba = registry._model("parkinsons").predict_proba(X.iloc[hold][active])[:, 1]
    recorded = meta["holdout_metrics"]["roc_auc"]
    assert abs(roc_auc_score(y.iloc[hold], proba) - recorded) < 1e-6


# --------------------------- API contract ---------------------------------
def test_api_valid_full_request_succeeds():
    r = client.post("/api/v1/predictions/parkinsons", json={"measurements": FULL})
    assert r.status_code == 200
    assert r.json()["prediction"] in (0, 1)


def test_api_valid_reduced_request_succeeds():
    r = client.post("/api/v1/predictions/parkinsons", json={"measurements": REDUCED})
    assert r.status_code == 200
    assert 0.0 <= r.json()["model_score"] <= 1.0


def test_api_missing_required_field_rejected():
    bad = {k: v for k, v in REDUCED.items() if k != "PPE"}
    r = client.post("/api/v1/predictions/parkinsons", json={"measurements": bad})
    assert r.status_code == 422
    assert "PPE" in r.json()["detail"]


def test_api_unknown_field_rejected():
    r = client.post("/api/v1/predictions/parkinsons",
                    json={"measurements": {**REDUCED, "MFCC1": 0.1}})
    assert r.status_code == 422
    assert "MFCC1" in r.json()["detail"]


def test_api_name_rejected():
    r = client.post("/api/v1/predictions/parkinsons",
                    json={"measurements": {**REDUCED, "name": "phon_R01_S01_1"}})
    assert r.status_code == 422


def test_api_status_rejected():
    r = client.post("/api/v1/predictions/parkinsons",
                    json={"measurements": {**REDUCED, "status": 1}})
    assert r.status_code == 422


def test_api_wrong_datatype_rejected():
    r = client.post("/api/v1/predictions/parkinsons",
                    json={"measurements": {**REDUCED, "HNR": "loud"}})
    assert r.status_code == 422


def test_api_impossible_range_rejected():
    r = client.post("/api/v1/predictions/parkinsons",
                    json={"measurements": {**REDUCED, "MDVP:Fo(Hz)": 5000}})
    assert r.status_code == 422


def test_api_metadata_endpoint():
    r = client.get("/api/v1/diseases/parkinsons")
    assert r.status_code == 200
    body = r.json()
    assert body["dataset"] == "parkinsons.csv"
    assert body["selected_model"]
    assert "holdout_metrics" in body


# --------------------------- integration ---------------------------------
def test_frontend_style_payload_round_trip():
    r = client.post("/api/v1/predictions/parkinsons", json={"measurements": REDUCED})
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["model_score"] <= 1.0
    assert body["threshold_result"] in ("at_or_above", "below")
    assert body["decision_threshold"] == 0.5
