"""Contract tests for the authoritative persisted-model release."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from joblib import load

from src.api.services.model_registry import Registry
from src.model_release import (
    load_release_manifest,
    release_file_issues,
    runtime_compatibility_issues,
)
from src.utils.config import DISEASES

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = json.loads((ROOT / "tests/fixtures/model_release_golden.json").read_text(encoding="utf-8"))
MANIFEST = load_release_manifest()


def test_manifest_is_current_and_complete():
    assert MANIFEST["release_schema_version"] == "1.0.0"
    assert set(MANIFEST["releases"]) == set(DISEASES)
    completed = subprocess.run(
        [sys.executable, "-m", "scripts.build_model_release_manifest", "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_release_hashes_and_runtime_are_compatible():
    assert release_file_issues(include_datasets=True) == []
    assert runtime_compatibility_issues() == []


@pytest.mark.parametrize("disease", sorted(DISEASES))
def test_released_pipeline_identity_and_contract(disease):
    release = MANIFEST["releases"][disease]
    metadata = json.loads((ROOT / release["metadata"]["path"]).read_text(encoding="utf-8"))
    feature_names = json.loads((ROOT / release["feature_list"]["path"]).read_text(encoding="utf-8"))
    pipeline = load(ROOT / release["model"]["path"])

    assert metadata["disease"] == release["metadata_disease_label"] == DISEASES[disease].title
    assert Path(metadata["artifact_path"]).parent.name == disease
    assert metadata["artifact_path"] == release["model"]["path"]
    assert metadata["selected_model"] in release["model"]["identifier"]
    assert type(pipeline.steps[-1][1]).__module__ + "." + type(pipeline.steps[-1][1]).__name__ == release["estimator_type"]
    assert feature_names == release["feature_list"]["transformed_features"]
    assert len(release["ordered_active_features"]) == release["feature_count"]


def test_release_statuses_reflect_evidence():
    for disease in ("liver", "diabetes", "heart", "kidney"):
        assert MANIFEST["releases"][disease]["release_status"] == "educational/research release"
    parkinsons = MANIFEST["releases"]["parkinsons"]
    assert parkinsons["release_status"] == "experimental"
    holdout = parkinsons["verified_metrics"]["subject_disjoint_holdout"]
    assert holdout["holdout_subjects"] == 7
    assert holdout["specificity"] == 0.0


@pytest.mark.parametrize("vector", GOLDEN["vectors"], ids=lambda vector: vector["disease_identifier"])
def test_golden_prediction_vector_through_production_registry(vector):
    registry = Registry()
    disease = vector["disease_identifier"]
    release = MANIFEST["releases"][disease]
    result = registry.predict(disease, vector["measurements"])

    assert registry.active_features(disease) == vector["normalized_feature_order"]
    assert registry.active_features(disease) == release["ordered_active_features"]
    assert release["api_request_schema"]["ordered_measurement_fields"] == vector["normalized_feature_order"]
    assert release["disease_identifier"] == result["disease"] == disease
    assert release["model"]["identifier"] == vector["model_identifier"]
    assert result["prediction"] == vector["expected_prediction"]
    assert result["threshold"] == vector["threshold"]
    assert result["probability"] == pytest.approx(
        vector["expected_probability"], abs=GOLDEN["score_tolerance"], rel=0
    )
