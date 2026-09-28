from __future__ import annotations

import hashlib
import json
from pathlib import Path

from fastapi.testclient import TestClient

from src.application_release import public_release_provenance, safe_source_revision
from src.api.main import app
from src.api.settings import _load
from scripts.build_application_release_manifest import OUTPUT, render


def test_release_identity_and_public_provenance_are_safe(monkeypatch):
    monkeypatch.setenv("APP_SOURCE_REVISION", "abcdef0123456789")
    public = public_release_provenance()
    assert public["version"] == "0.9.0-rc.1"
    assert public["release_status"] == "educational/research release candidate"
    assert public["intended_use"] == "Educational and research use only."
    assert public["clinical_status"] == "not clinically validated"
    assert public["source_revision"] == "abcdef0123456789"
    assert public["runtime_versions"] == {"backend_python": "3.11.15", "frontend_node": "20.19.0"}
    assert len(public["model_release"]["manifest_sha256"]) == 64
    assert "runtime_versions" in public
    assert "internal" not in json.dumps(public).lower()
    assert "api_key" not in json.dumps(public).lower()


def test_untrusted_revision_is_never_reflected(monkeypatch):
    monkeypatch.setenv("APP_SOURCE_REVISION", "http://internal.example/path?key=secret")
    assert safe_source_revision() == "unknown"


def test_health_exposes_application_and_model_release_identity():
    with TestClient(app) as client:
        response = client.get("/api/v1/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["version"] == "0.9.0-rc.1"
    assert payload["model_release"]["identifier"]
    assert len(payload["model_release"]["manifest_sha256"]) == 64


def test_model_manifest_hash_matches_health_provenance():
    from src.application_release import MODEL_RELEASE_PATH

    expected = hashlib.sha256(MODEL_RELEASE_PATH.read_bytes()).hexdigest()
    assert public_release_provenance()["model_release"]["manifest_sha256"] == expected


def test_production_reads_service_key_from_runtime_file(monkeypatch, tmp_path: Path):
    secret_path = tmp_path / "mounted-key"
    synthetic_key = "S" * 40
    secret_path.write_text(synthetic_key, encoding="utf-8")
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("API_KEY_FILE", str(secret_path))
    monkeypatch.delenv("API_KEY", raising=False)
    configured = _load()
    assert configured.auth_enabled
    assert configured.api_key == synthetic_key


def test_application_release_manifest_is_fresh():
    assert OUTPUT.exists()
    assert OUTPUT.read_text(encoding="utf-8") == render()
