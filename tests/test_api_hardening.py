"""Production-hardening tests: settings, readiness, error envelope, request id,
body-size limit, optional API key, CORS. No disease ML behaviour is touched.
"""
import importlib
import os
import warnings

import pytest
from fastapi.testclient import TestClient

from src.api import settings as settings_mod

warnings.filterwarnings("ignore")

LIVER = {
    "Age": 45, "Gender": "Male", "TB": 1.0, "DB": 0.3, "Alkphos": 190,
    "Sgpt": 25, "Sgot": 30, "TP": 6.8, "ALB": 3.3, "A/G Ratio": 0.9,
}


@pytest.fixture
def env(monkeypatch):
    """Set env vars, refresh cached settings, restore afterwards."""
    def _apply(**kw):
        for k, v in kw.items():
            if v is None:
                monkeypatch.delenv(k, raising=False)
            else:
                monkeypatch.setenv(k, str(v))
        return settings_mod.reload_settings()
    yield _apply
    settings_mod.reload_settings()


def _client():
    # rebuild the app so import-time settings (CORS) reflect current env
    import src.api.main as main_mod
    importlib.reload(main_mod)
    return TestClient(main_mod.app)


# --------------------------- settings --------------------------------------
def test_defaults_are_dev_friendly(env):
    s = env(APP_ENV=None, ALLOWED_ORIGINS=None, API_KEY=None)
    assert s.app_env == "development"
    assert s.allowed_origins == ("http://localhost:3000",)
    assert s.auth_enabled is False
    assert s.debug_errors is True


def test_production_requires_explicit_origins(env):
    s = env(APP_ENV="production", ALLOWED_ORIGINS=None)
    assert s.is_production is True
    assert s.allowed_origins == ()
    assert s.debug_errors is False


def test_origins_and_ints_parse(env):
    s = env(ALLOWED_ORIGINS="https://a.example, https://b.example",
            MAX_REQUEST_BYTES="1234", RATE_LIMIT_PER_MINUTE="not-a-number")
    assert s.allowed_origins == ("https://a.example", "https://b.example")
    assert s.max_request_bytes == 1234
    assert s.rate_limit_per_minute == 120  # bad int -> default


def test_api_key_enables_auth(env):
    assert env(API_KEY="secret").auth_enabled is True
    assert env(API_KEY="").auth_enabled is False


# --------------------------- readiness / integrity ------------------------
def test_ready_endpoint_ok_with_real_models(env):
    env(MODEL_ROOT=None)
    r = _client().get("/api/v1/ready")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ready"
    assert set(body["diseases"]) == {"liver", "heart", "diabetes", "kidney", "parkinsons"}
    assert all(d["ok"] for d in body["diseases"].values())


def test_ready_degrades_when_artifacts_missing(env, tmp_path):
    env(MODEL_ROOT=str(tmp_path))
    r = _client().get("/api/v1/ready")
    assert r.status_code == 503
    body = r.json()
    assert body["status"] == "degraded"
    assert any(not d["ok"] and d["issues"] for d in body["diseases"].values())


def test_health_is_cheap_and_lists_loaded(env):
    env(MODEL_ROOT=None)
    r = _client().get("/api/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "alive"
    assert "loaded_diseases" in body


# --------------------------- error envelope / request id -----------------
def test_error_envelope_on_unknown_disease(env):
    env(API_KEY=None)
    r = _client().post("/api/v1/predictions/pancreas", json={"measurements": {"x": 1}})
    assert r.status_code == 404
    body = r.json()
    assert body["error"]["code"] == "not_found"
    assert body["error"]["requestId"]
    assert body["detail"]  # compatibility alias retained


def test_validation_error_uses_envelope(env):
    env(API_KEY=None)
    r = _client().post("/api/v1/predictions/liver",
                       json={"measurements": {**LIVER, "Age": 999}})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "validation_error"


def test_request_id_header_present(env):
    env(MODEL_ROOT=None)
    r = _client().get("/api/v1/health")
    assert r.headers.get("X-Request-ID")
    assert r.headers.get("X-Content-Type-Options") == "nosniff"


def test_supplied_request_id_is_echoed(env):
    env(MODEL_ROOT=None)
    r = _client().get("/api/v1/health", headers={"X-Request-ID": "abc123"})
    assert r.headers.get("X-Request-ID") == "abc123"


# --------------------------- body size / rate limit ---------------------
def test_oversized_body_rejected(env):
    env(MAX_REQUEST_BYTES="200", API_KEY=None)
    big = {"measurements": {f"k{i}": i for i in range(500)}}
    r = _client().post("/api/v1/predictions/liver", json=big)
    assert r.status_code == 413
    assert r.json()["error"]["code"] == "request_too_large"


# --------------------------- optional API key --------------------------
def test_api_key_required_when_configured(env):
    env(API_KEY="s3cret", MODEL_ROOT=None)
    c = _client()
    assert c.post("/api/v1/predictions/liver", json={"measurements": LIVER}).status_code == 401
    ok = c.post("/api/v1/predictions/liver", json={"measurements": LIVER},
                headers={"X-API-Key": "s3cret"})
    assert ok.status_code == 200
    # health stays open
    assert c.get("/api/v1/health").status_code == 200


def test_demo_mode_needs_no_key(env):
    env(API_KEY=None, MODEL_ROOT=None)
    r = _client().post("/api/v1/predictions/liver", json={"measurements": LIVER})
    assert r.status_code == 200
    assert r.json()["prediction"] in (0, 1)


# --------------------------- CORS -------------------------------------
def test_cors_allows_configured_origin(env):
    env(ALLOWED_ORIGINS="http://localhost:3000", MODEL_ROOT=None)
    r = _client().get("/api/v1/health", headers={"Origin": "http://localhost:3000"})
    assert r.headers.get("access-control-allow-origin") == "http://localhost:3000"


# --------------------------- no ML regression -------------------------
def test_prediction_communication_contract(env):
    env(API_KEY=None, MODEL_ROOT=None)
    r = _client().post("/api/v1/predictions/liver", json={"measurements": LIVER})
    b = r.json()
    for key in ("prediction", "model_score", "decision_threshold", "threshold_result",
                "score_type", "model_identifier", "release_status", "intended_use",
                "limitations", "disclaimer"):
        assert key in b
    assert not {"probability", "threshold", "label", "selectedModel"} & set(b)
    assert "not a probability" in b["disclaimer"]


def test_prediction_openapi_response_is_typed(env):
    env(API_KEY=None, MODEL_ROOT=None)
    schema = _client().get("/openapi.json").json()
    response = schema["paths"]["/api/v1/predictions/{disease}"]["post"]["responses"]["200"]
    assert response["content"]["application/json"]["schema"]["$ref"].endswith(
        "/PredictionResponse"
    )
    properties = schema["components"]["schemas"]["PredictionResponse"]["properties"]
    assert set(properties) == {
        "disease", "prediction", "model_score", "decision_threshold",
        "threshold_result", "score_type", "model_identifier", "release_status",
        "intended_use", "limitations", "disclaimer",
    }
