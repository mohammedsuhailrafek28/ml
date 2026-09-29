"""Privacy, request tracing, and bounded metrics regression coverage."""
from __future__ import annotations

import importlib
import json
import logging
from pathlib import Path
import re

from fastapi.testclient import TestClient
import pytest

from src.api import settings as settings_module
from src.api.services.model_registry import Registry
from src.model_release import load_release_manifest

ROOT = Path(__file__).resolve().parents[1]
SECRET = "phase7-service-key-never-log-987654321"
INTERNAL_URL = "http://private-backend-marker.invalid:8123"
REQUEST_ID = "a3a2e986-324d-4e51-a99d-d0767e86b68f"
LIVER = {
    "Age": 47, "Gender": "Male", "TB": 2.718281828, "DB": 0.3,
    "Alkphos": 190, "Sgpt": 25, "Sgot": 30, "TP": 6.8,
    "ALB": 3.3, "A/G Ratio": 0.9,
}


def _production(monkeypatch, **values):
    variables = {
        "APP_ENV": "production", "API_KEY": SECRET, "ALLOWED_ORIGINS": "",
        "RATE_LIMIT_PER_MINUTE": "120", "MODEL_ROOT": None,
    }
    variables.update(values)
    for key, value in variables.items():
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, str(value))
    settings_module.reload_settings()
    import src.api.main as main

    return importlib.reload(main)


@pytest.fixture(autouse=True)
def _restore_cached_settings(monkeypatch):
    yield
    # The settings cache and app-level CORS setup outlive pytest's env patch.
    # Restore env first, then rebuild settings/app so later modules see isolation.
    monkeypatch.undo()
    settings_module.reload_settings()
    import src.api.main as main
    importlib.reload(main)


def test_production_logs_are_structured_and_redacted(monkeypatch, caplog):
    main = _production(monkeypatch)
    caplog.set_level(logging.INFO, logger="medical_ai.api")
    headers = {"X-API-Key": SECRET, "Authorization": f"Bearer {SECRET}", "X-Request-ID": REQUEST_ID}
    with TestClient(main.app) as client:
        prediction = client.post("/api/v1/predictions/liver", json={"measurements": LIVER}, headers=headers)
        assert prediction.status_code == 200
        assert prediction.headers["X-Request-ID"] == REQUEST_ID
        assert prediction.json()["model_identifier"] in prediction.text

        pdf = client.post("/api/v1/reports/liver", json={"measurements": LIVER}, headers=headers)
        assert pdf.status_code == 200 and pdf.content.startswith(b"%PDF")

        monkeypatch.setattr(main.registry, "predict", lambda *_args: (_ for _ in ()).throw(
            RuntimeError("PHASE7_EXCEPTION_DETAILS_MUST_NOT_ESCAPE")
        ))
        failed = client.post("/api/v1/predictions/liver", json={"measurements": LIVER}, headers=headers)
        assert failed.status_code == 500
        assert "Internal server error" in failed.text
        assert "PHASE7_EXCEPTION_DETAILS_MUST_NOT_ESCAPE" not in failed.text

    logs = "\n".join(record.getMessage() for record in caplog.records)
    release = load_release_manifest()["releases"]["liver"]
    forbidden = [
        SECRET, "Bearer", INTERNAL_URL, str(LIVER["TB"]), "PHASE7_EXCEPTION_DETAILS_MUST_NOT_ESCAPE",
        str(ROOT), f"{prediction.json()['model_score']:.4f}", "%PDF", "Input measurements:",
    ]
    assert all(marker not in logs for marker in forbidden)
    records = [json.loads(line) for line in logs.splitlines() if line.startswith("{")]
    request_log = next(record for record in records if record.get("event") == "http_request"
                       and record.get("request_id") == REQUEST_ID)
    assert request_log["http_method"] == "POST"
    assert request_log["route"] == "/api/v1/predictions/{disease}"
    assert request_log["disease"] == "liver"
    assert request_log["model_identifier"] == release["model"]["identifier"]
    assert request_log["release_status"] == release["release_status"]
    assert request_log["app_version"]
    assert request_log["environment"] == "production"
    error = next(record for record in records if record.get("event") == "request_exception")
    assert error["error_category"] == "unhandled_exception"


def test_metrics_endpoint_is_protected_and_operational_metrics_increment(monkeypatch):
    main = _production(monkeypatch)
    headers = {"X-API-Key": SECRET, "X-Request-ID": REQUEST_ID}
    with TestClient(main.app) as client:
        assert client.get("/internal/metrics").status_code == 401
        before = client.get("/internal/metrics", headers={"X-API-Key": SECRET})
        assert before.status_code == 200
        assert "/internal/metrics" not in client.get("/openapi.json").json()["paths"]

        response = client.post("/api/v1/predictions/liver", json={"measurements": LIVER}, headers=headers)
        assert response.status_code == 200
        report = client.post("/api/v1/reports/liver", json={"measurements": LIVER}, headers=headers)
        assert report.status_code == 200 and report.content.startswith(b"%PDF")
        assert client.get("/api/v1/ready").status_code == 200
        metrics = client.get("/internal/metrics", headers={"X-API-Key": SECRET})

    text = metrics.text
    expected = (
        'medical_ai_http_requests_total{method="POST",route="/api/v1/predictions/{disease}",status_class="2xx"}',
        'medical_ai_predictions_total{disease="liver",outcome="success"}',
        'medical_ai_reports_total{disease="liver",outcome="success"}',
        'medical_ai_threshold_results_total{disease="liver",result="at_or_above"}',
        'medical_ai_model_loads_total{disease="liver",result="success"}',
        'medical_ai_auth_rejections_total',
        'medical_ai_rate_limit_rejections_total',
        'medical_ai_http_request_duration_seconds_count{method="POST",route="/api/v1/predictions/{disease}"}',
        'medical_ai_prediction_duration_seconds_count{disease="liver"}',
        'medical_ai_report_duration_seconds_count{disease="liver"}',
        'medical_ai_readiness{state="ready"} 1.0',
        'medical_ai_in_flight_requests 0.0',
    )
    assert all(marker in text for marker in expected)
    assert re.search(r'medical_ai_auth_rejections_total\s+[1-9]', text)
    assert SECRET not in text and INTERNAL_URL not in text and str(LIVER["TB"]) not in text
    assert "user_agent" not in text.lower() and "client_ip" not in text.lower()


def test_auth_and_rate_rejections_are_counted_with_bounded_labels(monkeypatch):
    main = _production(monkeypatch, RATE_LIMIT_PER_MINUTE="2")
    with TestClient(main.app) as client:
        assert client.post("/api/v1/predictions/liver", json={"measurements": LIVER},
                           headers={"X-API-Key": SECRET}).status_code == 200
        assert client.post("/api/v1/predictions/liver", json={"measurements": LIVER}).status_code == 401
        limited = client.post("/api/v1/predictions/liver", json={"measurements": LIVER},
                              headers={"X-API-Key": SECRET})
        assert limited.status_code == 429
        scrape = client.get("/internal/metrics", headers={"X-API-Key": SECRET})

    assert re.search(r"medical_ai_rate_limit_rejections_total\s+[1-9]", scrape.text)
    assert re.search(r'medical_ai_http_errors_total\{[^}]*error_category="rate_limit"[^}]*\}', scrape.text)
    assert "PHASE7" not in scrape.text
    # Routes and label domains come from fixed allowlists, not user path content.
    routes = set(re.findall(r'route="([^"]+)"', scrape.text))
    assert routes <= {
        "/api/v1/health", "/api/v1/ready", "/api/v1/diseases",
        "/api/v1/diseases/{disease}", "/api/v1/predictions/{disease}",
        "/api/v1/reports/{disease}", "/internal/metrics", "/openapi.json", "/docs", "/redoc",
    }


def test_model_load_success_and_failure_never_log_paths(monkeypatch, tmp_path, caplog):
    main = _production(monkeypatch, MODEL_ROOT=str(tmp_path))
    from src.api.observability import METRICS_REGISTRY

    caplog.set_level(logging.INFO, logger="medical_ai.api")
    registry = Registry()
    with pytest.raises(FileNotFoundError):
        registry._model("liver")
    logs = caplog.text
    assert str(tmp_path) not in logs
    assert "model_load" in logs and "failure" in logs
    with TestClient(main.app) as client:
        scrape = client.get("/internal/metrics", headers={"X-API-Key": SECRET})
    assert 'medical_ai_model_loads_total{disease="liver",result="failure"} 1.0' in scrape.text
