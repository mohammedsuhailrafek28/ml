"""Privacy-safe application logs and bounded Prometheus metrics."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

from src.api.settings import get_settings
from src.utils.config import DISEASES

API_VERSION = "1.2"
METRICS_REGISTRY = CollectorRegistry()

HTTP_REQUESTS = Counter(
    "medical_ai_http_requests_total", "HTTP requests by bounded route and status class.",
    ("method", "route", "status_class"), registry=METRICS_REGISTRY,
)
HTTP_ERRORS = Counter(
    "medical_ai_http_errors_total", "HTTP errors by bounded route and category.",
    ("method", "route", "status_class", "error_category"), registry=METRICS_REGISTRY,
)
HTTP_DURATION = Histogram(
    "medical_ai_http_request_duration_seconds", "HTTP request duration.",
    ("method", "route"), registry=METRICS_REGISTRY,
)
PREDICTIONS = Counter(
    "medical_ai_predictions_total", "Prediction operations by disease and outcome.",
    ("disease", "outcome"), registry=METRICS_REGISTRY,
)
PREDICTION_DURATION = Histogram(
    "medical_ai_prediction_duration_seconds", "Prediction operation duration.",
    ("disease",), registry=METRICS_REGISTRY,
)
REPORTS = Counter(
    "medical_ai_reports_total", "Report operations by disease and outcome.",
    ("disease", "outcome"), registry=METRICS_REGISTRY,
)
REPORT_DURATION = Histogram(
    "medical_ai_report_duration_seconds", "Report operation duration.",
    ("disease",), registry=METRICS_REGISTRY,
)
THRESHOLD_RESULTS = Counter(
    "medical_ai_threshold_results_total", "Threshold classes by disease.",
    ("disease", "result"), registry=METRICS_REGISTRY,
)
MODEL_LOADS = Counter(
    "medical_ai_model_loads_total", "Persisted model loads by disease and result.",
    ("disease", "result"), registry=METRICS_REGISTRY,
)
READINESS = Gauge(
    "medical_ai_readiness", "Current readiness state (1=ready, 0=degraded).",
    ("state",), registry=METRICS_REGISTRY,
)
AUTH_REJECTIONS = Counter(
    "medical_ai_auth_rejections_total", "Rejected service-authentication requests.",
    registry=METRICS_REGISTRY,
)
RATE_LIMIT_REJECTIONS = Counter(
    "medical_ai_rate_limit_rejections_total", "Rejected requests due to rate limiting.",
    registry=METRICS_REGISTRY,
)
IN_FLIGHT = Gauge(
    "medical_ai_in_flight_requests", "Requests currently being processed.",
    registry=METRICS_REGISTRY,
)

_METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"}
_ROUTES = {
    "/api/v1/health", "/api/v1/ready", "/api/v1/diseases",
    "/api/v1/diseases/{disease}", "/api/v1/predictions/{disease}",
    "/api/v1/reports/{disease}", "/internal/metrics", "/openapi.json",
    "/docs", "/redoc",
}


def normalized_route(path: str) -> tuple[str, str | None]:
    """Return a fixed route label and a known disease, never a raw path."""
    parts = [part for part in path.split("/") if part]
    if path in _ROUTES:
        return path, None
    if len(parts) == 4 and parts[:2] == ["api", "v1"]:
        resource, disease = parts[2], parts[3]
        if resource in {"diseases", "predictions", "reports"}:
            label = f"/api/v1/{resource}/{{disease}}"
            return label, disease if disease in DISEASES else None
    return "unmatched", None


def bounded_method(method: str) -> str:
    return method if method in _METHODS else "OTHER"


def status_class(status_code: int) -> str:
    return f"{status_code // 100}xx" if 100 <= status_code <= 599 else "other"


def error_category(status_code: int) -> str:
    if status_code == 401:
        return "authentication"
    if status_code == 429:
        return "rate_limit"
    if status_code == 422:
        return "validation"
    if status_code == 413:
        return "request_size"
    if status_code >= 500:
        return "server"
    if status_code >= 400:
        return "client"
    return "none"


def record_http(method: str, route: str, status_code: int, duration_seconds: float) -> None:
    method = bounded_method(method)
    route = route if route in _ROUTES or route == "unmatched" else "unmatched"
    klass = status_class(status_code)
    HTTP_REQUESTS.labels(method, route, klass).inc()
    HTTP_DURATION.labels(method, route).observe(max(0.0, duration_seconds))
    if status_code >= 400:
        HTTP_ERRORS.labels(method, route, klass, error_category(status_code)).inc()


def observe_prediction(disease: str, duration_seconds: float, outcome: str) -> None:
    if disease not in DISEASES:
        return
    outcome = outcome if outcome in {"success", "client_error", "server_error"} else "server_error"
    PREDICTIONS.labels(disease, outcome).inc()
    PREDICTION_DURATION.labels(disease).observe(max(0.0, duration_seconds))


def observe_report(disease: str, duration_seconds: float, outcome: str) -> None:
    if disease not in DISEASES:
        return
    outcome = outcome if outcome in {"success", "client_error", "server_error"} else "server_error"
    REPORTS.labels(disease, outcome).inc()
    REPORT_DURATION.labels(disease).observe(max(0.0, duration_seconds))


def observe_model_load(disease: str, success: bool) -> None:
    if disease in DISEASES:
        MODEL_LOADS.labels(disease, "success" if success else "failure").inc()


def set_readiness(ready: bool) -> None:
    READINESS.labels("ready").set(1 if ready else 0)
    READINESS.labels("degraded").set(0 if ready else 1)


def emit_log(level: int, event: str, **fields) -> None:
    """Emit structured production records; fields are caller allowlisted."""
    settings = get_settings()
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "level": logging.getLevelName(level),
        "event": event,
        "app_version": API_VERSION,
        "environment": settings.app_env,
        **{key: value for key, value in fields.items() if value is not None},
    }
    logger = logging.getLogger("medical_ai.api")
    if settings.is_production:
        logger.log(level, json.dumps(record, separators=(",", ":"), sort_keys=True))
    else:
        readable = " ".join(f"{key}={value}" for key, value in record.items() if key not in {"timestamp", "level", "app_version", "environment"})
        logger.log(level, readable)
