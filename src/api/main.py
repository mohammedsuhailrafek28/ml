"""Medical AI Suite API.

Educational, non-diagnostic. The persisted Joblib pipelines are the only
inference layer. Runtime configuration comes from the environment via
``src.api.settings`` (see ``.env.example``).
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src.api.health import liveness, readiness, startup_integrity_log
from src.application_release import application_release
from src.api.middleware import (
    BodySizeLimitMiddleware,
    RateLimitMiddleware,
    RequestContextMiddleware,
    error_response,
    require_api_key,
)
from src.api.observability import (
    THRESHOLD_RESULTS, METRICS_REGISTRY, emit_log, observe_prediction, observe_report,
    set_readiness,
)
from src.api.schemas import (
    DiabetesPredictionRequest, HeartPredictionRequest, KidneyPredictionRequest,
    LiverPredictionRequest, ParkinsonsPredictionRequest, PredictionRequest, PredictionResponse,
)
from src.api.services.model_registry import registry
from src.api.settings import get_settings
from src.reporting.pdf_report import create_report
from src.utils.config import DISEASES

settings = get_settings()
logging.basicConfig(level=settings.log_level)
logger = logging.getLogger("medical_ai.api")

@asynccontextmanager
async def _lifespan(_: FastAPI):
    startup_integrity_log(logger)
    emit_log(logging.INFO, "service_authentication", status="enabled" if settings.auth_enabled else "demo_mode")
    yield


app = FastAPI(
    title="Medical AI Suite API",
    version=application_release()["version"],
    description=(
        "Educational persisted-model inference. Returned model_score values are "
        "uncalibrated scores, not disease probabilities, diagnoses, or screening results."
    ),
    lifespan=_lifespan,
)

# Middleware runs bottom-up on the request; RequestContext must be outermost so
# every response (including limiter/auth rejections) carries a request id.
app.add_middleware(RateLimitMiddleware)
app.add_middleware(BodySizeLimitMiddleware)
app.add_middleware(RequestContextMiddleware)

if settings.allowed_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.allowed_origins),
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "X-API-Key", "X-Request-ID"],
        expose_headers=["X-Request-ID"],
    )
elif settings.is_production:
    emit_log(logging.WARNING, "cors_configuration", status="disabled", error_category="origins_not_configured")


def _rid(request: Request) -> str:
    return getattr(request.state, "request_id", "unknown")


@app.exception_handler(RequestValidationError)
async def _validation_handler(request: Request, exc: RequestValidationError):
    # Expose only schema locations, never submitted values or Pydantic's raw
    # error context. This gives callers useful field-level guidance safely.
    fields = sorted({str(location[-1]) for error in exc.errors()
                     if (location := error.get("loc")) and isinstance(location[-1], str)
                     and location[-1] not in {"body", "measurements"}})
    message = "Invalid request body" + (f"; check fields: {', '.join(fields)}" if fields else "")
    return error_response(422, "validation_error", message, _rid(request))


@app.exception_handler(HTTPException)
async def _http_handler(request: Request, exc: HTTPException):
    codes = {400: "bad_request", 401: "unauthorized", 404: "not_found",
             413: "request_too_large", 422: "validation_error", 429: "rate_limited"}
    message = str(exc.detail)
    if settings.is_production:
        message = {
            400: "Invalid request", 401: "Authentication required", 404: "Not found",
            413: "Request too large", 422: "Invalid request body",
            429: "Rate limit exceeded",
        }.get(exc.status_code, "Request failed")
    return error_response(
        exc.status_code, codes.get(exc.status_code, "error"),
        message, _rid(request),
    )


@app.exception_handler(Exception)
async def _unhandled_handler(request: Request, exc: Exception):
    emit_log(logging.ERROR, "request_exception", request_id=_rid(request),
             error_category="unhandled_exception")
    if not settings.is_production:
        logger.error("Unhandled %s for request_id=%s", type(exc).__name__, _rid(request))
    msg = "Internal server error" if settings.is_production else f"{type(exc).__name__}: {exc}"
    return error_response(500, "internal_error", msg, _rid(request))


def _require_known(disease: str) -> None:
    if disease not in DISEASES:
        raise HTTPException(status_code=404, detail=f"Unknown disease: {disease}")


@app.get("/api/v1/health")
def health():
    body = liveness()
    body["loaded_diseases"] = registry.available()  # back-compat field
    return body


@app.get("/api/v1/ready")
def ready(request: Request):
    from fastapi.responses import JSONResponse

    payload, status = readiness()
    set_readiness(status == 200)
    if status == 200:
        return payload
    return JSONResponse(status_code=503, content=payload,
                        headers={"X-Request-ID": _rid(request)})


@app.get("/api/v1/diseases")
def diseases(_: None = Depends(require_api_key)):
    return registry.catalog()


@app.get("/api/v1/diseases/{disease}")
def disease(disease: str, _: None = Depends(require_api_key)):
    _require_known(disease)
    return registry.metadata(disease)


@app.get("/internal/metrics", include_in_schema=False)
def internal_metrics(_: None = Depends(require_api_key)):
    return Response(generate_latest(METRICS_REGISTRY), media_type=CONTENT_TYPE_LATEST,
                    headers={"Cache-Control": "no-store"})


def _record_result(request: Request, result: dict) -> None:
    request.state.model_identifier = result["model_identifier"]
    request.state.release_status = result["release_status"]
    THRESHOLD_RESULTS.labels(result["disease"], result["threshold_result"]).inc()


def _prediction(disease: str, req, request: Request):
    started = time.perf_counter()
    try:
        result = registry.predict(disease, req.measurements.model_dump(exclude_unset=True))
        _record_result(request, result)
        observe_prediction(disease, time.perf_counter() - started, "success")
        return result
    except KeyError as e:
        observe_prediction(disease, time.perf_counter() - started, "client_error")
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        observe_prediction(disease, time.perf_counter() - started, "client_error")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception:
        observe_prediction(disease, time.perf_counter() - started, "server_error")
        raise


def _report(disease: str, req, request: Request):
    started = time.perf_counter()
    try:
        measurements = req.measurements.model_dump(exclude_unset=True)
        result = registry.predict(disease, measurements)
        _record_result(request, result)
    except KeyError as e:
        observe_report(disease, time.perf_counter() - started, "client_error")
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        observe_report(disease, time.perf_counter() - started, "client_error")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception:
        observe_report(disease, time.perf_counter() - started, "server_error")
        raise
    try:
        pdf = create_report(registry.metadata(disease)["name"], measurements, result)
        observe_report(disease, time.perf_counter() - started, "success")
        return Response(
            content=pdf,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{disease}_report.pdf"',
                "Cache-Control": "no-store",
            },
        )
    except Exception:
        observe_report(disease, time.perf_counter() - started, "server_error")
        raise


# Fixed disease paths preserve the public URLs while giving every operation its
# own measurement schema in generated OpenAPI documentation.
@app.post("/api/v1/predictions/liver", response_model=PredictionResponse)
def prediction_liver(request: Request, req: LiverPredictionRequest, _: None = Depends(require_api_key)):
    return _prediction("liver", req, request)


@app.post("/api/v1/predictions/diabetes", response_model=PredictionResponse)
def prediction_diabetes(request: Request, req: DiabetesPredictionRequest, _: None = Depends(require_api_key)):
    return _prediction("diabetes", req, request)


@app.post("/api/v1/predictions/heart", response_model=PredictionResponse)
def prediction_heart(request: Request, req: HeartPredictionRequest, _: None = Depends(require_api_key)):
    return _prediction("heart", req, request)


@app.post("/api/v1/predictions/kidney", response_model=PredictionResponse)
def prediction_kidney(request: Request, req: KidneyPredictionRequest, _: None = Depends(require_api_key)):
    return _prediction("kidney", req, request)


@app.post("/api/v1/predictions/parkinsons", response_model=PredictionResponse)
def prediction_parkinsons(request: Request, req: ParkinsonsPredictionRequest, _: None = Depends(require_api_key)):
    return _prediction("parkinsons", req, request)


@app.post("/api/v1/reports/liver")
def report_liver(request: Request, req: LiverPredictionRequest, _: None = Depends(require_api_key)):
    return _report("liver", req, request)


@app.post("/api/v1/reports/diabetes")
def report_diabetes(request: Request, req: DiabetesPredictionRequest, _: None = Depends(require_api_key)):
    return _report("diabetes", req, request)


@app.post("/api/v1/reports/heart")
def report_heart(request: Request, req: HeartPredictionRequest, _: None = Depends(require_api_key)):
    return _report("heart", req, request)


@app.post("/api/v1/reports/kidney")
def report_kidney(request: Request, req: KidneyPredictionRequest, _: None = Depends(require_api_key)):
    return _report("kidney", req, request)


@app.post("/api/v1/reports/parkinsons")
def report_parkinsons(request: Request, req: ParkinsonsPredictionRequest, _: None = Depends(require_api_key)):
    return _report("parkinsons", req, request)


# Keep the historical unknown-slug error envelope. These fallback operations
# are deliberately omitted from OpenAPI so published schemas stay disease-specific.
@app.post("/api/v1/predictions/{disease}", include_in_schema=False)
def prediction_unknown(disease: str, request: Request, req: PredictionRequest, _: None = Depends(require_api_key)):
    _require_known(disease)
    model = {
        "liver": LiverPredictionRequest, "diabetes": DiabetesPredictionRequest,
        "heart": HeartPredictionRequest, "kidney": KidneyPredictionRequest,
        "parkinsons": ParkinsonsPredictionRequest,
    }[disease]
    return _prediction(disease, model.model_validate({"measurements": req.measurements}), request)


@app.post("/api/v1/reports/{disease}", include_in_schema=False)
def report_unknown(disease: str, request: Request, req: PredictionRequest, _: None = Depends(require_api_key)):
    _require_known(disease)
    model = {
        "liver": LiverPredictionRequest, "diabetes": DiabetesPredictionRequest,
        "heart": HeartPredictionRequest, "kidney": KidneyPredictionRequest,
        "parkinsons": ParkinsonsPredictionRequest,
    }[disease]
    return _report(disease, model.model_validate({"measurements": req.measurements}), request)
