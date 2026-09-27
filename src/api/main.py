"""Medical AI Suite API.

Educational, non-diagnostic. The persisted Joblib pipelines are the only
inference layer. Runtime configuration comes from the environment via
``src.api.settings`` (see ``.env.example``).
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from src.api.health import API_VERSION, liveness, readiness, startup_integrity_log
from src.api.middleware import (
    BodySizeLimitMiddleware,
    RateLimitMiddleware,
    RequestContextMiddleware,
    error_response,
    require_api_key,
)
from src.api.schemas import PredictionRequest, PredictionResponse
from src.api.services.model_registry import registry
from src.api.settings import get_settings
from src.reporting.pdf_report import create_report
from src.utils.config import DISEASES

settings = get_settings()
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("medical_ai.api")

@asynccontextmanager
async def _lifespan(_: FastAPI):
    startup_integrity_log(logger)
    logger.info(
        '{"event":"auth","status":"%s"}'
        % ("enabled" if settings.auth_enabled else "demo_mode")
    )
    yield


app = FastAPI(
    title="Medical AI Suite API",
    version=API_VERSION,
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
    )
elif settings.is_production:
    logger.warning(
        '{"event":"cors","status":"disabled","reason":"ALLOWED_ORIGINS not set in production"}'
    )


def _rid(request: Request) -> str:
    return getattr(request.state, "request_id", "unknown")


@app.exception_handler(RequestValidationError)
async def _validation_handler(request: Request, exc: RequestValidationError):
    return error_response(422, "validation_error", "Invalid request body", _rid(request))


@app.exception_handler(HTTPException)
async def _http_handler(request: Request, exc: HTTPException):
    codes = {400: "bad_request", 401: "unauthorized", 404: "not_found",
             413: "request_too_large", 422: "validation_error", 429: "rate_limited"}
    return error_response(
        exc.status_code, codes.get(exc.status_code, "error"),
        str(exc.detail), _rid(request),
    )


@app.exception_handler(Exception)
async def _unhandled_handler(request: Request, exc: Exception):
    logger.exception("unhandled error request_id=%s", _rid(request))
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


@app.post("/api/v1/predictions/{disease}", response_model=PredictionResponse)
def prediction(disease: str, req: PredictionRequest, _: None = Depends(require_api_key)):
    """Return the typed, non-diagnostic communication contract for one model."""
    _require_known(disease)
    try:
        return registry.predict(disease, req.measurements)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/api/v1/reports/{disease}")
def report(disease: str, req: PredictionRequest, _: None = Depends(require_api_key)):
    _require_known(disease)
    try:
        result = registry.predict(disease, req.measurements)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    pdf = create_report(registry.metadata(disease)["name"], req.measurements, result)
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{disease}_report.pdf"',
            "Cache-Control": "no-store",
        },
    )
