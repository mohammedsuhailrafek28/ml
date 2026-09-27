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
from src.api.schemas import (
    DiabetesPredictionRequest, HeartPredictionRequest, KidneyPredictionRequest,
    LiverPredictionRequest, ParkinsonsPredictionRequest, PredictionRequest, PredictionResponse,
)
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


def _prediction(disease: str, req):
    try:
        return registry.predict(disease, req.measurements.model_dump(exclude_unset=True))
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


def _report(disease: str, req):
    try:
        measurements = req.measurements.model_dump(exclude_unset=True)
        result = registry.predict(disease, measurements)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    pdf = create_report(registry.metadata(disease)["name"], measurements, result)
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{disease}_report.pdf"',
            "Cache-Control": "no-store",
        },
    )


# Fixed disease paths preserve the public URLs while giving every operation its
# own measurement schema in generated OpenAPI documentation.
@app.post("/api/v1/predictions/liver", response_model=PredictionResponse)
def prediction_liver(req: LiverPredictionRequest, _: None = Depends(require_api_key)):
    return _prediction("liver", req)


@app.post("/api/v1/predictions/diabetes", response_model=PredictionResponse)
def prediction_diabetes(req: DiabetesPredictionRequest, _: None = Depends(require_api_key)):
    return _prediction("diabetes", req)


@app.post("/api/v1/predictions/heart", response_model=PredictionResponse)
def prediction_heart(req: HeartPredictionRequest, _: None = Depends(require_api_key)):
    return _prediction("heart", req)


@app.post("/api/v1/predictions/kidney", response_model=PredictionResponse)
def prediction_kidney(req: KidneyPredictionRequest, _: None = Depends(require_api_key)):
    return _prediction("kidney", req)


@app.post("/api/v1/predictions/parkinsons", response_model=PredictionResponse)
def prediction_parkinsons(req: ParkinsonsPredictionRequest, _: None = Depends(require_api_key)):
    return _prediction("parkinsons", req)


@app.post("/api/v1/reports/liver")
def report_liver(req: LiverPredictionRequest, _: None = Depends(require_api_key)):
    return _report("liver", req)


@app.post("/api/v1/reports/diabetes")
def report_diabetes(req: DiabetesPredictionRequest, _: None = Depends(require_api_key)):
    return _report("diabetes", req)


@app.post("/api/v1/reports/heart")
def report_heart(req: HeartPredictionRequest, _: None = Depends(require_api_key)):
    return _report("heart", req)


@app.post("/api/v1/reports/kidney")
def report_kidney(req: KidneyPredictionRequest, _: None = Depends(require_api_key)):
    return _report("kidney", req)


@app.post("/api/v1/reports/parkinsons")
def report_parkinsons(req: ParkinsonsPredictionRequest, _: None = Depends(require_api_key)):
    return _report("parkinsons", req)


# Keep the historical unknown-slug error envelope. These fallback operations
# are deliberately omitted from OpenAPI so published schemas stay disease-specific.
@app.post("/api/v1/predictions/{disease}", include_in_schema=False)
def prediction_unknown(disease: str, req: PredictionRequest, _: None = Depends(require_api_key)):
    _require_known(disease)
    model = {
        "liver": LiverPredictionRequest, "diabetes": DiabetesPredictionRequest,
        "heart": HeartPredictionRequest, "kidney": KidneyPredictionRequest,
        "parkinsons": ParkinsonsPredictionRequest,
    }[disease]
    return _prediction(disease, model.model_validate({"measurements": req.measurements}))


@app.post("/api/v1/reports/{disease}", include_in_schema=False)
def report_unknown(disease: str, req: PredictionRequest, _: None = Depends(require_api_key)):
    _require_known(disease)
    model = {
        "liver": LiverPredictionRequest, "diabetes": DiabetesPredictionRequest,
        "heart": HeartPredictionRequest, "kidney": KidneyPredictionRequest,
        "parkinsons": ParkinsonsPredictionRequest,
    }[disease]
    return _report(disease, model.model_validate({"measurements": req.measurements}))
