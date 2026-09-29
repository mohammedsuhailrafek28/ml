"""Lightweight production-baseline middleware: request IDs, structured logging,
body-size limit, a simple in-memory abuse guard, security headers, and a shared
error envelope. No new dependencies; nothing here touches model behaviour.

Logging deliberately records only safe metadata (endpoint, disease, status,
latency, request id) and NEVER the request body / measurement payload.
"""
from __future__ import annotations

import re
import hmac
import logging
import time
import uuid
from collections import defaultdict, deque

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from src.api.settings import get_settings
from src.api.observability import (
    AUTH_REJECTIONS, IN_FLIGHT, RATE_LIMIT_REJECTIONS, bounded_method,
    emit_log, error_category, normalized_route, record_http,
)

_REQUEST_ID = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-8][0-9a-fA-F]{3}-[89aAbB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}$")

_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store",
}


def error_response(status: int, code: str, message: str, request_id: str) -> JSONResponse:
    """Shared error envelope. `detail` is kept as a compatibility alias."""
    return JSONResponse(
        status_code=status,
        content={
            "error": {"code": code, "message": message, "requestId": request_id},
            "detail": message,
        },
        headers={"X-Request-ID": request_id, **_SECURITY_HEADERS},
    )


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assign a request id, time the request, add headers, log safe metadata."""

    async def dispatch(self, request: Request, call_next):
        supplied = request.headers.get("X-Request-ID", "").strip()
        rid = supplied.lower() if _REQUEST_ID.fullmatch(supplied) else str(uuid.uuid4())
        request.state.request_id = rid
        start = time.perf_counter()
        route, disease = normalized_route(request.url.path)
        method = bounded_method(request.method)
        count_in_flight = route != "/internal/metrics"
        if count_in_flight:
            IN_FLIGHT.inc()
        try:
            response = await call_next(request)
        except Exception:  # noqa: BLE001 - converted to a safe 500 below
            emit_log(logging.ERROR, "request_exception", request_id=rid,
                     error_category="unhandled_exception")
            if not get_settings().is_production:
                logging.getLogger("medical_ai.api").error("Unhandled request exception request_id=%s", rid)
            settings = get_settings()
            msg = "Internal server error" if settings.is_production else "Unhandled exception"
            response = error_response(500, "internal_error", msg, rid)
        try:
            duration = time.perf_counter() - start
            response.headers["X-Request-ID"] = rid
            for k, v in _SECURITY_HEADERS.items():
                response.headers.setdefault(k, v)
            record_http(method, route, response.status_code, duration)
            emit_log(logging.WARNING if response.status_code >= 400 else logging.INFO,
                     "http_request", request_id=rid, http_method=method, route=route,
                     status_code=response.status_code, duration_ms=round(duration * 1000, 2),
                     disease=disease,
                     model_identifier=getattr(request.state, "model_identifier", None),
                     release_status=getattr(request.state, "release_status", None),
                     error_category=error_category(response.status_code))
            return response
        finally:
            if count_in_flight:
                IN_FLIGHT.dec()


class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject over-large request bodies (Content-Length based) with 413."""

    async def dispatch(self, request: Request, call_next):
        limit = get_settings().max_request_bytes
        cl = request.headers.get("content-length")
        if cl is not None:
            try:
                if int(cl) > limit:
                    rid = getattr(request.state, "request_id", uuid.uuid4().hex)
                    return error_response(
                        413, "request_too_large",
                        f"Request body exceeds {limit} bytes", rid,
                    )
            except ValueError:
                pass
        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Very small fixed-window in-memory guard for prediction/report routes.

    Not a distributed rate limiter - it is a single-process abuse brake. Set
    RATE_LIMIT_PER_MINUTE=0 to disable. Keyed by client IP.
    """

    def __init__(self, app):
        super().__init__(app)
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    async def dispatch(self, request: Request, call_next):
        limit = get_settings().rate_limit_per_minute
        path = request.url.path
        guarded = path.startswith("/api/v1/predictions") or path.startswith("/api/v1/reports")
        if limit <= 0 or not guarded:
            return await call_next(request)
        client = request.client.host if request.client else "unknown"
        now = time.monotonic()
        window = self._hits[client]
        while window and now - window[0] > 60.0:
            window.popleft()
        if len(window) >= limit:
            RATE_LIMIT_REJECTIONS.inc()
            rid = getattr(request.state, "request_id", uuid.uuid4().hex)
            return error_response(
                429, "rate_limited",
                f"Rate limit of {limit} requests/minute exceeded", rid,
            )
        window.append(now)
        return await call_next(request)


def require_api_key(request: Request) -> None:
    """FastAPI dependency: enforce X-API-Key when API_KEY is configured.

    No-op in demo mode (API_KEY unset), so local development needs no header.
    """
    settings = get_settings()
    if not settings.auth_enabled:
        return
    supplied = request.headers.get("X-API-Key", "")
    if not hmac.compare_digest(supplied, settings.api_key or ""):
        AUTH_REJECTIONS.inc()
        rid = getattr(request.state, "request_id", uuid.uuid4().hex)
        from fastapi import HTTPException

        raise HTTPException(status_code=401, detail="Missing or invalid API key",
                            headers={"X-Request-ID": rid})
