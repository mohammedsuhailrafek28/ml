"""Lightweight production-baseline middleware: request IDs, structured logging,
body-size limit, a simple in-memory abuse guard, security headers, and a shared
error envelope. No new dependencies; nothing here touches model behaviour.

Logging deliberately records only safe metadata (endpoint, disease, status,
latency, request id) and NEVER the request body / measurement payload.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from collections import defaultdict, deque

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from src.api.settings import get_settings

logger = logging.getLogger("medical_ai.api")

_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store",
}


def _disease_from_path(path: str) -> str | None:
    parts = [p for p in path.split("/") if p]
    if len(parts) >= 4 and parts[:2] == ["api", "v1"] and parts[2] in {
        "predictions", "reports", "diseases",
    }:
        return parts[3]
    return None


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
        rid = request.headers.get("X-Request-ID", "").strip() or uuid.uuid4().hex
        request.state.request_id = rid
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:  # noqa: BLE001 - converted to a safe 500 below
            latency_ms = round((time.perf_counter() - start) * 1000, 1)
            logger.error(
                json.dumps(
                    {
                        "event": "request_error",
                        "request_id": rid,
                        "method": request.method,
                        "path": request.url.path,
                        "disease": _disease_from_path(request.url.path),
                        "status_code": 500,
                        "latency_ms": latency_ms,
                    }
                )
            )
            settings = get_settings()
            msg = "Internal server error" if settings.is_production else "Unhandled exception"
            return error_response(500, "internal_error", msg, rid)
        latency_ms = round((time.perf_counter() - start) * 1000, 1)
        response.headers["X-Request-ID"] = rid
        for k, v in _SECURITY_HEADERS.items():
            response.headers.setdefault(k, v)
        logger.info(
            json.dumps(
                {
                    "event": "request",
                    "request_id": rid,
                    "method": request.method,
                    "path": request.url.path,
                    "disease": _disease_from_path(request.url.path),
                    "status_code": response.status_code,
                    "latency_ms": latency_ms,
                }
            )
        )
        return response


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
    if supplied != settings.api_key:
        rid = getattr(request.state, "request_id", uuid.uuid4().hex)
        from fastapi import HTTPException

        raise HTTPException(status_code=401, detail="Missing or invalid API key",
                            headers={"X-Request-ID": rid})
