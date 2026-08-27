"""Environment-driven runtime configuration for the FastAPI service.

No new dependency: a small frozen dataclass populated from ``os.environ`` with
safe local-development defaults. Nothing here affects model training or metrics.

Environment variables (all optional):

    APP_ENV               "development" (default) | "production"
    ALLOWED_ORIGINS       comma-separated CORS origins.
                          Default in development: http://localhost:3000
                          In production: empty unless set (no wildcard).
    API_HOST              bind host for `python -m src.api` (default 127.0.0.1)
    API_PORT              bind port (default 8000)
    API_KEY               if set, /predictions and /reports require this key in
                          the `X-API-Key` header. If unset -> open demo mode.
    MAX_REQUEST_BYTES     max request body size (default 65536)
    RATE_LIMIT_PER_MINUTE per-client request cap for prediction/report routes
                          (default 120; set 0 to disable)
    MODEL_ROOT            directory holding the per-disease model folders
                          (default: <repo>/models)
    REPORT_DIR            directory for generated PDF reports
                          (default: system temp dir)
    LOG_LEVEL             root log level (default INFO)
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TRUE = {"1", "true", "yes", "on"}


def _split_origins(raw: str) -> tuple[str, ...]:
    return tuple(o.strip() for o in raw.split(",") if o.strip())


@dataclass(frozen=True)
class Settings:
    app_env: str = "development"
    allowed_origins: tuple[str, ...] = ("http://localhost:3000",)
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    api_key: str | None = None
    max_request_bytes: int = 64 * 1024
    rate_limit_per_minute: int = 120
    model_root: Path = field(default_factory=lambda: _REPO_ROOT / "models")
    report_dir: Path = field(default_factory=lambda: Path(tempfile.gettempdir()))
    log_level: str = "INFO"

    @property
    def is_production(self) -> bool:
        return self.app_env.lower() == "production"

    @property
    def auth_enabled(self) -> bool:
        return bool(self.api_key)

    @property
    def debug_errors(self) -> bool:
        """Only expose exception detail outside production."""
        return not self.is_production


def _load() -> Settings:
    env = os.environ.get("APP_ENV", "development").strip() or "development"

    if "ALLOWED_ORIGINS" in os.environ:
        origins = _split_origins(os.environ["ALLOWED_ORIGINS"])
    elif env.lower() == "production":
        origins = ()  # must be set explicitly in production; never wildcard
    else:
        origins = ("http://localhost:3000",)

    def _int(name: str, default: int) -> int:
        try:
            return int(os.environ.get(name, default))
        except (TypeError, ValueError):
            return default

    model_root = Path(os.environ.get("MODEL_ROOT", _REPO_ROOT / "models")).resolve()
    report_dir = Path(os.environ.get("REPORT_DIR", tempfile.gettempdir())).resolve()

    return Settings(
        app_env=env,
        allowed_origins=origins,
        api_host=os.environ.get("API_HOST", "127.0.0.1"),
        api_port=_int("API_PORT", 8000),
        api_key=(os.environ.get("API_KEY") or None),
        max_request_bytes=_int("MAX_REQUEST_BYTES", 64 * 1024),
        rate_limit_per_minute=_int("RATE_LIMIT_PER_MINUTE", 120),
        model_root=model_root,
        report_dir=report_dir,
        log_level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return _load()


def reload_settings() -> Settings:
    """Test helper: clear the cache and re-read the environment."""
    get_settings.cache_clear()
    return get_settings()
