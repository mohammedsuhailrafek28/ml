"""Liveness, readiness and model-artifact integrity checks.

Liveness  = the process is up (cheap, no I/O).
Readiness = every disease's persisted pipeline + metadata load and are
            internally consistent. A degraded result returns HTTP 503 so an
            orchestrator will not route traffic to a broken model.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from src.api.settings import get_settings
from src.utils.config import DISEASES

API_VERSION = "1.1"


def liveness() -> dict:
    return {"status": "alive", "version": API_VERSION}


def _check_disease(key: str) -> dict:
    settings = get_settings()
    model_dir = settings.model_root / key
    issues: list[str] = []

    pipe_path = model_dir / f"{key}_pipeline.joblib"
    meta_path = model_dir / "metadata.json"
    selected_model = None

    if not pipe_path.exists():
        issues.append(f"missing pipeline artifact: {pipe_path.name}")
    if not meta_path.exists():
        issues.append("missing metadata.json")

    meta: dict = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except Exception as exc:  # noqa: BLE001
            issues.append(f"metadata.json unparseable: {exc}")

    if meta:
        selected_model = meta.get("selected_model") or meta.get("selected_algorithm")
        if not selected_model:
            issues.append("metadata missing selected_model")
        if "holdout_metrics" not in meta and "metrics" not in meta:
            issues.append("metadata missing holdout metrics")
        ap = meta.get("artifact_path")
        if ap and not (settings.model_root.parent / ap).exists() and not pipe_path.exists():
            issues.append(f"metadata artifact_path does not resolve: {ap}")

    if pipe_path.exists():
        try:
            from joblib import load

            model = load(pipe_path)
            if not hasattr(model, "predict_proba"):
                issues.append("loaded model has no predict_proba")
        except Exception as exc:  # noqa: BLE001
            issues.append(f"pipeline failed to load: {exc}")

    return {
        "ok": not issues,
        "selected_model": selected_model,
        "issues": issues,
    }


def readiness() -> tuple[dict, int]:
    """Return (payload, http_status). 200 when ready, 503 when degraded."""
    results = {key: _check_disease(key) for key in DISEASES}
    ready = all(r["ok"] for r in results.values())
    payload = {
        "status": "ready" if ready else "degraded",
        "version": API_VERSION,
        "checkedAt": datetime.now(timezone.utc).isoformat(),
        "modelRoot": str(get_settings().model_root),
        "diseases": results,
    }
    return payload, (200 if ready else 503)


def startup_integrity_log(logger) -> bool:
    """Run readiness once at startup and log the outcome. Returns True if ready."""
    payload, status = readiness()
    if status == 200:
        logger.info(json.dumps({"event": "startup_integrity", "status": "ready"}))
        return True
    broken = {k: v["issues"] for k, v in payload["diseases"].items() if not v["ok"]}
    logger.error(
        json.dumps({"event": "startup_integrity", "status": "degraded", "problems": broken})
    )
    return False
