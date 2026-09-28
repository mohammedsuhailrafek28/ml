"""Liveness, readiness and model-artifact integrity checks.

Liveness  = the process is up (cheap, no I/O).
Readiness = every disease's persisted pipeline + metadata load and are
            internally consistent. A degraded result returns HTTP 503 so an
            orchestrator will not route traffic to a broken model.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from src.api.settings import get_settings
from src.api.observability import emit_log, set_readiness
from src.application_release import public_release_provenance
from src.utils.config import DISEASES

def liveness() -> dict:
    return {"status": "alive", **public_release_provenance()}


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
        except Exception:  # noqa: BLE001 - public health output must not leak paths/details
            issues.append("metadata.json is invalid")

    if meta:
        selected_model = meta.get("selected_model") or meta.get("selected_algorithm")
        if not selected_model:
            issues.append("metadata missing selected_model")
        if "holdout_metrics" not in meta and "metrics" not in meta:
            issues.append("metadata missing holdout metrics")
        ap = meta.get("artifact_path")
        if ap and not (settings.model_root.parent / ap).exists() and not pipe_path.exists():
            issues.append("metadata artifact_path does not resolve")

    if pipe_path.exists():
        try:
            from joblib import load

            model = load(pipe_path)
            if not hasattr(model, "predict_proba"):
                issues.append("loaded model has no predict_proba")
        except Exception:  # noqa: BLE001 - public health output must not leak paths/details
            issues.append("pipeline failed to load")

    return {
        "ok": not issues,
        "selected_model": selected_model,
        "issues": issues,
    }


def readiness() -> tuple[dict, int]:
    """Return (payload, http_status). 200 when ready, 503 when degraded."""
    from src.model_release import (
        load_release_manifest,
        release_file_issues,
        runtime_compatibility_issues,
    )

    results = {key: _check_disease(key) for key in DISEASES}
    settings = get_settings()
    release_manifest_path = settings.model_root / "release_manifest.json"
    try:
        manifest = load_release_manifest(release_manifest_path)
        release_issues = runtime_compatibility_issues(manifest) + release_file_issues(
            include_datasets=False, manifest=manifest, model_root=settings.model_root
        )
    except Exception:  # noqa: BLE001 - readiness must report safely, not crash
        release_issues = ["release manifest unavailable or invalid"]
    ready = all(r["ok"] for r in results.values()) and not release_issues
    payload = {
        "status": "ready" if ready else "degraded",
        **public_release_provenance(),
        "checkedAt": datetime.now(timezone.utc).isoformat(),
        "releaseIntegrity": {"ok": not release_issues, "issues": release_issues},
        "diseases": results,
    }
    return payload, (200 if ready else 503)


def startup_integrity_log(logger) -> bool:
    """Run readiness once at startup and log the outcome. Returns True if ready."""
    payload, status = readiness()
    ready = status == 200
    set_readiness(ready)
    if status == 200:
        emit_log(logging.INFO, "startup_integrity", status="ready")
        return True
    broken = [k for k, value in payload["diseases"].items() if not value["ok"]]
    emit_log(logging.ERROR, "startup_integrity", status="degraded",
             affected_diseases=broken,
             error_category="release_integrity" if not payload["releaseIntegrity"]["ok"] else "model_integrity")
    return False
