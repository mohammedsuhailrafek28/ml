"""Safe application release identity shared by API health and operations."""
from __future__ import annotations

import hashlib
import json
import os
import re
from functools import lru_cache

from src.utils.config import ROOT

APPLICATION_RELEASE_PATH = ROOT / "release" / "application.json"
MODEL_RELEASE_PATH = ROOT / "models" / "release_manifest.json"
_REVISION = re.compile(r"^(?:[0-9a-f]{7,64}|unknown|local)$")


@lru_cache(maxsize=1)
def application_release() -> dict[str, str]:
    """Read the committed, non-secret and deterministic release identity."""
    document = json.loads(APPLICATION_RELEASE_PATH.read_text(encoding="utf-8"))
    return {
        "version": str(document["version"]),
        "release_status": str(document["release_status"]),
        "intended_use": str(document["intended_use"]),
        "clinical_status": str(document["clinical_status"]),
        "runtime_versions": document["runtime_versions"],
    }


def safe_source_revision() -> str:
    """Expose only a Git-like revision token, never arbitrary env contents."""
    revision = os.environ.get("APP_SOURCE_REVISION", "unknown").strip().lower()
    return revision if _REVISION.fullmatch(revision) else "unknown"


def public_release_provenance() -> dict:
    app = application_release()
    model_doc = json.loads(MODEL_RELEASE_PATH.read_text(encoding="utf-8"))
    return {
        **app,
        "source_revision": safe_source_revision(),
        "model_release": {
            "identifier": str(model_doc["release_id"]),
            "manifest_sha256": hashlib.sha256(MODEL_RELEASE_PATH.read_bytes()).hexdigest(),
        },
    }
