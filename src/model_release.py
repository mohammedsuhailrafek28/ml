"""Verification helpers for the checked-in persisted-model release."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

from src.utils.config import ROOT

MANIFEST_PATH = ROOT / "models" / "release_manifest.json"
PACKAGE_NAMES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "scikit-learn": "scikit-learn",
    "joblib": "joblib",
}
TEXT_SUFFIXES = {
    ".csv", ".css", ".example", ".json", ".lock", ".md", ".mjs", ".py",
    ".toml", ".ts", ".tsx", ".yaml", ".yml",
}


def load_release_manifest(path: Path = MANIFEST_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    content = path.read_bytes()
    # Git may check text files out with platform-native line endings. Hash the
    # canonical LF representation so release manifests verify across OSes.
    if path.suffix.lower() in TEXT_SUFFIXES or path.name in {"Dockerfile", ".dockerignore"}:
        content = content.replace(b"\r\n", b"\n")
    return hashlib.sha256(content).hexdigest()


def training_metadata_dataset_sha256(path: Path) -> str:
    """Reproduce the CRLF-byte digest stored by the original Windows training runs."""
    content = path.read_bytes().replace(b"\r\n", b"\n")
    content = content.replace(b"\n", b"\r\n")
    return hashlib.sha256(content).hexdigest()


def active_features(disease: str, manifest: dict | None = None) -> list[str]:
    doc = manifest or load_release_manifest()
    return list(doc["releases"][disease]["ordered_active_features"])


def runtime_compatibility_issues(manifest: dict | None = None) -> list[str]:
    doc = manifest or load_release_manifest()
    expected = doc["release_environment"]
    issues: list[str] = []
    actual_python = ".".join(platform.python_version_tuple()[:2])
    if actual_python != expected["python"]:
        issues.append(f"python version mismatch: expected {expected['python']}, got {actual_python}")
    for manifest_name, package_name in PACKAGE_NAMES.items():
        actual = importlib.metadata.version(package_name)
        if actual != expected[manifest_name]:
            issues.append(f"{manifest_name} version mismatch: expected {expected[manifest_name]}, got {actual}")
    return issues


def release_file_issues(
    include_datasets: bool = True,
    manifest: dict | None = None,
    model_root: Path | None = None,
) -> list[str]:
    doc = manifest or load_release_manifest()
    issues: list[str] = []
    for disease, release in doc["releases"].items():
        entries = ["model", "metadata", "feature_list"]
        if include_datasets:
            entries.append("dataset")
        for entry in entries:
            item = release[entry]
            if model_root is not None and entry in {"model", "metadata", "feature_list"}:
                path = model_root / disease / Path(item["path"]).name
            else:
                path = ROOT / item["path"]
            if not path.exists():
                issues.append(f"{disease} {entry} missing: {item['path']}")
                continue
            actual = file_sha256(path)
            if actual != item["sha256"]:
                issues.append(
                    f"{disease} {entry} hash mismatch: expected {item['sha256']}, got {actual}"
                )
    return issues


def assert_release_compatible(include_datasets: bool = True) -> None:
    issues = runtime_compatibility_issues() + release_file_issues(include_datasets=include_datasets)
    if issues:
        raise RuntimeError("Persisted model release is incompatible:\n- " + "\n- ".join(issues))
