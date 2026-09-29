"""Verify and summarize the current five persisted production pipelines.

This deliberately does not recreate train/test splits. The authoritative
metrics are those recorded when each released artifact was trained. This
script verifies hashes, runs the locked golden vectors through Registry.predict
(the production service path), and publishes a release-identity report.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.api.services.model_registry import Registry  # noqa: E402
from src.model_release import (  # noqa: E402
    load_release_manifest,
    release_file_issues,
    runtime_compatibility_issues,
)

GOLDEN_PATH = ROOT / "tests" / "fixtures" / "model_release_golden.json"
OUTPUT_DIR = ROOT / "reports" / "evaluation" / "current"


def evaluate() -> dict:
    manifest = load_release_manifest()
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    registry = Registry()
    file_issues = release_file_issues(include_datasets=True, manifest=manifest)
    environment_issues = runtime_compatibility_issues(manifest)
    if file_issues or environment_issues:
        raise RuntimeError("release verification failed:\n- " + "\n- ".join(file_issues + environment_issues))

    releases: dict[str, dict] = {}
    for vector in golden["vectors"]:
        disease = vector["disease_identifier"]
        contract = manifest["releases"][disease]
        result = registry.predict(disease, vector["measurements"])
        releases[disease] = {
            "model_identifier": contract["model"]["identifier"],
            "model_sha256": contract["model"]["sha256"],
            "metadata_sha256": contract["metadata"]["sha256"],
            "dataset_sha256": contract["dataset"]["sha256"],
            "estimator_type": contract["estimator_type"],
            "decision_threshold": result["decision_threshold"],
            "golden_prediction": result["prediction"],
            "golden_model_score": result["model_score"],
            "evaluation_method": contract["evaluation_method"],
            "verified_metrics": contract["verified_metrics"],
            "release_status": contract["release_status"],
        }
    return {
        "report_schema_version": "1.0.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "release_id": manifest["release_id"],
        "method": "hash verification plus golden vectors through src.api.services.model_registry.Registry.predict",
        "retrained": False,
        "recreated_historical_split": False,
        "releases": releases,
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# Current persisted-model release verification",
        "",
        f"Release: `{report['release_id']}`",
        "",
        "This is the current report. It verifies file identity and executes one locked",
        "golden prediction through the production registry for each disease. It does not",
        "retrain models or recreate historical holdout splits. Evaluation metrics below",
        "come from the metadata stored with each released artifact.",
        "",
        "| Disease | Model | SHA-256 | Golden class | Golden model score | Status |",
        "|---|---|---|---:|---:|---|",
    ]
    for disease, result in report["releases"].items():
        lines.append(
            f"| {disease} | {result['estimator_type'].split('.')[-1]} | "
            f"`{result['model_sha256']}` | {result['golden_prediction']} | "
            f"{result['golden_model_score']:.12f} | {result['release_status']} |"
        )
    lines.extend([
        "",
        "Parkinson's remains experimental: its subject-disjoint holdout contains only",
        "seven subjects and reports ROC-AUC 0.5860 with specificity 0.0 at threshold 0.5.",
        "The stronger development group-CV result is not a substitute for external validation.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Verify only; do not write reports")
    args = parser.parse_args()
    report = evaluate()
    if args.check:
        print(json.dumps(report, indent=2))
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "release_verification.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (OUTPUT_DIR / "README.md").write_text(render_markdown(report), encoding="utf-8")
    print(f"wrote {OUTPUT_DIR.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
