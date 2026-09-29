"""Build the authoritative manifest for the already-persisted model release.

This script does not train or mutate a model. It inventories the checked-in
datasets, metadata, feature lists and Joblib pipelines, then writes a
deterministic JSON release contract for human review.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from joblib import load

from src.model_release import file_sha256
from src.utils.config import DISEASES

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "models" / "release_manifest.json"
RELEASE_ID = "medical-ai-suite-model-release-2026-09-24"
RELEASE_ENVIRONMENT = {
    "python": "3.11",
    "numpy": "2.4.6",
    "pandas": "3.0.6",
    "scipy": "1.17.1",
    "scikit-learn": "1.9.0",
    "joblib": "1.5.3",
}
STATUS = {
    "liver": "educational/research release",
    "diabetes": "educational/research release",
    "heart": "educational/research release",
    "kidney": "educational/research release",
    "parkinsons": "experimental",
}
NEGATIVE_CLASS = {
    "liver": "no liver-disease pattern (ILPD Selector == 2)",
    "diabetes": "no diabetes onset within 5 years (Pima Outcome == 0)",
    "heart": "no angiographic heart-disease pattern (Cleveland num == 0)",
    "kidney": "no chronic kidney disease (UCI class == notckd)",
    "parkinsons": "control subject (status == 0)",
}
OPTIONAL_FIELDS = {
    "liver": ["A/G Ratio"],
    "diabetes": [],
    "heart": ["ca", "thal"],
    "kidney": None,  # every active feature is optional and imputed
    "parkinsons": [],
}


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _evaluation_method(meta: dict) -> dict:
    result = {
        "split_strategy": meta["split_strategy"],
        "model_selection": meta.get("model_selection", {}).get("rule"),
        "cross_validation": meta.get("cv_metrics", {}).get("cv", "stratified CV on development partition"),
        "holdout_is_reused": False,
    }
    if meta.get("group_cv_strategy"):
        result["grouping_strategy"] = meta["group_cv_strategy"]
    return result


def build_manifest() -> dict:
    releases: dict[str, dict] = {}
    for disease, config in DISEASES.items():
        model_dir = ROOT / "models" / disease
        dataset_path = config.raw_path
        model_path = model_dir / f"{disease}_pipeline.joblib"
        metadata_path = model_dir / "metadata.json"
        feature_path = model_dir / "feature_names.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        transformed_features = json.loads(feature_path.read_text(encoding="utf-8"))
        active_features = list(metadata.get("active_features", config.features))
        pipeline = load(model_path)
        steps = [
            {"name": name, "type": f"{type(component).__module__}.{type(component).__name__}"}
            for name, component in pipeline.steps
        ]
        optional = active_features if OPTIONAL_FIELDS[disease] is None else OPTIONAL_FIELDS[disease]
        required = [feature for feature in active_features if feature not in optional]
        holdout = metadata["holdout_metrics"]
        cv_metrics = metadata["cv_metrics"]
        model_hash = file_sha256(model_path)

        releases[disease] = {
            "release_schema_version": "1.0.0",
            "disease_identifier": disease,
            "metadata_disease_label": metadata["disease"],
            "dataset": {"path": relative(dataset_path), "sha256": file_sha256(dataset_path)},
            "model": {
                "path": relative(model_path),
                "sha256": model_hash,
                "identifier": f"{disease}:{metadata['selected_model']}:{model_hash[:12]}",
            },
            "metadata": {"path": relative(metadata_path), "sha256": file_sha256(metadata_path)},
            "feature_list": {
                "path": relative(feature_path),
                "sha256": file_sha256(feature_path),
                "transformed_features": transformed_features,
            },
            "ordered_active_features": active_features,
            "feature_count": len(active_features),
            "pipeline_steps": steps,
            "estimator_type": steps[-1]["type"],
            "threshold": float(metadata.get("operating_threshold", metadata.get("threshold", 0.5))),
            "classes": {
                "positive": {"value": 1, "definition": metadata["positive_class"]},
                "negative": {"value": 0, "definition": NEGATIVE_CLASS[disease]},
            },
            "evaluation_method": _evaluation_method(metadata),
            "verified_metrics": {
                "development_cv": cv_metrics,
                "subject_disjoint_holdout" if disease == "parkinsons" else "holdout": holdout,
            },
            "training_versions": {
                "scikit-learn": metadata["sklearn_version"],
                "numpy": "not recorded by trainer",
                "pandas": "not recorded by trainer",
                "scipy": "not recorded by trainer",
                "joblib": "not recorded by trainer",
            },
            "runtime_versions": RELEASE_ENVIRONMENT,
            "api_request_schema": {
                "endpoint": f"POST /api/v1/predictions/{disease}",
                "body": {"measurements": "object"},
                "ordered_measurement_fields": active_features,
                "required_fields": required,
                "optional_imputed_fields": optional,
                "additional_fields_allowed": False,
            },
            "preprocessing": metadata["preprocessing"],
            "intended_use": "Educational and research demonstration of persisted-model inference; not diagnosis, screening, triage, or treatment guidance.",
            "known_limitations": metadata["limitations"],
            "release_status": STATUS[disease],
        }

    return {
        "release_schema_version": "1.0.0",
        "release_id": RELEASE_ID,
        "created_date": "2026-09-24",
        "authoritative": True,
        "release_environment": RELEASE_ENVIRONMENT,
        "releases": releases,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true", help="Fail if output differs from the generated manifest")
    args = parser.parse_args()
    rendered = json.dumps(build_manifest(), indent=2, ensure_ascii=False) + "\n"
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if args.check:
        if not output.exists() or output.read_text(encoding="utf-8") != rendered:
            raise SystemExit(f"release manifest is stale: run {Path(__file__).name}")
        print(f"release manifest is current: {relative(output)}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    print(f"wrote {relative(output)}")


if __name__ == "__main__":
    main()
