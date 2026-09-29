"""Generate or verify the deterministic application release-input manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.model_release import file_sha256

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "release" / "application-release-manifest.json"
INPUTS = {
    "application_identity": "release/application.json",
    "compose_environment_example": ".env.example",
    "model_release_manifest": "models/release_manifest.json",
    "python_dependency_lock": "requirements.lock",
    "frontend_package_lock": "frontend/package-lock.json",
    "frontend_package_manifest": "frontend/package.json",
    "backend_dockerfile": "Dockerfile",
    "frontend_dockerfile": "frontend/Dockerfile",
    "frontend_contract_types": "frontend/features/assessments/generated/contracts.ts",
    "frontend_contract_schema": "frontend/features/assessments/generated/contracts.schema.json",
    "frontend_next_config": "frontend/next.config.mjs",
    "backend_dockerignore": ".dockerignore",
    "frontend_dockerignore": "frontend/.dockerignore",
    "compose_topology": "docker-compose.yml",
    "backend_app": "src/api/main.py",
    "backend_health": "src/api/health.py",
    "backend_observability": "src/api/observability.py",
    "backend_settings": "src/api/settings.py",
    "release_provenance_runtime": "src/application_release.py",
    "frontend_bff": "frontend/lib/backend.mjs",
    "frontend_release_footer": "frontend/app/release-footer.tsx",
    "frontend_global_styles": "frontend/app/globals.css",
    "browser_e2e_runner": "frontend/scripts/run-e2e.mjs",
    "standalone_assets_setup": "frontend/scripts/prepare-standalone.mjs",
    "container_smoke_tool": "scripts/container_smoke.mjs",
    "container_runtime_check": "scripts/check_container_runtime.py",
    "container_metrics_auth_check": "scripts/check_container_metrics_auth.py",
    "container_browser_check": "scripts/check_container_browser_boundary.py",
    "load_smoke_runner": "scripts/run_load_smoke.py",
    "container_browser_spec": "frontend/e2e/container-release.spec.ts",
    "release_ci_workflow": ".github/workflows/release-candidate.yml",
}


def build_manifest() -> dict:
    identity = json.loads((ROOT / INPUTS["application_identity"]).read_text(encoding="utf-8"))
    sample_version = next(
        (line.partition("=")[2].strip() for line in (ROOT / INPUTS["compose_environment_example"]).read_text(encoding="utf-8").splitlines()
         if line.startswith("APP_VERSION=")),
        "",
    )
    if sample_version != identity["version"]:
        raise ValueError(".env.example APP_VERSION does not match the authoritative application identity")
    return {
        "schema_version": "1.0.0",
        "application_version": identity["version"],
        "release_status": identity["release_status"],
        "intended_use": identity["intended_use"],
        "clinical_status": identity["clinical_status"],
        "runtime_versions": identity["runtime_versions"],
        "inputs": {
            name: {"path": relative, "sha256": file_sha256(ROOT / relative)}
            for name, relative in INPUTS.items()
        },
    }


def render() -> str:
    return json.dumps(build_manifest(), indent=2, sort_keys=True) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if the checked-in manifest is stale")
    args = parser.parse_args()
    expected = render()
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_text(encoding="utf-8") != expected:
            print("application release manifest is stale; run python -m scripts.build_application_release_manifest")
            return 1
        print(f"application release manifest is current: {OUTPUT.relative_to(ROOT)}")
        return 0
    OUTPUT.write_text(expected, encoding="utf-8")
    print(f"wrote {OUTPUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
