"""Produce a safe Trivy JSON artifact without embedded secret match text."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SAFE_SECRET_FIELDS = {
    "RuleID",
    "Category",
    "Severity",
    "Title",
    "StartLine",
    "EndLine",
    "StartColumn",
    "EndColumn",
    "Layer",
}


def redact_report(report: Any) -> tuple[Any, int]:
    if not isinstance(report, dict) or not isinstance(report.get("Results"), list):
        raise ValueError("Trivy report must be a JSON object with a Results array")

    finding_count = 0
    for result in report["Results"]:
        if not isinstance(result, dict):
            raise ValueError("Trivy Results entries must be JSON objects")
        findings = result.get("Secrets")
        if findings is None:
            continue
        if not isinstance(findings, list):
            raise ValueError("Trivy Secrets entries must be a JSON array")
        finding_count += len(findings)
        redacted_findings = []
        for finding in findings:
            if not isinstance(finding, dict):
                raise ValueError("Trivy secret findings must be JSON objects")
            redacted_findings.append(
                {key: value for key, value in finding.items() if key in SAFE_SECRET_FIELDS}
            )
        result["Secrets"] = redacted_findings
    return report, finding_count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    try:
        report = json.loads(args.input.read_text(encoding="utf-8"))
        redacted, finding_count = redact_report(report)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        failure_report = {
            "report_status": "missing_or_invalid",
            "error_type": type(error).__name__,
            "secret_values_included": False,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(failure_report, indent=2) + "\n", encoding="utf-8")
        print(f"Trivy report redaction failed: {type(error).__name__}")
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(redacted, indent=2) + "\n", encoding="utf-8")
    print(f"Redacted Trivy JSON report written; secret findings={finding_count} (values removed).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
