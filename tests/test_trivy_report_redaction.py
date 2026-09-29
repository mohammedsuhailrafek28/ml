import json

import pytest

from scripts.redact_trivy_report import redact_report


def test_redact_report_preserves_finding_metadata_without_match_content():
    marker = "placeholder text that must not be included in an artifact"
    report = {
        "SchemaVersion": 2,
        "Results": [
            {
                "Target": "image-layer/path/config.txt",
                "Secrets": [
                    {
                        "RuleID": "generic-api-key",
                        "Category": "General",
                        "Severity": "HIGH",
                        "Title": "Generic API Key",
                        "StartLine": 4,
                        "Match": marker,
                        "Code": {"Lines": [{"Content": marker}]},
                    }
                ],
            }
        ],
    }

    redacted, finding_count = redact_report(report)
    serialized = json.dumps(redacted)

    assert finding_count == 1
    assert marker not in serialized
    assert "Match" not in serialized
    assert "Code" not in serialized
    assert redacted["Results"][0]["Secrets"] == [
        {
            "RuleID": "generic-api-key",
            "Category": "General",
            "Severity": "HIGH",
            "Title": "Generic API Key",
            "StartLine": 4,
        }
    ]


def test_redact_report_rejects_unexpected_trivy_shape():
    with pytest.raises(ValueError):
        redact_report({"Results": [{"Secrets": {"unexpected": "shape"}}]})
