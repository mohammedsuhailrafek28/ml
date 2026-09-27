"""Disease input contracts, generated frontend parity, and public OpenAPI schemas."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.api.contracts import FIELD_SPECS, MEASUREMENT_MODELS
from src.model_release import load_release_manifest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = load_release_manifest()


def valid_values(disease: str) -> dict:
    values = {}
    for field in FIELD_SPECS[disease]:
        if field["options"]:
            values[field["name"]] = field["options"][0]["value"]
        elif field["kind"] == "integer":
            values[field["name"]] = int(field["min"])
        else:
            values[field["name"]] = (field["min"] + field["max"]) / 2
    return values


@pytest.mark.parametrize("disease", sorted(MEASUREMENT_MODELS))
def test_contract_feature_order_matches_manifest_and_rejects_extras(disease):
    model = MEASUREMENT_MODELS[disease]
    active = MANIFEST["releases"][disease]["ordered_active_features"]
    assert list(model.model_fields)[:len(active)] == active
    assert model.model_validate(valid_values(disease))
    with pytest.raises(ValidationError):
        model.model_validate({**valid_values(disease), "unexpected": 1})


@pytest.mark.parametrize("disease", sorted(MEASUREMENT_MODELS))
def test_generated_frontend_contract_matches_backend_fields(disease):
    generated = json.loads((ROOT / "frontend/features/assessments/generated/contracts.schema.json").read_text(encoding="utf-8"))[disease]
    expected = list(MANIFEST["releases"][disease]["ordered_active_features"])
    assert list(generated["properties"]) == expected
    assert generated["additionalProperties"] is False
    required = set(generated.get("required", []))
    assert required == {field["name"] for field in FIELD_SPECS[disease] if field["required"]}
    for field in FIELD_SPECS[disease]:
        prop = generated["properties"][field["name"]]
        assert prop["title"] == field["label"]
        assert prop["description"] == field["help"]
        assert prop["ui"]["group"] == field["group"]
        assert prop["ui"]["display_order"] == field["display_order"]
        assert prop["ui"]["control"] == field["control"]
        assert prop["ui"]["kind"] == field["kind"]
        assert prop["ui"]["unit"] == field["unit"]
        assert prop["ui"]["step"] == field["step"]
        assert prop["ui"]["required"] == field["required"]
        assert prop["ui"]["nullable"] == field["nullable"]
        assert prop["ui"]["min"] == field["min"]
        assert prop["ui"]["max"] == field["max"]
        if field["options"]:
            enums = [prop["enum"]] if "enum" in prop else [schema["enum"] for schema in prop.get("anyOf", []) if "enum" in schema]
            assert enums == [[option["value"] for option in field["options"]]]
        else:
            expected_type = "integer" if field["kind"] == "integer" else "number"
            if field["nullable"]:
                assert any(schema.get("type") == expected_type for schema in prop["anyOf"])
            else:
                assert prop["type"] == expected_type
            if field["min"] is not None:
                assert prop.get("minimum") == field["min"] or prop.get("anyOf", [{}])[0].get("minimum") == field["min"]
            if field["max"] is not None:
                assert prop.get("maximum") == field["max"] or prop.get("anyOf", [{}])[0].get("maximum") == field["max"]


@pytest.mark.parametrize("disease", sorted(MEASUREMENT_MODELS))
def test_contract_accepts_inclusive_numeric_boundaries(disease):
    model = MEASUREMENT_MODELS[disease]
    values = valid_values(disease)
    for field in FIELD_SPECS[disease]:
        if field["kind"] == "categorical":
            continue
        for bound in (field["min"], field["max"]):
            candidate = dict(values)
            candidate[field["name"]] = int(bound) if field["kind"] == "integer" else bound
            model.model_validate(candidate)


def test_required_missing_fractional_integer_boolean_nonfinite_and_category_rejected():
    liver = MEASUREMENT_MODELS["liver"]
    values = valid_values("liver")
    for invalid in (
        {key: value for key, value in values.items() if key != "Age"},
        {**values, "Age": 40.5},
        {**values, "Age": True},
        {**values, "TB": float("nan")},
        {**values, "TB": float("inf")},
        {**values, "Gender": "Other"},
    ):
        with pytest.raises(ValidationError):
            liver.model_validate(invalid)
    with pytest.raises(ValidationError):
        liver.model_validate({**values, "TB": "1.2"})
    with pytest.raises(ValidationError):
        MEASUREMENT_MODELS["heart"].model_validate({**valid_values("heart"), "sex": True})


def test_optional_contract_fields_can_be_omitted_or_null():
    liver_values = valid_values("liver")
    liver_values.pop("A/G Ratio")
    assert "A/G Ratio" not in MEASUREMENT_MODELS["liver"].model_validate(liver_values).model_dump(exclude_unset=True)
    assert MEASUREMENT_MODELS["liver"].model_validate({**liver_values, "A/G Ratio": None})
    kidney = MEASUREMENT_MODELS["kidney"]
    assert kidney.model_validate({}).model_dump()["age"] is None


def test_disease_openapi_paths_keep_public_urls_and_specific_request_schemas():
    from src.api.main import app

    openapi = app.openapi()
    for disease in MEASUREMENT_MODELS:
        for prefix in ("predictions", "reports"):
            path = f"/api/v1/{prefix}/{disease}"
            assert path in openapi["paths"]
    assert "/api/v1/predictions/{disease}" not in openapi["paths"]
