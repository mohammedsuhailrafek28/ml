"""Typed public API contracts."""
from typing import Literal
from pydantic import create_model
from src.api.contracts import MEASUREMENT_MODELS

from pydantic import BaseModel, ConfigDict, Field


def _request_model(disease: str):
    return create_model(
        f"{disease.title()}PredictionRequest",
        __config__=ConfigDict(extra="forbid"),
        measurements=(MEASUREMENT_MODELS[disease], Field(..., description="Disease-specific validated input fields.")),
    )


class PredictionRequest(BaseModel):
    """Backwards-compatible generic import; route handlers use disease models."""
    measurements: dict[str, object] = Field(min_length=1)


DISEASE_REQUEST_MODELS = {disease: _request_model(disease) for disease in MEASUREMENT_MODELS}
LiverPredictionRequest = DISEASE_REQUEST_MODELS["liver"]
DiabetesPredictionRequest = DISEASE_REQUEST_MODELS["diabetes"]
HeartPredictionRequest = DISEASE_REQUEST_MODELS["heart"]
KidneyPredictionRequest = DISEASE_REQUEST_MODELS["kidney"]
ParkinsonsPredictionRequest = DISEASE_REQUEST_MODELS["parkinsons"]


class PredictionResponse(BaseModel):
    disease: str
    prediction: Literal[0, 1]
    model_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Uncalibrated positive-class model output; not disease probability.",
    )
    decision_threshold: float = Field(
        ge=0.0,
        le=1.0,
        description="Classification threshold from the authoritative release manifest.",
    )
    threshold_result: Literal["at_or_above", "below"]
    score_type: Literal["uncalibrated_model_score"] = Field(
        description="Explicit semantic label for model_score."
    )
    model_identifier: str
    release_status: str
    intended_use: str
    limitations: list[str]
    disclaimer: str
