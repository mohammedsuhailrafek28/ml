"""Typed public API contracts."""
from typing import Any, Literal

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    measurements: dict[str, Any] = Field(min_length=1)


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
