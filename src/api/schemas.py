from pydantic import BaseModel, Field
from typing import Any
class PredictionRequest(BaseModel): measurements: dict[str,Any]=Field(min_length=1)
