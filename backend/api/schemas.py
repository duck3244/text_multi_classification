from typing import List

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000)
    threshold: float = Field(0.5, ge=0.0, le=1.0)


class LabelScore(BaseModel):
    name: str
    probability: float
    predicted: bool


class PredictResponse(BaseModel):
    text: str
    threshold: float
    labels: List[LabelScore]
    top: LabelScore


class LabelMeta(BaseModel):
    name: str
    description: str = ""


class LabelsResponse(BaseModel):
    labels: List[LabelMeta]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
