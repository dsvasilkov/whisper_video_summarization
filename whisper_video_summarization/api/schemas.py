from pydantic import BaseModel
from typing import Optional


class TrainRequest(BaseModel):
    config_path: str
    dataset_path: Optional[str] = None


class InferRequest(BaseModel):
    text: str


class InferResponse(BaseModel):
    summary: str


class InferVideoRequest(BaseModel):
    path: str
