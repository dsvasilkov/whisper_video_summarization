from pydantic import BaseModel


class TrainRequest(BaseModel):
    config_path: str
    dataset_path: str | None = None


class InferRequest(BaseModel):
    text: str


class InferResponse(BaseModel):
    summary: str


class InferVideoRequest(BaseModel):
    path: str
