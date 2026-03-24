from datetime import datetime
from uuid import UUID

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


class TaskCreateResponse(BaseModel):
    task_id: UUID


class TaskStatusResponse(BaseModel):
    task_id: UUID
    status: str
    task_type: str
    result_transcription: str | None = None
    result_summary: str | None = None
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime
