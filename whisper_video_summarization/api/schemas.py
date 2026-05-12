from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field


class InferRequest(BaseModel):
    text: str


class InferResponse(BaseModel):
    summary: str


class InferAudioRequest(BaseModel):
    path: str


class TaskCreateResponse(BaseModel):
    task_id: UUID


class UploadPathResponse(BaseModel):
    path: str


class PresignAudioUploadRequest(BaseModel):
    filename: str = Field(min_length=1, max_length=512)
    content_type: str = Field(min_length=1, max_length=256)
    sha256: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[a-f0-9]{64}$",
        description="SHA-256 hex (lowercase) of the exact bytes being uploaded",
    )


class PresignAudioUploadResponse(BaseModel):
    task_id: UUID
    upload_url: str
    required_headers: dict[str, str]
    s3_uri: str


class TaskStatusResponse(BaseModel):
    task_id: UUID
    status: str
    task_type: str
    result_transcription: dict[str, Any] | None = None
    result_summary: str | None = None
    result_topic_graph: dict[str, Any] | None = None
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime


class UserRegister(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=256)


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ForgotPasswordResponse(BaseModel):
    message: str
    reset_token: str | None = None


class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str = Field(min_length=8, max_length=256)


class TaskQuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=16000)


class TaskQuestionAnswerResponse(BaseModel):
    answer: str


class ChunkEmbeddingItem(BaseModel):
    chunk_id: int
    embedding: list[float]


class TaskChunkEmbeddingsResponse(BaseModel):
    chunks: list[ChunkEmbeddingItem]