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


class TaskStatusResponse(BaseModel):
    task_id: UUID
    status: str
    task_type: str
    result_transcription: dict[str, Any] | None = None
    result_summary: str | None = None
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
