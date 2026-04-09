import os
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from whisper_video_summarization.api.deps import get_db
from whisper_video_summarization.api.schemas import (
    ForgotPasswordRequest,
    ForgotPasswordResponse,
    ResetPasswordRequest,
    TokenResponse,
    UserLogin,
    UserRegister,
)
from whisper_video_summarization.api.security import (
    create_access_token,
    hash_password,
    hash_reset_token,
    new_reset_token,
    reset_token_expires_at,
    verify_password,
)
from whisper_video_summarization.db.models import User

DEBUG = os.getenv("DEBUG", "0").lower() in ("1", "true", "yes")

router = APIRouter()


def _normalize_email(email: str) -> str:
    return email.strip().lower()


def _expires_valid(expires: datetime | None) -> bool:
    if expires is None:
        return False
    exp = expires if expires.tzinfo else expires.replace(tzinfo=UTC)
    return exp >= datetime.now(UTC)


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(body: UserRegister, db: AsyncSession = Depends(get_db)):
    email = _normalize_email(str(body.email))
    result = await db.execute(select(User).where(User.email == email))
    if result.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Email already registered")
    user = User(email=email, hashed_password=hash_password(body.password))
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return TokenResponse(access_token=create_access_token(user.id))


@router.post("/login", response_model=TokenResponse)
async def login(body: UserLogin, db: AsyncSession = Depends(get_db)):
    email = _normalize_email(str(body.email))
    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()
    if not user or not verify_password(body.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )
    return TokenResponse(access_token=create_access_token(user.id))


@router.post("/forgot-password", response_model=ForgotPasswordResponse)
async def forgot_password(
    body: ForgotPasswordRequest,
    db: AsyncSession = Depends(get_db),
):
    email = _normalize_email(str(body.email))
    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()
    msg = (
        "Запрос принят. Если адрес зарегистрирован, используйте ссылку из письма "
        "для сброса пароля. В режиме DEBUG токен может быть возвращён в ответе API."
    )
    if not user:
        return ForgotPasswordResponse(message=msg, reset_token=None)
    token_plain = new_reset_token()
    user.password_reset_token_hash = hash_reset_token(token_plain)
    user.password_reset_expires = reset_token_expires_at()
    await db.commit()
    reset_token = token_plain if DEBUG else None
    return ForgotPasswordResponse(message=msg, reset_token=reset_token)


@router.post("/reset-password")
async def reset_password(body: ResetPasswordRequest, db: AsyncSession = Depends(get_db)):
    th = hash_reset_token(body.token)
    result = await db.execute(select(User).where(User.password_reset_token_hash == th))
    user = result.scalar_one_or_none()
    if not user or not _expires_valid(user.password_reset_expires):
        raise HTTPException(status_code=400, detail="Invalid or expired reset token")
    user.hashed_password = hash_password(body.new_password)
    user.password_reset_token_hash = None
    user.password_reset_expires = None
    await db.commit()
    return {"message": "Пароль успешно изменён"}
