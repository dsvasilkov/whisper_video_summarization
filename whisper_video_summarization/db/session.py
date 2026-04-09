import os
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session, sessionmaker

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://postgres:postgres@localhost:5432/whisper_inference",
)


def _sync_db_url() -> str:
    return DATABASE_URL


def _async_db_url() -> str:
    # Заменяем sync-драйвер на async-драйвер psycopg при необходимости.
    if DATABASE_URL.startswith("postgresql+psycopg2"):
        return DATABASE_URL.replace("postgresql+psycopg2", "postgresql+psycopg")
    if DATABASE_URL.startswith("postgresql://"):
        return DATABASE_URL.replace("postgresql://", "postgresql+psycopg://")
    return DATABASE_URL


def get_engine():
    """Синхронный engine (используется для Alembic и, при необходимости, старого sync-кода)."""
    from sqlalchemy import create_engine

    return create_engine(
        _sync_db_url(),
        pool_pre_ping=True,
        echo=os.getenv("SQL_ECHO", "0") == "1",
    )


def get_session_factory():
    """Синхронная фабрика сессий (для совместимости, Celery и т.п.)."""
    engine = get_engine()
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_async_engine():
    """Асинхронный engine для работы через AsyncSession."""
    return create_async_engine(
        _async_db_url(),
        pool_pre_ping=True,
        echo=os.getenv("SQL_ECHO", "0") == "1",
    )


def get_async_session_factory() -> async_sessionmaker[AsyncSession]:
    engine = get_async_engine()
    return async_sessionmaker(
        bind=engine,
        expire_on_commit=False,
        autoflush=False,
        autocommit=False,
    )


def _alembic_config() -> Config:
    root = Path(__file__).resolve().parents[2]
    ini = root / "alembic.ini"
    if not ini.is_file():
        raise FileNotFoundError(f"alembic.ini not found at {ini}")
    return Config(str(ini))


def init_db() -> None:
    """Применить миграции Alembic до head (старт API, Job, локальная разработка)."""
    command.upgrade(_alembic_config(), "head")


@contextmanager
def get_db():
    """Синхронный контекст работы с БД (для фоновых задач, миграций и т.п.)."""
    SessionLocal = get_session_factory()
    session: Session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@asynccontextmanager
async def get_async_db():
    """Асинхронный контекст работы с БД (используется в FastAPI)."""
    SessionLocal = get_async_session_factory()
    async with SessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
