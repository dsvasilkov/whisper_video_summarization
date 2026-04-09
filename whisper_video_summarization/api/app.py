import logging
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import UUID

import aiofiles
from fastapi import (
    BackgroundTasks,
    Depends,
    File,
    FastAPI,
    Form,
    HTTPException,
    Response,
    UploadFile,
)
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from whisper_video_summarization.api.deps import get_current_user, get_db
from whisper_video_summarization.api.routes_auth import router as auth_router
from whisper_video_summarization.api.schemas import (
    TaskCreateResponse,
    TaskStatusResponse,
)
from whisper_video_summarization.utils.prometheus_multiproc import registry_for_export
from whisper_video_summarization.celery_app.app import celery_app
from whisper_video_summarization.celery_app.tasks import run_infer_audio_task
from whisper_video_summarization.db.models import (
    InferenceTask,
    TaskStatus,
    TaskType,
    User,
)


DEBUG = os.getenv("DEBUG", "0").lower() in ("1", "true", "yes")

UPLOAD_AUDIO_DIR = Path(os.getenv("UPLOAD_AUDIO_DIR", "/app/data/uploads"))

logger = logging.getLogger("app")


def _task_transcription_payload(task: InferenceTask):
    return task.result_transcription_json




# -----------------------------
# Helpers
# -----------------------------
def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.lower() in ("1", "true", "yes", "on")


def _pyannote_enabled() -> bool:
    return _env_bool(
        "PYANNOTE_ENABLED",
        default=False,
    )


def _safe_upload_name(filename: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9_.-]", "_", filename)
    return name


async def _save_upload_file(file: UploadFile, path: Path):
    async with aiofiles.open(path, "wb") as out:
        while chunk := await file.read(1024 * 1024):
            await out.write(chunk)


def _dvc_track_upload(path: str):
    try:
        track_path_in_dvc(path)
    except Exception:
        logger.exception("DVC tracking failed")


# -----------------------------
# API app
# -----------------------------
api_app = FastAPI(title="Whisper ASR API", debug=DEBUG)

api_app.include_router(auth_router, prefix="/auth", tags=["auth"])


@api_app.post("/uploads/audio", response_model=TaskCreateResponse)
async def infer_audio_upload(
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    file: UploadFile = File(...),
    force_disable_diarization: bool = Form(False),
    db: AsyncSession = Depends(get_db),
):
    UPLOAD_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    path = UPLOAD_AUDIO_DIR / _safe_upload_name(file.filename)

    # async save
    await _save_upload_file(file, path)

    logger.info("Uploaded audio saved: %s", path)

    task = InferenceTask(
        status=TaskStatus.PENDING,
        task_type=TaskType.AUDIO_UPLOAD,
        user_id=current_user.id,
        input_path=str(path),
    )

    db.add(task)
    await db.commit()
    await db.refresh(task)

    diarization_enabled = _pyannote_enabled()
    if diarization_enabled and not force_disable_diarization:
        await run_infer_audio_task.apply_async(args=[str(task.id), str(path), True], queue="asr")
    else:
        await run_infer_audio_task.apply_async(args=[str(task.id), str(path), False], queue="asr")

    return TaskCreateResponse(task_id=task.id)



@api_app.get("/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task(
    task_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(InferenceTask).where(InferenceTask.id == task_id)
    )
    task = result.scalar_one_or_none()

    if not task or task.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Task not found")

    return TaskStatusResponse(
        task_id=task.id,
        status=task.status.value,
        task_type=task.task_type.value,
        result_transcription=_task_transcription_payload(task),
        result_summary=task.result_summary,
        error_message=task.error_message,
        created_at=task.created_at,
        updated_at=task.updated_at,
    )


@api_app.get("/tasks", response_model=list[TaskStatusResponse])
async def list_tasks(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    limit: int = 50,
    offset: int = 0,
):
    result = await db.execute(
        select(InferenceTask)
        .where(InferenceTask.user_id == current_user.id)
        .order_by(InferenceTask.created_at.desc())
        .offset(offset)
        .limit(limit)
    )

    rows = result.scalars().all()

    return [
        TaskStatusResponse(
            task_id=t.id,
            status=t.status.value,
            task_type=t.task_type.value,
            result_transcription=_task_transcription_payload(t),
            result_summary=t.result_summary,
            error_message=t.error_message,
            created_at=t.created_at,
            updated_at=t.updated_at,
        )
        for t in rows
    ]



@asynccontextmanager
async def _root_lifespan(_app: FastAPI):
    async with celery_app.setup():
        yield


app = FastAPI(title="Whisper ASR", debug=DEBUG, lifespan=_root_lifespan)

_cors_origins = ["*"] if DEBUG else ["http://whisper.local"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/metrics")
def prometheus_metrics():
    return Response(
        content=generate_latest(registry_for_export()),
        media_type=CONTENT_TYPE_LATEST,
    )


app.mount("/api", api_app)