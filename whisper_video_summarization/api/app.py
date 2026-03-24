import logging
import os
import re
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import BackgroundTasks, Depends, File, FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from whisper_video_summarization.api.deps import get_db
from whisper_video_summarization.api.schemas import (
    InferRequest,
    InferVideoRequest,
    TaskCreateResponse,
    TaskStatusResponse,
    TrainRequest,
)
from whisper_video_summarization.api.train import run_training
from whisper_video_summarization.celery_app.tasks import (
    run_infer_text_task,
    run_infer_video_task,
    run_infer_video_upload_task,
)
from whisper_video_summarization.db.models import InferenceTask, TaskStatus, TaskType
from whisper_video_summarization.utils.dvc import add_whisper_to_dvc, dvc_pull, track_path_in_dvc

DEBUG = os.getenv("DEBUG", "0").lower() in ("1", "true", "yes")

# Общий том k8s/docker: ./data → /app/data; загрузки попадают в DVC (track_path_in_dvc)
UPLOAD_VIDEO_DIR = Path(os.getenv("UPLOAD_VIDEO_DIR", "/app/data/uploads"))
UPLOAD_DATASET_DIR = Path(os.getenv("UPLOAD_DATASET_DIR", "/app/data/datasets"))

logger = logging.getLogger("app")


def _safe_upload_name(original: str | None) -> str:
    stem = Path(original or "upload").stem
    suffix = Path(original or "").suffix or ".bin"
    safe = re.sub(r"[^\w\-_.]", "_", stem)[:120]
    return f"{uuid4().hex}_{safe}{suffix}"


def _dvc_track_upload(path_str: str) -> None:
    try:
        track_path_in_dvc(Path(path_str))
    except Exception:
        logger.exception("DVC track failed for %s", path_str)

# Подприложение с роутами без префикса; при монтировании на /api пути станут /api/train, /api/infer/...
api_app = FastAPI(title="Whisper Video Summarization API", debug=DEBUG)


@api_app.post("/train")
def train(request: TrainRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(run_training, request.config_path, request.dataset_path)
    return {"status": "training started"}


# ---------- Producer: постановка задач инференса в очередь ----------

@api_app.post("/infer", response_model=TaskCreateResponse)
def infer_text(request: InferRequest, db: Session = Depends(get_db)):
    """Поставить в очередь задачу суммаризации текста. Результат — через GET /tasks/{task_id}."""
    task = InferenceTask(
        status=TaskStatus.PENDING,
        task_type=TaskType.TEXT,
        input_text=request.text,
    )
    db.add(task)
    db.commit()
    db.refresh(task)
    run_infer_text_task.delay(str(task.id), request.text)
    return TaskCreateResponse(task_id=task.id)


@api_app.post("/infer/video", response_model=TaskCreateResponse)
def infer_video(request: InferVideoRequest, db: Session = Depends(get_db)):
    """Поставить в очередь задачу транскрипции и суммаризации по пути к файлу."""
    video_path = Path(request.path)
    if not video_path.is_absolute():
        video_path = Path("/app") / video_path
    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"Video file not found: {video_path}")
    task = InferenceTask(
        status=TaskStatus.PENDING,
        task_type=TaskType.VIDEO,
        input_path=str(video_path),
    )
    db.add(task)
    db.commit()
    db.refresh(task)
    run_infer_video_task.delay(str(task.id), str(video_path))
    return TaskCreateResponse(task_id=task.id)


@api_app.post("/infer/video/upload", response_model=TaskCreateResponse)
async def infer_video_upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Загрузить файл, поставить в очередь задачу транскрипции и суммаризации."""
    UPLOAD_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    path = UPLOAD_VIDEO_DIR / _safe_upload_name(file.filename)
    content = await file.read()
    path.write_bytes(content)
    logger.info("Uploaded file saved: %s", path)
    background_tasks.add_task(_dvc_track_upload, str(path))
    task = InferenceTask(
        status=TaskStatus.PENDING,
        task_type=TaskType.VIDEO_UPLOAD,
        input_path=str(path),
    )
    db.add(task)
    db.commit()
    db.refresh(task)
    run_infer_video_upload_task.delay(str(task.id), str(path))
    return TaskCreateResponse(task_id=task.id)


# ---------- Роуты статусов из БД (backend опрашивает сам) ----------

@api_app.get("/tasks/{task_id}", response_model=TaskStatusResponse)
def get_task(task_id: UUID, db: Session = Depends(get_db)):
    """Получить статус и результат задачи по ID."""
    task = db.query(InferenceTask).filter(InferenceTask.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return TaskStatusResponse(
        task_id=task.id,
        status=task.status.value,
        task_type=task.task_type.value,
        result_transcription=task.result_transcription,
        result_summary=task.result_summary,
        error_message=task.error_message,
        created_at=task.created_at,
        updated_at=task.updated_at,
    )


@api_app.get("/tasks", response_model=list[TaskStatusResponse])
def list_tasks(
    db: Session = Depends(get_db),
    limit: int = 50,
    offset: int = 0,
):
    """Список задач с пагинацией (последние сначала)."""
    rows = (
        db.query(InferenceTask)
        .order_by(InferenceTask.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    return [
        TaskStatusResponse(
            task_id=t.id,
            status=t.status.value,
            task_type=t.task_type.value,
            result_transcription=t.result_transcription,
            result_summary=t.result_summary,
            error_message=t.error_message,
            created_at=t.created_at,
            updated_at=t.updated_at,
        )
        for t in rows
    ]


@api_app.post("/upload/dataset")
async def upload_dataset(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """Сохраняет загруженный датасет (Gazeta) и возвращает путь для POST /train."""
    UPLOAD_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    path = UPLOAD_DATASET_DIR / _safe_upload_name(file.filename or "dataset.jsonl")
    content = await file.read()
    path.write_bytes(content)
    background_tasks.add_task(_dvc_track_upload, str(path))
    return {"path": str(path)}


# ---------- Главное приложение: CORS, startup, монтирование API под /api ----------

def _startup():
    """Pull models, data, and configs from DVC; создаём таблицы БД."""
    add_whisper_to_dvc()
    dvc_pull()
    from whisper_video_summarization.db import init_db
    init_db()


_cors_origins = (
    ["*"] if DEBUG
    else [
        "http://localhost",
        "http://127.0.0.1",
        "https://localhost",
        "https://127.0.0.1",
        "http://localhost:5173",
        "http://localhost:3000",
        "http://whisper.local",
        "https://whisper.local",
        "http://whisper.local/",
        "https://whisper.local/",
    ]
)

app = FastAPI(title="Whisper Video Summarization", debug=DEBUG)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_origin_regex=r"^https?://(localhost(:\d+)?|127\.0\.0\.1(:\d+)?|whisper\.local)/?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)
app.on_event("startup")(_startup)
app.mount("/api", api_app)
