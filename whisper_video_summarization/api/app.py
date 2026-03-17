import logging
import os
from pathlib import Path
from uuid import UUID

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
from whisper_video_summarization.utils.dvc import add_whisper_to_dvc, dvc_pull

DEBUG = os.getenv("DEBUG", "0").lower() in ("1", "true", "yes")

app = FastAPI(title="Whisper Video Summarization", debug=DEBUG)
logger = logging.getLogger("app")

# CORS: в production за nginx достаточно своего origin; в DEBUG — локальные адреса
_cors_origins = (
    ["*"] if DEBUG
    else ["http://localhost", "http://127.0.0.1", "https://localhost", "https://127.0.0.1"]
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    """Pull models, data, and configs from DVC; создаём таблицы БД."""
    add_whisper_to_dvc()
    dvc_pull()
    from whisper_video_summarization.db import init_db
    init_db()


@app.post("/train")
def train(request: TrainRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(run_training, request.config_path, request.dataset_path)
    return {"status": "training started"}


# ---------- Producer: постановка задач инференса в очередь ----------

@app.post("/infer", response_model=TaskCreateResponse)
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


@app.post("/infer/video", response_model=TaskCreateResponse)
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


@app.post("/infer/video/upload", response_model=TaskCreateResponse)
async def infer_video_upload(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Загрузить файл, поставить в очередь задачу транскрипции и суммаризации."""
    tmp_dir = Path("/app/tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    path = tmp_dir / (file.filename or "upload")
    content = await file.read()
    path.write_bytes(content)
    logger.info("Uploaded file saved: %s", path)
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

@app.get("/tasks/{task_id}", response_model=TaskStatusResponse)
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


@app.get("/tasks", response_model=list[TaskStatusResponse])
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


@app.post("/upload/dataset")
async def upload_dataset(file: UploadFile = File(...)):
    """Сохраняет загруженный датасет (Gazeta) и возвращает путь для POST /train."""
    tmp_dir = Path("/app/tmp/datasets")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    path = tmp_dir / (file.filename or "dataset.jsonl")
    content = await file.read()
    path.write_bytes(content)
    return {"path": str(path)}
