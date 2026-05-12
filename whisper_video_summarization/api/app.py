import logging
import os
import re
from contextlib import asynccontextmanager
from uuid import UUID
from urllib.parse import unquote_plus
import asyncio
import json
from pathlib import Path
from typing import Any

from aio_celery.exceptions import TimeoutError as CeleryTimeoutError

from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Response,
    Request,
)
from fastapi.responses import StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select
from sqlalchemy.orm import load_only
from sqlalchemy.ext.asyncio import AsyncSession

from whisper_video_summarization.api.deps import get_current_user, get_db
from whisper_video_summarization.api.routes_auth import router as auth_router
from whisper_video_summarization.api.schemas import (
    ChunkEmbeddingItem,
    PresignAudioUploadRequest,
    PresignAudioUploadResponse,
    TaskChunkEmbeddingsResponse,
    TaskCreateResponse,
    TaskQuestionAnswerResponse,
    TaskQuestionRequest,
    TaskStatusResponse,
)
from whisper_video_summarization.utils.prometheus_multiproc import registry_for_export
from whisper_video_summarization.utils.observability import register_api_http_metrics_middleware
from whisper_video_summarization.celery_app.app import celery_app
from whisper_video_summarization.celery_app.tasks import run_infer_audio_task
from whisper_video_summarization.utils.s3 import (
    build_s3_uri,
    ensure_bucket_exists,
    ensure_bucket_event_notifications,
    presign_put_object_url,
    s3_bucket,
)
from whisper_video_summarization.celery_app.tasks_rag import (
    rag_answer_question_task,
    rag_chunk_embeddings_task,
)
from whisper_video_summarization.db.models import (
    InferenceTask,
    TaskStatus,
    TaskType,
    User,
)
from whisper_video_summarization.utils.task_events import (
    task_events_channel,
    task_events_redis_client,
)


DEBUG = os.getenv("DEBUG", "0").lower() in ("1", "true", "yes")

logger = logging.getLogger(__name__)


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


def _minio_webhook_token() -> str | None:
    raw = os.getenv("MINIO_WEBHOOK_TOKEN", "").strip()
    return raw or None


def _validate_minio_event_auth(request: Request) -> None:
    token = _minio_webhook_token()
    if not token:
        return
    # MinIO webhook can send Authorization header; accept both exact token and Bearer token.
    auth = (request.headers.get("authorization") or "").strip()
    if auth == token or auth == f"Bearer {token}":
        return
    # Also accept a dedicated header if configured externally.
    alt = (request.headers.get("x-minio-webhook-token") or "").strip()
    if alt == token:
        return
    raise HTTPException(status_code=401, detail="Invalid MinIO webhook token")


async def _enqueue_asr_once_for_task(
    db: AsyncSession,
    task: InferenceTask,
    *,
    enable_diarization: bool,
) -> bool:
    # Idempotency: mark in result_transcription_json._meta before enqueue.
    payload = task.result_transcription_json if isinstance(task.result_transcription_json, dict) else {}
    meta = payload.get("_meta") if isinstance(payload.get("_meta"), dict) else {}
    if meta.get("asr_enqueued"):
        return False
    meta = dict(meta)
    meta["asr_enqueued"] = True
    payload = dict(payload)
    payload["_meta"] = meta
    task.result_transcription_json = payload
    db.add(task)
    await db.commit()

    await run_infer_audio_task.apply_async(
        args=[str(task.id), str(task.input_path), bool(enable_diarization)],
        queue="asr",
    )
    return True


# -----------------------------
# API app
# -----------------------------
api_app = FastAPI(title="Whisper ASR API", debug=DEBUG)
register_api_http_metrics_middleware(api_app)

api_app.include_router(auth_router, prefix="/auth", tags=["auth"])


@api_app.post("/uploads/audio/presign", response_model=PresignAudioUploadResponse)
async def presign_audio_upload(
    body: PresignAudioUploadRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    safe_name = _safe_upload_name(body.filename or "audio.wav")
    content_type = (body.content_type or "").strip() or "application/octet-stream"

    bucket = s3_bucket()
    await ensure_bucket_exists(bucket)
    await ensure_bucket_event_notifications(bucket)

    task = InferenceTask(
        status=TaskStatus.PENDING,
        task_type=TaskType.AUDIO_UPLOAD,
        user_id=current_user.id,
        input_path=None,
    )
    db.add(task)
    await db.commit()
    await db.refresh(task)

    sha = str(body.sha256 or "").strip().lower()
    ext = Path(safe_name).suffix
    if not ext:
        ext = ".wav"
    key = f"objects/{sha}{ext}"
    s3_uri = build_s3_uri(bucket, key)
    upload_url = await presign_put_object_url(
        bucket=bucket,
        key=key,
        content_type=content_type,
        expires_seconds=int(os.getenv("S3_PRESIGN_EXPIRES_SECONDS", "900")),
    )

    task.input_path = s3_uri
    db.add(task)
    await db.commit()

    # For presigned PUT, the client must send the same Content-Type used for signing.
    return PresignAudioUploadResponse(
        task_id=task.id,
        upload_url=upload_url,
        required_headers={"Content-Type": content_type},
        s3_uri=s3_uri,
    )


@api_app.post("/minio/events")
async def minio_events(request: Request, db: AsyncSession = Depends(get_db)):
    _validate_minio_event_auth(request)
    body = await request.json()
    records = body.get("Records") if isinstance(body, dict) else None
    if not isinstance(records, list):
        raise HTTPException(status_code=400, detail="Invalid event payload")

    diarization_enabled = _pyannote_enabled()

    enqueued = 0
    for rec in records:
        if not isinstance(rec, dict):
            continue
        s3 = rec.get("s3")
        if not isinstance(s3, dict):
            continue
        bucket = s3.get("bucket", {})
        obj = s3.get("object", {})
        if not isinstance(bucket, dict) or not isinstance(obj, dict):
            continue
        bucket_name = (bucket.get("name") or "").strip()
        key_raw = (obj.get("key") or "").strip()
        if not bucket_name or not key_raw:
            continue
        key = unquote_plus(key_raw)
        s3_uri = build_s3_uri(bucket_name, key)

        result = await db.execute(
            select(InferenceTask).where(
                InferenceTask.input_path == s3_uri,
                InferenceTask.status == TaskStatus.PENDING,
            )
        )
        tasks = result.scalars().all()
        if not tasks:
            logger.warning("MinIO event for unknown object: %s", s3_uri)
            continue

        for task in tasks:
            try:
                if await _enqueue_asr_once_for_task(
                    db,
                    task,
                    enable_diarization=bool(diarization_enabled),
                ):
                    enqueued += 1
            except Exception:
                logger.exception("Failed to enqueue ASR for %s (task=%s)", s3_uri, task.id)
                continue

    return {"ok": True, "enqueued": enqueued}



def _celery_backend_configured() -> bool:
    return bool(os.getenv("CELERY_RESULT_BACKEND", "").strip())


def _rag_async_timeout_seconds(env_name: str, default: float) -> float:
    raw = os.getenv(env_name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@api_app.post(
    "/tasks/{task_id}/qa",
    response_model=TaskQuestionAnswerResponse,
)
async def task_ask_question(
    task_id: UUID,
    body: TaskQuestionRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if not _celery_backend_configured():
        raise HTTPException(
            status_code=503,
            detail="CELERY_RESULT_BACKEND is not configured; cannot await RAG task result.",
        )
    result = await db.execute(
        select(InferenceTask).where(InferenceTask.id == task_id)
    )
    task = result.scalar_one_or_none()

    if not task or task.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Task not found")

    timeout = _rag_async_timeout_seconds("RAG_QA_ASYNC_TIMEOUT_SECONDS", 900.0)
    ar = await rag_answer_question_task.apply_async(
        args=[str(task_id), body.question],
        queue="rag",
    )
    try:
        answer = await ar.get(timeout=timeout)
    except CeleryTimeoutError as e:
        raise HTTPException(status_code=504, detail="RAG / LLM task timed out") from e
    except RuntimeError as e:
        if "No result backend" in str(e) or "result backend is not configured" in str(e).lower():
            raise HTTPException(status_code=503, detail=str(e)) from e
        raise
    if answer is None:
        raise HTTPException(status_code=500, detail="Empty answer from worker")
    return TaskQuestionAnswerResponse(answer=str(answer))


@api_app.get(
    "/tasks/{task_id}/chunks/embeddings",
    response_model=TaskChunkEmbeddingsResponse,
)
async def task_chunk_embeddings(
    task_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if not _celery_backend_configured():
        raise HTTPException(
            status_code=503,
            detail="CELERY_RESULT_BACKEND is not configured; cannot await RAG task result.",
        )
    result = await db.execute(
        select(InferenceTask).where(InferenceTask.id == task_id)
    )
    task = result.scalar_one_or_none()

    if not task or task.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Task not found")

    timeout = _rag_async_timeout_seconds("RAG_EMBEDDINGS_TASK_TIMEOUT_SECONDS", 300.0)
    ar = await rag_chunk_embeddings_task.apply_async(
        args=[str(task_id)],
        queue="rag",
    )
    try:
        rows = await ar.get(timeout=timeout)
    except CeleryTimeoutError as e:
        raise HTTPException(status_code=504, detail="Embeddings task timed out") from e
    except RuntimeError as e:
        if "No result backend" in str(e) or "result backend is not configured" in str(e).lower():
            raise HTTPException(status_code=503, detail=str(e)) from e
        raise
    chunks = [
        ChunkEmbeddingItem(chunk_id=int(r["chunk_id"]), embedding=list(r["embedding"]))
        for r in (rows or [])
        if isinstance(r, dict) and isinstance(r.get("embedding"), list)
    ]
    return TaskChunkEmbeddingsResponse(chunks=chunks)


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
        result_topic_graph=task.result_topic_graph,
        error_message=task.error_message,
        created_at=task.created_at,
        updated_at=task.updated_at,
    )


@api_app.get("/tasks/{task_id}/events")
async def task_events_sse(
    task_id: UUID,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(InferenceTask).where(InferenceTask.id == task_id))
    task = result.scalar_one_or_none()
    if not task or task.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Task not found")

    redis_client = task_events_redis_client()
    if redis_client is None:
        logger.warning("SSE task=%s: Redis not configured for task events", task_id)
        raise HTTPException(
            status_code=503,
            detail="Task events Redis is not configured (set TASK_EVENTS_REDIS_URL or CELERY_RESULT_BACKEND=redis://...)",
        )

    peer = request.client.host if request.client else None
    logger.info(
        "SSE start task_id=%s user_id=%s status=%s peer=%s",
        task_id,
        current_user.id,
        task.status.value,
        peer,
    )

    async def _gen():
        initial = {
            "task_id": str(task.id),
            "status": task.status.value,
            "task_type": task.task_type.value,
            "error_message": task.error_message,
            "updated_at": task.updated_at.isoformat() if task.updated_at else None,
        }
        yield f"event: task\ndata: {json.dumps(initial, ensure_ascii=False)}\n\n"

        if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
            logger.info("SSE end task=%s (terminal snapshot only)", task_id)
            try:
                await redis_client.aclose()
            except Exception:
                pass
            return

        pubsub = redis_client.pubsub()
        try:
            await pubsub.subscribe(task_events_channel(str(task_id)))
            # Используем listen() (parse_response(block=True)), а не get_message(timeout=…):
            # иначе в redis-py async часто не приходят сообщения pub/sub до разрыва соединения.
            listen_iter = pubsub.listen().__aiter__()
            keepalive_s = float(os.getenv("SSE_KEEPALIVE_SECONDS", "15") or "15")
            if keepalive_s < 5:
                keepalive_s = 5.0
            while True:
                if await request.is_disconnected():
                    logger.info("SSE disconnect task=%s (client)", task_id)
                    break
                try:
                    msg = await asyncio.wait_for(listen_iter.__anext__(), timeout=keepalive_s)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                    continue
                except StopAsyncIteration:
                    logger.info("SSE listen ended task=%s", task_id)
                    break
                except Exception:
                    logger.exception("SSE redis listen task=%s", task_id)
                    break
                if not msg or msg.get("type") != "message":
                    continue
                data = msg.get("data")
                try:
                    if isinstance(data, bytes):
                        data = data.decode("utf-8", "replace")
                    payload: dict[str, Any] | None = None
                    if isinstance(data, str):
                        payload = json.loads(data)
                    elif isinstance(data, dict):
                        payload = data
                    if isinstance(payload, dict):
                        st = payload.get("status")
                        logger.debug("SSE forward task=%s status=%s", task_id, st)
                        data_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
                        yield f"event: task\ndata: {data_json}\n\n"
                        if st in ("completed", "failed"):
                            logger.info("SSE end task=%s terminal_status=%s", task_id, st)
                            break
                    elif isinstance(data, str) and data.strip():
                        yield f"event: task\ndata: {data.strip()}\n\n"
                except Exception:
                    logger.exception("SSE invalid payload task=%s", task_id)
        finally:
            try:
                await pubsub.unsubscribe(task_events_channel(str(task_id)))
            except Exception:
                pass
            try:
                await pubsub.aclose()
            except Exception:
                pass
            try:
                await redis_client.aclose()
            except Exception:
                pass

    return StreamingResponse(
        _gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@api_app.get("/tasks", response_model=list[TaskStatusResponse])
async def list_tasks(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    limit: int = 50,
    offset: int = 0,
    include_results: bool = False,
):
    q = (
        select(InferenceTask)
        .where(InferenceTask.user_id == current_user.id)
        .order_by(InferenceTask.created_at.desc())
        .offset(offset)
        .limit(limit)
    )
    if not include_results:
        q = q.options(
            load_only(
                InferenceTask.id,
                InferenceTask.status,
                InferenceTask.task_type,
                InferenceTask.error_message,
                InferenceTask.created_at,
                InferenceTask.updated_at,
            )
        )
    result = await db.execute(q)

    rows = result.scalars().all()

    return [
        TaskStatusResponse(
            task_id=t.id,
            status=t.status.value,
            task_type=t.task_type.value,
            result_transcription=_task_transcription_payload(t) if include_results else None,
            result_summary=t.result_summary if include_results else None,
            result_topic_graph=t.result_topic_graph if include_results else None,
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