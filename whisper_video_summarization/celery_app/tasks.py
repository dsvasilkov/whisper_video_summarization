"""
Consumer: обработка задач инференса из очереди.
"""
import asyncio
import copy
import logging
import os
from pathlib import Path
from typing import Any
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from whisper_video_summarization.celery_app.app import celery_app
from whisper_video_summarization.db.models import TaskStatus, InferenceTask
from whisper_video_summarization.db.session import get_async_session_factory
from whisper_video_summarization.llm.qa_rag import rag_indexing_enabled
from whisper_video_summarization.utils.observability import (
    observe_inference_task_terminal,
    start_worker_metrics_server,
)
from whisper_video_summarization.utils.s3 import download_to_temp_file, parse_s3_uri
from whisper_video_summarization.utils.task_events import publish_task_event

logger = logging.getLogger("celery.tasks")

_TOPIC_GRAPH_UNSET = object()

# -----------------------------
# Utils
# -----------------------------
def _ensure_worker_metrics_started():
    try:
        start_worker_metrics_server()
    except Exception:
        logger.exception("Failed to start worker metrics server")
def _build_llm_transcription_payload(transcription: dict[str, Any]) -> dict[str, Any]:
    """Минимальный payload для LLM из ASR-результата: speaker/text + start/end (нужны графу для t0/t1).

    Тяжёлые поля (`words`, словарные подписи) отбрасываются — сообщение между Celery-воркерами идёт
    через DB, не RabbitMQ, но всё равно бережём байты. ``start``/``end`` — это два числа, без них
    рассыпается timeline mind map (узлы графа теряют t0/t1)."""
    segments = transcription.get("segments", []) if isinstance(transcription, dict) else []
    llm_segments: list[dict[str, Any]] = []
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        speaker = str(seg.get("speaker") or "Unknown").strip() or "Unknown"
        text = str(seg.get("text") or "").strip()
        if not text:
            continue
        item: dict[str, Any] = {"speaker": speaker, "text": text}
        start = seg.get("start")
        if start is None:
            start = seg.get("start_time")
        end = seg.get("end")
        if end is None:
            end = seg.get("end_time")
        try:
            if start is not None:
                item["start"] = float(start)
        except (TypeError, ValueError):
            pass
        try:
            if end is not None:
                item["end"] = float(end)
        except (TypeError, ValueError):
            pass
        llm_segments.append(item)
    return {"segments": llm_segments}
def _as_payload(row: InferenceTask) -> dict[str, Any]:
    raw = row.result_transcription_json
    return dict(raw) if isinstance(raw, dict) else {}
def _as_meta(payload: dict[str, Any]) -> dict[str, Any]:
    raw = payload.get("_meta", {})
    return dict(raw) if isinstance(raw, dict) else {}
def _summary_task_timeout_seconds() -> float:
    raw = os.getenv("SUMMARY_TASK_TIMEOUT_SECONDS", "").strip()
    if not raw:
        return 3600.0
    try:
        return max(10.0, float(raw))
    except ValueError:
        return 3600.0


async def _enqueue_llm_once(
    session: AsyncSession,
    task_uuid: UUID,
    task_id: str,
    reason: str,
) -> bool:
    result = await session.execute(
        select(InferenceTask).where(InferenceTask.id == task_uuid).with_for_update()
    )
    row = result.scalar_one_or_none()
    if not row:
        return False
    payload = _as_payload(row)
    meta = _as_meta(payload)
    llm_done = bool(meta.get("llm_enqueued"))
    rag_done = bool(meta.get("rag_index_enqueued"))
    want_rag = rag_indexing_enabled()
    if llm_done and (rag_done or not want_rag):
        return False
    changed = False
    if not llm_done:
        meta["llm_enqueued"] = True
        changed = True
    if want_rag and not rag_done:
        meta["rag_index_enqueued"] = True
        changed = True
    if not changed:
        return False
    payload["_meta"] = meta
    row.result_transcription_json = payload
    await session.commit()
    if not llm_done:
        await run_infer_summary_task.apply_async(args=[task_id], queue="llm")
        logger.info("Task %s: sent to LLM queue (%s)", task_id, reason)
    if want_rag and not rag_done:
        from whisper_video_summarization.celery_app.tasks_rag import rag_index_transcript_task

        await rag_index_transcript_task.apply_async(args=[task_id], queue="rag")
        logger.info("Task %s: sent to RAG index queue (%s)", task_id, reason)
    return True
async def _update_task_status(
    session: AsyncSession,
    task_id: UUID,
    status: TaskStatus,
    result_transcription_json: dict[str, Any] | None = None,
    result_summary: str | None = None,
    result_topic_graph: Any = _TOPIC_GRAPH_UNSET,
    error_message: str | None = None,
):
    result = await session.execute(
        select(InferenceTask).where(InferenceTask.id == task_id)
    )
    row = result.scalar_one_or_none()
    if not row:
        logger.error(f"Task {task_id} not found in DB")
        return
    row.status = status
    if result_transcription_json is not None:
        row.result_transcription_json = result_transcription_json
    if result_summary is not None:
        row.result_summary = result_summary
    if result_topic_graph is not _TOPIC_GRAPH_UNSET:
        row.result_topic_graph = result_topic_graph
    if error_message is not None:
        row.error_message = error_message
    await session.commit()
    await session.refresh(row)

    try:
        observe_inference_task_terminal(row)
    except Exception:
        logger.exception("Failed to observe terminal task metric for %s", task_id)

    await publish_task_event(
        str(task_id),
        {
            "task_id": str(task_id),
            "status": row.status.value if hasattr(row.status, "value") else str(row.status),
            "task_type": row.task_type.value if hasattr(row.task_type, "value") else str(row.task_type),
            "error_message": row.error_message,
            "updated_at": row.updated_at.isoformat() if getattr(row, "updated_at", None) else None,
        },
    )
# -----------------------------
# LLM (queue = llm)
# -----------------------------
@celery_app.task(bind=True, name="inference.run_summary", queue="llm")
async def run_infer_summary_task(self, task_id: str):
    """Суммаризация по task_id: транскрипт берётся из БД, чтобы не слать большие тела в RabbitMQ
    (лимит frame_max ~128 KiB по умолчанию)."""
    from whisper_video_summarization.api.infer import run_infer
    _ensure_worker_metrics_started()
    task_uuid = UUID(task_id)
    SessionLocal = get_async_session_factory()
    async with SessionLocal() as session:
        try:
            result = await session.execute(
                select(InferenceTask).where(InferenceTask.id == task_uuid)
            )
            row = result.scalar_one_or_none()
            raw = row.result_transcription_json if row else None
            if not isinstance(raw, dict) or not raw:
                raise RuntimeError(
                    f"Task {task_id}: result_transcription_json missing in DB before summary"
                )
            transcription_json = _build_llm_transcription_payload(raw)
            logger.info("Task %s: starting LLM summary inference", task_id)
            infer_out = await asyncio.wait_for(
                run_infer(transcription_json, lecture_id=task_id),
                timeout=_summary_task_timeout_seconds(),
            )
            merged = copy.deepcopy(raw)
            meta = dict(merged.get("_meta") or {})
            tw = infer_out.get("_task_wall_seconds") or {}
            meta["task_wall_qwen_seconds"] = float(tw.get("qwen") or 0.0)
            meta["task_wall_embeddings_seconds"] = float(tw.get("embeddings") or 0.0)
            merged["_meta"] = meta
            await _update_task_status(
                session,
                task_uuid,
                TaskStatus.COMPLETED,
                result_transcription_json=merged,
                result_summary=infer_out["summary"],
                result_topic_graph=infer_out.get("topic_graph"),
            )
            logger.info(f"Task {task_id} completed (summary)")
        except asyncio.TimeoutError:
            msg = f"LLM summary timed out after {_summary_task_timeout_seconds():.0f}s"
            logger.exception("Task %s failed (summary): %s", task_id, msg)
            await _update_task_status(
                session,
                task_uuid,
                TaskStatus.FAILED,
                error_message=msg,
            )
            raise
        except Exception as e:
            logger.exception(f"Task {task_id} failed (summary): {e}")
            await _update_task_status(
                session,
                task_uuid,
                TaskStatus.FAILED,
                error_message=str(e),
            )
            raise
# -----------------------------
# ASR pipeline (queue = asr)
# -----------------------------
async def _do_audio_transcription(
    session: AsyncSession,
    task_uuid: UUID,
    task_id: str,
    audio_path: str,
    enable_diarization: bool = False,
):
    from whisper_video_summarization.whisper.transcribe import transcribe_audio
    tmp_path: Path | None = None
    try:
        # Support both legacy shared filesystem paths and S3/MinIO references (s3://bucket/key).
        if isinstance(audio_path, str) and audio_path.strip().startswith("s3://"):
            loc = parse_s3_uri(audio_path.strip())
            suffix = Path(loc.key).suffix
            tmp_path = await download_to_temp_file(bucket=loc.bucket, key=loc.key, suffix=suffix)
            path = tmp_path
        else:
            path = Path(audio_path)
            if not path.is_absolute():
                path = Path("/app") / path

        await _update_task_status(session, task_uuid, TaskStatus.PROCESSING)
        # ASR и диаризация (HTTP Ray pyannote) параллельно в transcribe_audio.
        transcription = await transcribe_audio(path, diarize=enable_diarization)
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                logger.warning("Failed to remove temp audio file: %s", tmp_path)
    text = str(transcription.get("text") or "").strip()
    if not text or not text.strip():
        raise RuntimeError("ASR returned empty text")
    await _update_task_status(
        session,
        task_uuid,
        TaskStatus.PROCESSING,
        result_transcription_json=transcription,
    )
    meta_hint = (
        transcription.get("_meta", {}) if isinstance(transcription, dict) else {}
    )
    if enable_diarization and meta_hint.get("diarization_skipped"):
        enqueue_reason = "transcription complete (diarization skipped)"
    elif enable_diarization:
        enqueue_reason = "transcription complete"
    else:
        enqueue_reason = "diarization disabled"
    await _enqueue_llm_once(session, task_uuid, task_id, enqueue_reason)
# -----------------------------
# ASR task (queue = asr)
# -----------------------------
@celery_app.task(bind=True, name="inference.run_audio", queue="asr")
async def run_infer_audio_task(
    self,
    task_id: str,
    audio_path: str,
    enable_diarization: bool = False,
):
    _ensure_worker_metrics_started()
    task_uuid = UUID(task_id)
    SessionLocal = get_async_session_factory()
    async with SessionLocal() as session:
        try:
            await _do_audio_transcription(
                session,
                task_uuid,
                task_id,
                audio_path,
                enable_diarization=enable_diarization,
            )
        except Exception as e:
            logger.exception(f"Task {task_id} failed: {e}")
            await _update_task_status(
                session,
                task_uuid,
                TaskStatus.FAILED,
                error_message=str(e),
            )
            raise
