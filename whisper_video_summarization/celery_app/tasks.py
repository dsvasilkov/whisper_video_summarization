"""
Consumer: обработка задач инференса из очереди.
"""
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
from whisper_video_summarization.utils.observability import start_worker_metrics_server
logger = logging.getLogger("celery.tasks")
# -----------------------------
# Utils
# -----------------------------
def _ensure_worker_metrics_started():
    try:
        start_worker_metrics_server()
    except Exception:
        logger.exception("Failed to start worker metrics server")
def _build_llm_transcription_payload(transcription: dict[str, Any]) -> dict[str, Any]:
    segments = transcription.get("segments", []) if isinstance(transcription, dict) else []
    llm_segments: list[dict[str, str]] = []
    for seg in segments:
        speaker = str(seg.get("speaker") or "Unknown").strip() or "Unknown"
        text = str(seg.get("text") or "").strip()
        if not text:
            continue
        llm_segments.append({"speaker": speaker, "text": text})
    return {"segments": llm_segments}
def _as_payload(row: InferenceTask) -> dict[str, Any]:
    raw = row.result_transcription_json
    return dict(raw) if isinstance(raw, dict) else {}
def _as_meta(payload: dict[str, Any]) -> dict[str, Any]:
    raw = payload.get("_meta", {})
    return dict(raw) if isinstance(raw, dict) else {}
def _merge_ready(meta: dict[str, Any]) -> bool:
    return (
        bool(meta.get("asr_done"))
        and bool(meta.get("diarization_ready"))
        and not bool(meta.get("merge_done"))
        and not bool(meta.get("merge_enqueued"))
    )


def _pyannote_worker_enabled() -> bool:
    # Backward compatible env names across API/worker configs.
    for key in ("PYANNOTE_ENABLED", "PYANNOTE_PIPELINE_ENABLED"):
        raw = os.getenv(key, "")
        if raw.lower() in {"1", "true", "yes", "on"}:
            return True
    return False


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
    if bool(meta.get("llm_enqueued")):
        return False
    meta["llm_enqueued"] = True
    payload["_meta"] = meta
    row.result_transcription_json = payload
    await session.commit()
    await run_infer_summary_task.apply_async(args=[task_id], queue="llm")
    logger.info("Task %s: sent to LLM queue (%s)", task_id, reason)
    return True
async def _update_task_status(
    session: AsyncSession,
    task_id: UUID,
    status: TaskStatus,
    result_transcription_json: dict[str, Any] | None = None,
    result_summary: str | None = None,
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
    if error_message is not None:
        row.error_message = error_message
    await session.commit()
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
            summary = await run_infer(transcription_json)
            await _update_task_status(
                session,
                task_uuid,
                TaskStatus.COMPLETED,
                result_summary=summary,
            )
            logger.info(f"Task {task_id} completed (summary)")
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
    path = Path(audio_path)
    if not path.is_absolute():
        path = Path("/app") / path
    await _update_task_status(session, task_uuid, TaskStatus.PROCESSING)
    # Start diarization immediately on pyannote queue.
    if enable_diarization:
        await run_prepare_diarization_task.apply_async(args=[task_id, str(path)], queue="pyannote")
    transcription = await transcribe_audio(path, diarize=False)
    text = str(transcription.get("text") or "").strip()
    if not text or not text.strip():
        raise RuntimeError("ASR returned empty text")
    await _update_task_status(
        session,
        task_uuid,
        TaskStatus.PROCESSING,
        result_transcription_json=transcription,
    )
    if enable_diarization:
        result = await session.execute(
            select(InferenceTask).where(InferenceTask.id == task_uuid).with_for_update()
        )
        row = result.scalar_one_or_none()
        if row:
            payload = _as_payload(row)
            meta = _as_meta(payload)
            meta["asr_done"] = True
            should_enqueue_merge = _merge_ready(meta)
            if should_enqueue_merge:
                meta["merge_enqueued"] = True
            payload["_meta"] = meta
            row.result_transcription_json = payload
            await session.commit()
            if should_enqueue_merge:
                await run_merge_diarization_task.apply_async(args=[task_id], queue="pyannote")
            else:
                if bool(meta.get("diarization_skipped")):
                    await _enqueue_llm_once(session, task_uuid, task_id, "diarization skipped")
    else:
        await _enqueue_llm_once(session, task_uuid, task_id, "diarization disabled")
@celery_app.task(bind=True, name="inference.run_diarization_prepare", queue="pyannote")
async def run_prepare_diarization_task(self, task_id: str, audio_path: str):
    from whisper_video_summarization.whisper.transcribe import (
        diarize_audio,
    )
    _ensure_worker_metrics_started()
    if not _pyannote_worker_enabled():
        logger.info("Task %s: pyannote disabled on this worker, skipping prepare", task_id)
        task_uuid = UUID(task_id)
        SessionLocal = get_async_session_factory()
        async with SessionLocal() as session:
            result = await session.execute(
                select(InferenceTask).where(InferenceTask.id == task_uuid).with_for_update()
            )
            row = result.scalar_one_or_none()
            if row:
                payload = _as_payload(row)
                meta = _as_meta(payload)
                meta["diarization_skipped"] = True
                payload["_meta"] = meta
                row.result_transcription_json = payload
                await session.commit()
                if bool(meta.get("asr_done")):
                    await _enqueue_llm_once(session, task_uuid, task_id, "pyannote worker disabled")
        return
    path = Path(audio_path)
    if not path.is_absolute():
        path = Path("/app") / path
    speakers = await diarize_audio(path)
    if not speakers:
        logger.info("Task %s: pyannote returned no speakers; skip prepare", task_id)
    task_uuid = UUID(task_id)
    SessionLocal = get_async_session_factory()
    async with SessionLocal() as session:
        result = await session.execute(
            select(InferenceTask).where(InferenceTask.id == task_uuid).with_for_update()
        )
        row = result.scalar_one_or_none()
        if not row:
            logger.info("Task %s: not found during diarization prepare", task_id)
            return
        payload = _as_payload(row)
        meta = _as_meta(payload)
        if speakers:
            meta["diarization_ready"] = True
            meta["diarization_speakers"] = speakers
            should_enqueue_merge = _merge_ready(meta)
            if should_enqueue_merge:
                meta["merge_enqueued"] = True
        else:
            meta["diarization_skipped"] = True
            should_enqueue_merge = False
        payload["_meta"] = meta
        row.result_transcription_json = payload
        await session.commit()
        if speakers:
            logger.info("Task %s: diarization prepared (%d speaker segments)", task_id, len(speakers))
        # ASR may already be done by this moment.
        if should_enqueue_merge:
            await run_merge_diarization_task.apply_async(args=[task_id], queue="pyannote")
        elif bool(meta.get("asr_done")) and bool(meta.get("diarization_skipped")):
            await _enqueue_llm_once(session, task_uuid, task_id, "pyannote returned no speakers")
@celery_app.task(bind=True, name="inference.run_diarization_merge", queue="pyannote")
async def run_merge_diarization_task(self, task_id: str):
    from whisper_video_summarization.whisper.transcribe import (
        _assign_segment_speakers,
        _assign_speakers,
    )
    _ensure_worker_metrics_started()
    task_uuid = UUID(task_id)
    SessionLocal = get_async_session_factory()
    async with SessionLocal() as session:
        result = await session.execute(
            select(InferenceTask).where(InferenceTask.id == task_uuid)
        )
        row = result.scalar_one_or_none()
        if not row or not isinstance(row.result_transcription_json, dict):
            logger.info("Task %s: transcription missing before diarization merge", task_id)
            return
        payload = _as_payload(row)
        meta = _as_meta(payload)
        if meta.get("merge_done"):
            return
        if not bool(meta.get("asr_done")):
            logger.info("Task %s: merge signal received before ASR done", task_id)
            return
        speakers_raw = meta.get("diarization_speakers", [])
        speakers = [s for s in speakers_raw if isinstance(s, dict)] if isinstance(speakers_raw, list) else []
        if not speakers:
            logger.info("Task %s: merge signal received before diarization ready", task_id)
            return
        raw_segments = row.result_transcription_json.get("segments", [])
        if not isinstance(raw_segments, list) or not raw_segments:
            logger.info("Task %s: no segments in transcription for diarization merge", task_id)
            return
        segments = [dict(seg) for seg in raw_segments if isinstance(seg, dict)]
        if not segments:
            logger.info("Task %s: no valid segments for diarization merge", task_id)
            return
        segments = _assign_speakers(segments, speakers)
        segments = _assign_segment_speakers(segments, speakers)
        merged = dict(payload)
        merged["segments"] = segments
        merged["format"] = "speaker_segments_v1"
        meta["merge_done"] = True
        meta["merge_enqueued"] = False
        meta.pop("diarization_speakers", None)
        merged["_meta"] = meta
        row.result_transcription_json = merged
        await session.commit()
        logger.info("Task %s: merged pyannote speakers into ASR transcription", task_id)
        await _enqueue_llm_once(session, task_uuid, task_id, "diarization merged")
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
