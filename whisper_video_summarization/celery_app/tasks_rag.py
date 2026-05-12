"""
Задачи очереди rag: индекс Qdrant, выгрузка эмбеддингов чанков, ответы по RAG.
"""
import logging
import os
from typing import Any
from uuid import UUID

from sqlalchemy import select

from whisper_video_summarization.api.infer import get_inference_model_context_max, get_inference_model_name
from whisper_video_summarization.celery_app.app import celery_app
from whisper_video_summarization.db.models import InferenceTask
from whisper_video_summarization.db.session import get_async_session_factory
from whisper_video_summarization.llm import qa_rag
from whisper_video_summarization.llm.infer import _segments_prompt_meta, answer_question_with_rag
from whisper_video_summarization.utils.observability import start_worker_metrics_server

logger = logging.getLogger("celery.tasks_rag")


def _ensure_worker_metrics_started():
    try:
        start_worker_metrics_server()
    except Exception:
        logger.exception("Failed to start worker metrics server")


def _as_payload(row: InferenceTask) -> dict[str, Any]:
    raw = row.result_transcription_json
    return dict(raw) if isinstance(raw, dict) else {}


@celery_app.task(bind=True, name="rag.index_transcript", queue="rag")
async def rag_index_transcript_task(self, task_id: str):
    """Сырой склеенный текст после Whisper → чанки и эмбеддинги в Qdrant для lecture_id == task_id."""
    _ensure_worker_metrics_started()
    task_uuid = UUID(task_id)
    SessionLocal = get_async_session_factory()
    async with SessionLocal() as session:
        result = await session.execute(select(InferenceTask).where(InferenceTask.id == task_uuid))
        row = result.scalar_one_or_none()
        raw_payload = row.result_transcription_json if row else None
        if not isinstance(raw_payload, dict) or not raw_payload:
            raise RuntimeError(f"Task {task_id}: transcription missing before RAG index")
        segments = raw_payload.get("segments") if isinstance(raw_payload.get("segments"), list) else []
        block, _, _, _ = _segments_prompt_meta(segments)
        n = qa_rag.index_full_transcript_to_qdrant_sync(block, task_id)
        logger.info("Task %s: RAG index stored (%d chunks)", task_id, n)


@celery_app.task(bind=True, name="rag.chunk_embeddings", queue="rag")
async def rag_chunk_embeddings_task(self, task_id: str) -> list[dict[str, Any]]:
    """Возвращает эмбеддинги чанков из Qdrant (после rag.index_transcript)."""
    _ensure_worker_metrics_started()
    return qa_rag.list_chunk_embeddings_from_qdrant_sync(task_id)


@celery_app.task(bind=True, name="rag.answer_question", queue="rag")
async def rag_answer_question_task(self, task_id: str, question: str) -> str:
    """RAG + один вызов vLLM (как answer_question_with_rag)."""
    _ensure_worker_metrics_started()
    task_uuid = UUID(task_id)
    SessionLocal = get_async_session_factory()
    async with SessionLocal() as session:
        result = await session.execute(select(InferenceTask).where(InferenceTask.id == task_uuid))
        row = result.scalar_one_or_none()
        raw_payload = row.result_transcription_json if row else None
        if not isinstance(raw_payload, dict):
            return "Не удалось ответить: нет транскрипта для задачи."
        segments = raw_payload.get("segments") if isinstance(raw_payload.get("segments"), list) else []
        transcription_json = {"segments": segments}

    lecture_id = task_id
    model_name = get_inference_model_name()
    effective_context = get_inference_model_context_max()
    max_nt = os.getenv("RAG_QA_MAX_NEW_TOKENS", "").strip()
    max_new = int(max_nt) if max_nt.isdigit() else None

    return await answer_question_with_rag(
        lecture_id=lecture_id,
        transcription_json=transcription_json,
        question=question,
        model_name=model_name,
        max_length=effective_context,
        max_new_tokens=max_new,
    )
