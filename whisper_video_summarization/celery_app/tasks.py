"""
Consumer: обработка задач инференса из очереди.
"""
import logging
from pathlib import Path
from uuid import UUID

from sqlalchemy.orm import Session

from whisper_video_summarization.celery_app.app import celery_app
from whisper_video_summarization.db.models import TaskStatus, TaskType, InferenceTask
from whisper_video_summarization.db.session import get_session_factory

logger = logging.getLogger("celery.tasks")


def _update_task_status(
    session: Session,
    task_id: UUID,
    status: TaskStatus,
    result_transcription: str | None = None,
    result_summary: str | None = None,
    error_message: str | None = None,
):
    row = session.query(InferenceTask).filter(InferenceTask.id == task_id).first()
    if not row:
        logger.error(f"Task {task_id} not found in DB")
        return
    row.status = status
    if result_transcription is not None:
        row.result_transcription = result_transcription
    if result_summary is not None:
        row.result_summary = result_summary
    if error_message is not None:
        row.error_message = error_message
    session.commit()


@celery_app.task(bind=True, name="inference.run_text")
def run_infer_text_task(self, task_id: str, text: str):
    from whisper_video_summarization.api.infer import run_infer

    task_uuid = UUID(task_id)
    SessionLocal = get_session_factory()
    session = SessionLocal()
    try:
        _update_task_status(session, task_uuid, TaskStatus.PROCESSING)
        summary = run_infer(text)
        _update_task_status(
            session,
            task_uuid,
            TaskStatus.COMPLETED,
            result_summary=summary,
        )
        logger.info(f"Task {task_id} completed (text)")
    except Exception as e:
        logger.exception(f"Task {task_id} failed: {e}")
        _update_task_status(
            session,
            task_uuid,
            TaskStatus.FAILED,
            error_message=str(e),
        )
        raise
    finally:
        session.close()


def _do_video_inference(session: Session, task_uuid: UUID, task_id: str, video_path: str):
    from whisper_video_summarization.api.infer import run_infer
    from whisper_video_summarization.whisper.transcribe import transcribe_video

    path = Path(video_path)
    if not path.is_absolute():
        path = Path("/app") / path
    _update_task_status(session, task_uuid, TaskStatus.PROCESSING)
    text = transcribe_video(path)
    summary = run_infer(text)
    _update_task_status(
        session,
        task_uuid,
        TaskStatus.COMPLETED,
        result_transcription=text,
        result_summary=summary,
    )
    logger.info(f"Task {task_id} completed (video)")


@celery_app.task(bind=True, name="inference.run_video")
def run_infer_video_task(self, task_id: str, video_path: str):
    task_uuid = UUID(task_id)
    SessionLocal = get_session_factory()
    session = SessionLocal()
    try:
        _do_video_inference(session, task_uuid, task_id, video_path)
    except Exception as e:
        logger.exception(f"Task {task_id} failed: {e}")
        _update_task_status(
            session,
            task_uuid,
            TaskStatus.FAILED,
            error_message=str(e),
        )
        raise
    finally:
        session.close()


@celery_app.task(bind=True, name="inference.run_video_upload")
def run_infer_video_upload_task(self, task_id: str, video_path: str):
    task_uuid = UUID(task_id)
    SessionLocal = get_session_factory()
    session = SessionLocal()
    try:
        _do_video_inference(session, task_uuid, task_id, video_path)
    except Exception as e:
        logger.exception(f"Task {task_id} failed: {e}")
        _update_task_status(
            session,
            task_uuid,
            TaskStatus.FAILED,
            error_message=str(e),
        )
        raise
    finally:
        session.close()
