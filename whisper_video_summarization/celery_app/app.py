import os

from celery import Celery

BROKER_URL = os.getenv(
    "CELERY_BROKER_URL",
    "amqp://guest:guest@localhost:5672//",
)

celery_app = Celery(
    "whisper_video_summarization",
    broker=BROKER_URL,
    backend=None,
    include=["whisper_video_summarization.celery_app.tasks"],
)
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_acks_late=True,
    task_track_started=True,
)
