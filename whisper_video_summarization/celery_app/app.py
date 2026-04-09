import os

from aio_celery import Celery

BROKER_URL = os.getenv(
    "CELERY_BROKER_URL",
    "amqp://guest:guest@localhost:5672//",
)

celery_app = Celery()
celery_app.conf.update(
    broker_url=BROKER_URL
)

# Import side-effect: register task functions on app instance.
import whisper_video_summarization.celery_app.tasks  # noqa: E402,F401
