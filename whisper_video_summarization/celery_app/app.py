import os

from aio_celery import Celery

BROKER_URL = os.getenv(
    "CELERY_BROKER_URL",
    "amqp://guest:guest@localhost:5672//",
)

celery_app = Celery()
_conf: dict = {"broker_url": BROKER_URL}
_result_backend = os.getenv("CELERY_RESULT_BACKEND", "").strip()
if _result_backend:
    _conf["result_backend"] = _result_backend
celery_app.conf.update(**_conf)

# Import side-effect: register task functions on app instance.
import whisper_video_summarization.celery_app.tasks  # noqa: E402,F401
import whisper_video_summarization.celery_app.tasks_rag  # noqa: E402,F401
