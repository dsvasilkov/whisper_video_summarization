import logging
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI

from whisper_video_summarization.api.infer import run_infer
from whisper_video_summarization.api.schemas import (
    InferRequest,
    InferResponse,
    InferVideoRequest,
    TrainRequest,
)
from whisper_video_summarization.api.train import run_training
from whisper_video_summarization.utils.dvc import add_whisper_to_dvc, dvc_pull
from whisper_video_summarization.whisper.transcribe import transcribe_video

app = FastAPI(title="Whisper Video Summarization")


@app.on_event("startup")
def startup():
    """Pull models, data, and configs from DVC on startup."""
    add_whisper_to_dvc()
    dvc_pull()


@app.post("/train")
def train(request: TrainRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(run_training, request.config_path, request.dataset_path)
    return {"status": "training started"}


@app.post("/infer", response_model=InferResponse)
def infer_text(request: InferRequest):
    summary = run_infer(request.text)
    return InferResponse(summary=summary)


@app.post("/infer/video")
async def infer_video(request: InferVideoRequest):
    logger = logging.getLogger("app")
    video_path = Path(request.path)

    if not video_path.is_absolute():
        video_path = Path("/app") / video_path

    logger.info(f"Starting video transcription for: {video_path}")

    if not video_path.exists():
        logger.error(f"Video file not found: {video_path}")
        raise FileNotFoundError(f"Video file not found: {video_path}")

    try:
        text = transcribe_video(video_path)
        summary = run_infer(text)
        logger.info("Video transcription completed")
        return {
            "transcription": text,
            "summary": summary,
        }
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}", exc_info=True)
        raise
