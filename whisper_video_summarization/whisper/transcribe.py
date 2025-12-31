from pathlib import Path

import torch
import whisper

from whisper_video_summarization.utils.dvc import get_whisper_model_dir


def transcribe_video(video_path: Path, language: str = "ru") -> tuple[str, list[dict]]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fp16 = device == "cuda"

    whisper_model_dir = get_whisper_model_dir()
    model = whisper.load_model(
        "base", download_root=str(whisper_model_dir), device=device
    )

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    result = model.transcribe(
        str(video_path), language=language, fp16=fp16, verbose=True
    )

    return result["text"]
