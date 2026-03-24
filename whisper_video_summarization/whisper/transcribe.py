import os
from pathlib import Path

import torch
import whisper

from whisper_video_summarization.utils.dvc import get_whisper_model_dir


def transcribe_video(video_path: Path, language: str = "ru") -> tuple[str, list[dict]]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fp16 = device == "cuda"

    # Использовать почти всю видеопамять одного процесса (0.0–1.0). По умолчанию ~95 %.
    if device == "cuda":
        frac = float(os.getenv("WHISPER_GPU_MEMORY_FRACTION", "0.95"))
        try:
            torch.cuda.set_per_process_memory_fraction(min(1.0, max(0.1, frac)))
        except Exception:
            pass

    whisper_model_dir = get_whisper_model_dir()
    model = whisper.load_model(
        "large-v3", download_root=str(whisper_model_dir), device=device
    )

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    result = model.transcribe(
        str(video_path), language=language, fp16=fp16, verbose=True
    )

    return result["text"]
