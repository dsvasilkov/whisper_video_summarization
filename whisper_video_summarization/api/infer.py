from pathlib import Path
from typing import Any

from hydra import compose, initialize

from whisper_video_summarization.llm.infer import infer


async def run_infer(transcription_json: dict[str, Any]) -> str:
    with initialize(config_path="../../configs", version_base=None):
        cfg = compose(config_name="infer")

    model_path = Path(cfg.paths.summarizer_checkpoint_file)
    summaries = await infer(
        model_checkpoint=model_path,
        texts=[transcription_json],
        model_name=cfg.model.name,
        model_type=getattr(cfg.model, "type", "qwen"),
        max_length=16384,
        device="cuda",
        max_new_tokens=getattr(cfg.model, "max_new_tokens", None),
    )
    return summaries[0]
