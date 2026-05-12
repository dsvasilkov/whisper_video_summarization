import os
from pathlib import Path
from typing import Any

from hydra import compose, initialize

from whisper_video_summarization.llm.infer import infer


def get_inference_model_name() -> str:
    with initialize(config_path="../../configs", version_base=None):
        cfg = compose(config_name="infer")
    return str(cfg.model.name)


def get_inference_model_context_max() -> int:
    with initialize(config_path="../../configs", version_base=None):
        cfg = compose(config_name="infer")
    context_max = int(os.getenv("LLM_CONTEXT_MAX_LENGTH", "8192"))
    cfg_max = int(getattr(cfg.model, "context_max_length", context_max))
    return max(context_max, cfg_max)


async def run_infer(
    transcription_json: dict[str, Any],
    lecture_id: str | None = None,
) -> dict[str, Any]:
    """Суммаризация: ``summary`` (текст), ``topic_graph`` (``nodes`` + ``links`` для mind map или None)."""
    with initialize(config_path="../../configs", version_base=None):
        cfg = compose(config_name="infer")

    model_path = Path(cfg.paths.summarizer_checkpoint_file)
    context_max = int(os.getenv("LLM_CONTEXT_MAX_LENGTH", "8192"))
    cfg_max = int(getattr(cfg.model, "context_max_length", context_max))
    max_len = max(context_max, cfg_max)

    items = await infer(
        model_checkpoint=model_path,
        texts=[transcription_json],
        model_name=cfg.model.name,
        model_type=getattr(cfg.model, "type", "qwen"),
        max_length=max_len,
        device="cuda",
        max_new_tokens=getattr(cfg.model, "max_new_tokens", None),
        lecture_id=lecture_id,
    )
    first = dict(items[0])
    wall = first.pop("_task_wall_seconds", None) or {}
    return {
        "summary": first["summary"],
        "topic_graph": first.get("topic_graph"),
        "_task_wall_seconds": wall,
    }
