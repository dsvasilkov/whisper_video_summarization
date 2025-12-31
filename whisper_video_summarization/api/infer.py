from pathlib import Path

import torch
from hydra import compose, initialize

from whisper_video_summarization.training.infer import infer


def run_infer(text: str) -> str:
    with initialize(config_path="../../configs", version_base=None):
        cfg = compose(config_name="train")

    model_path = Path(cfg.paths.summarizer_checkpoint_file)
    summaries = infer(
        model_checkpoint=model_path,
        texts=[text],
        model_name=cfg.model.name,
        max_length=16384,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    return summaries[0]
