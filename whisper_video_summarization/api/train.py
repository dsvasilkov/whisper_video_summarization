from typing import Optional

from whisper_video_summarization.training.train import train
from hydra import compose, initialize


def run_training(config_path: str, dataset_path: Optional[str] = None) -> None:
    with initialize(config_path="../../configs", version_base=None):
        cfg = compose(config_name="train")
    train(cfg, dataset_path)
    return {"status": "training started"}
