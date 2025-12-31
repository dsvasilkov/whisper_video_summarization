from hydra import compose, initialize

from whisper_video_summarization.training.train import train


def run_training(config_path: str, dataset_path: str | None = None) -> None:
    with initialize(config_path="../../configs", version_base=None):
        cfg = compose(config_name="train")
    train(cfg, dataset_path)
    return {"status": "training started"}
