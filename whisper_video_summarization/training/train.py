from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import T5Tokenizer

from whisper_video_summarization.data.gazeta_dataset import GazetaDataset
from whisper_video_summarization.models.summarizer import T5Summarizer
from whisper_video_summarization.utils.logging import get_mlflow_logger


def train(cfg, dataset_path: str | None = None):
    tokenizer = T5Tokenizer.from_pretrained(cfg.model.name)

    if dataset_path:
        dataset_path_obj = Path(dataset_path)
        if dataset_path_obj.suffix == ".jsonl":
            df = pd.read_json(dataset_path_obj, lines=True)
        else:
            df = pd.read_csv(dataset_path_obj)

        train_df, val_df = train_test_split(df, test_size=0.1, random_state=cfg.train.seed)

        data_dir = Path(cfg.paths.gazeta_data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)

        train_path = data_dir / "train_split.csv"
        val_path = data_dir / "val_split.csv"

        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
    else:
        raise ValueError(
            "dataset_path is required. Please upload a dataset through Streamlit or provide a path."
        )

    train_dataset = GazetaDataset(str(train_path), tokenizer, cfg.data.max_length)
    val_dataset = GazetaDataset(str(val_path), tokenizer, cfg.data.max_length)

    model = T5Summarizer(
        model_name=cfg.model.name,
        learning_rate=cfg.model.lr,
        max_length=cfg.data.max_length,
    )

    checkpoint_dir = Path(cfg.paths.summarizer_checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="best",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=False,
    )

    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs,
        logger=get_mlflow_logger(cfg),
        callbacks=[checkpoint_callback],
    )

    trainer.fit(
        model,
        DataLoader(train_dataset, batch_size=cfg.train.batch_size),
        DataLoader(val_dataset, batch_size=cfg.train.batch_size),
    )


if __name__ == "__main__":
    from hydra import compose, initialize

    with initialize(config_path="../../configs", version_base=None):
        cfg = compose(config_name="train")
    train(cfg)
