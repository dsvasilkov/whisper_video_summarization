from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset


class GazetaDataset(Dataset):

    def __init__(self, path: str, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        path_obj = Path(path)
        if path_obj.suffix == ".jsonl":
            self.df = pd.read_json(path_obj, lines=True)
        else:
            self.df = pd.read_csv(path_obj)
        if "text" not in self.df.columns or "summary" not in self.df.columns:
            raise ValueError(
                "Dataset must have 'text' and 'summary' columns. "
                f"Got: {list(self.df.columns)}"
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row["text"])
        summary = str(row["summary"])

        source = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        target = self.tokenizer(
            summary,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        labels = target["input_ids"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": source["input_ids"].squeeze(0),
            "attention_mask": source["attention_mask"].squeeze(0),
            "labels": labels,
        }
