import pytorch_lightning as pl
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

from whisper_video_summarization.utils.metrics import compute_rouge


class T5Summarizer(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        learning_rate: float,
        max_length: int = 128,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

    def training_step(self, batch, batch_idx):
        output = self.model(**batch)
        self.log("train_loss", output.loss, prog_bar=True)
        return output.loss

    def validation_step(self, batch, batch_idx):
        output = self.model(**batch)
        self.log("val_loss", output.loss, prog_bar=True)

        generated_ids = self.model.generate(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            max_length=self.hparams.max_length,
        )

        preds = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        refs = self.tokenizer.batch_decode(
            batch["labels"], skip_special_tokens=True
        )

        rouge_scores = compute_rouge(preds, refs)

        for key, value in rouge_scores.items():
            self.log(
                f"val_{key}",
                value,
                prog_bar=True,
                on_epoch=True,
            )

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), lr=self.hparams.learning_rate
        )
