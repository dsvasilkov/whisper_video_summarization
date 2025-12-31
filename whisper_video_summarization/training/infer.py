from pathlib import Path

import torch
from transformers import T5Tokenizer

from whisper_video_summarization.models.summarizer import T5Summarizer


def infer(
    model_checkpoint: Path,
    texts: list[str],
    model_name: str,
    max_length: int,
    device: str,
) -> list[str]:
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    if model_checkpoint.exists():
        model = T5Summarizer.load_from_checkpoint(
            checkpoint_path=str(model_checkpoint),
            model_name=model_name,
            learning_rate=0.0,
        )
    else:
        model = T5Summarizer(
            model_name=model_name,
            learning_rate=0.0,
            max_length=max_length,
        )
    model.to(device)
    model.eval()

    summaries = []
    with torch.no_grad():
        for text in texts:
            encoded = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            ).to(device)

            generated_ids = model.model.generate(
                **encoded,
                max_length=max_length,
                min_length=100,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True,
                no_repeat_ngram_size=4,
                do_sample=False,
            )
            summary = tokenizer.decode(
                generated_ids[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            summaries.append(summary)
    return summaries
