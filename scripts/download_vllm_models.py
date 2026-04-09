#!/usr/bin/env python3
"""Скачать vLLM-модели в data/models/vllm/... только если их ещё нет.

Дальше в корне репозитория:
  dvc add data/models/vllm/openai__whisper-large-v3 data/models/vllm/unsloth__Qwen3.5-9B-GGUF data/models/vllm/Qwen__Qwen3.5-9B
  dvc push

Пример:
  uv sync --group models
  uv run python scripts/download_vllm_models.py
"""

from whisper_video_summarization.utils.hf_models import main

if __name__ == "__main__":
    main()
