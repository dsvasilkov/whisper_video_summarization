"""Загрузка Hugging Face артефактов для vLLM (data/models/vllm/...)."""

from __future__ import annotations

import logging
from pathlib import Path

from whisper_video_summarization.utils.paths import get_paths

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Снапшоты HF (Transformers), путь из paths.yaml
# https://huggingface.co/openai/whisper-large-v3
WHISPER_HF_REPO = "openai/whisper-large-v3"
VLLM_HF_SNAPSHOTS: tuple[tuple[str, str], ...] = (
    (WHISPER_HF_REPO, "vllm_whisper_model_dir"),
)

# В репозитории несколько форматов весов; vLLM достаточно model.safetensors + конфиг/токенайзер.
WHISPER_HF_IGNORE_WEIGHT_PATTERNS: tuple[str, ...] = (
    "model.fp32-*.safetensors",
    "model.safetensors.index.fp32.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.fp32.json",
    "pytorch_model.fp32-*.bin",
    "flax_model.msgpack",
)

# 8-bit GGUF: https://huggingface.co/unsloth/Qwen3.5-9B-GGUF
QWEN_GGUF_REPO = "unsloth/Qwen3.5-9B-GGUF"
QWEN_GGUF_FILENAME = "Qwen3.5-9B-Q8_0.gguf"
QWEN_PATH_KEY = "vllm_qwen_model_dir"

# Базовая HF-модель: токенайзер + config для vLLM (--tokenizer / --hf-config-path).
QWEN_HF_BASE_REPO = "Qwen/Qwen3.5-9B"
QWEN_HF_PATH_KEY = "vllm_qwen_hf_tokenizer_dir"

# Только мелкие файлы; веса не качаем (инференс идёт по GGUF).
QWEN_HF_IGNORE_WEIGHT_PATTERNS: tuple[str, ...] = (
    "*.safetensors",
    "*.safetensors.index.json",
    "*.bin",
    "*.bin.index.json",
    "*.pt",
    "*.pth",
    "*.msgpack",
    "*.h5",
    "*.onnx",
    "*.gguf",
    "*.zip",
)


def _whisper_hf_ready(model_dir: Path) -> bool:
    """Офлайн-артефакты для vLLM: model.safetensors + processor + токенайзер (без дубликатов FP32/PyTorch/Flax)."""
    if not model_dir.is_dir():
        return False
    if not (model_dir / "config.json").is_file():
        return False
    if not (model_dir / "model.safetensors").is_file():
        return False
    if not (model_dir / "preprocessor_config.json").is_file():
        return False
    if (model_dir / "tokenizer.json").is_file():
        return True
    return (
        (model_dir / "tokenizer_config.json").is_file()
        and (model_dir / "vocab.json").is_file()
        and (model_dir / "merges.txt").is_file()
    )


def _qwen_gguf_ready(model_dir: Path) -> bool:
    return (model_dir / QWEN_GGUF_FILENAME).is_file()


def _qwen_hf_tokenizer_ready(hf_dir: Path) -> bool:
    if not (hf_dir / "config.json").is_file():
        return False
    if (hf_dir / "tokenizer.json").is_file():
        return True
    return (hf_dir / "tokenizer.model").is_file()


def download_vllm_snapshots_if_missing() -> None:
    """Whisper — config + tokenizer + model.safetensors (без лишних весов); Qwen — GGUF + HF tokenizer."""
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError as e:
        raise ImportError(
            "Нужен пакет huggingface_hub: uv sync --group models"
        ) from e

    paths = get_paths()

    for repo_id, path_key in VLLM_HF_SNAPSHOTS:
        rel = getattr(paths, path_key, None)
        if not rel:
            continue
        local_dir = PROJECT_ROOT / rel
        if _whisper_hf_ready(local_dir):
            logger.info("Пропуск (уже есть): %s -> %s", repo_id, local_dir)
            continue
        local_dir.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Загрузка %s -> %s (без FP32/PyTorch/Flax дубликатов)", repo_id, local_dir)
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            ignore_patterns=list(WHISPER_HF_IGNORE_WEIGHT_PATTERNS),
        )

    rel = getattr(paths, QWEN_PATH_KEY, None)
    if rel:
        qwen_dir = PROJECT_ROOT / rel
        if _qwen_gguf_ready(qwen_dir):
            logger.info(
                "Пропуск (уже есть): %s/%s",
                qwen_dir,
                QWEN_GGUF_FILENAME,
            )
        else:
            qwen_dir.mkdir(parents=True, exist_ok=True)
            logger.info(
                "Загрузка %s/%s -> %s",
                QWEN_GGUF_REPO,
                QWEN_GGUF_FILENAME,
                qwen_dir,
            )
            hf_hub_download(
                repo_id=QWEN_GGUF_REPO,
                filename=QWEN_GGUF_FILENAME,
                local_dir=str(qwen_dir),
            )

    hf_rel = getattr(paths, QWEN_HF_PATH_KEY, None)
    if hf_rel:
        hf_dir = PROJECT_ROOT / hf_rel
        if _qwen_hf_tokenizer_ready(hf_dir):
            logger.info("Пропуск (уже есть): %s", hf_dir)
        else:
            hf_dir.parent.mkdir(parents=True, exist_ok=True)
            logger.info(
                "Загрузка токенайзера и config %s -> %s (без весов)",
                QWEN_HF_BASE_REPO,
                hf_dir,
            )
            snapshot_download(
                repo_id=QWEN_HF_BASE_REPO,
                local_dir=str(hf_dir),
                ignore_patterns=list(QWEN_HF_IGNORE_WEIGHT_PATTERNS),
            )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    download_vllm_snapshots_if_missing()
