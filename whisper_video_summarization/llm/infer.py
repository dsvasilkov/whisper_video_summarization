from pathlib import Path

import os
import re
import time
from typing import Any

from openai import AsyncOpenAI

from whisper_video_summarization.utils.observability import observe_qwen_inference


def _get_vllm_openai_client() -> AsyncOpenAI:
    """
    Return OpenAI-compatible async client pointing to vLLM server.

    Controlled via env:
    - VLLM_LLM_BASE_URL (default: http://localhost:8000/v1)
    """
    base_url = os.getenv("VLLM_LLM_BASE_URL", "http://localhost:8000/v1")
    api_key = os.getenv("VLLM_OPENAI_API_KEY", "EMPTY")
    return AsyncOpenAI(api_key=api_key, base_url=base_url)


def _segments_prompt_meta(
    segments: list[Any],
) -> tuple[str, bool, str]:
    """Текст для промпта, флаг «несколько спикеров», список id для текста промпта, множество id для разбора ответа."""
    lines: list[str] = []
    speakers: set[str] = set()
    current: str | None = None
    buf: list[str] = []

    for seg in segments:
        if not isinstance(seg, dict):
            continue
        sp = str(seg.get("speaker") or "Unknown").strip() or "Unknown"
        t = str(seg.get("text") or "").strip()
        if not t:
            continue
        if sp.lower() != "unknown":
            speakers.add(sp)
        if sp == current:
            buf.append(t)
        else:
            if current is not None and buf:
                lines.append(f"{current}: {' '.join(buf)}")
            current = sp
            buf = [t]

    if current is not None and buf:
        lines.append(f"{current}: {' '.join(buf)}")

    uniq = sorted(speakers)
    return "\n".join(lines), len(uniq) > 1, ", ".join(uniq), frozenset(speakers)


async def infer(
    model_checkpoint: Path,
    texts: list[dict[str, Any]],
    model_name: str,
    model_type: str,
    max_length: int,
    device: str,
    max_new_tokens: int | None = None,
) -> list[str]:
    del model_checkpoint, max_length, device

    if model_type != "qwen":
        raise ValueError(f"Unsupported model_type: {model_type}. Expected 'qwen'.")

    client = _get_vllm_openai_client()
    gen_max_new_tokens = max_new_tokens if max_new_tokens is not None else 16384

    summaries: list[str] = []
    system_prompt = (
        "Ты академический помощник по суммаризации лекций и учебных выступлений. "
        "Отвечай только текстом в требуемом формате строк, без markdown и без пояснений. "
        "Не добавляй рассуждения и теги <redacted_thinking>. "
        "Пиши содержательно и структурно: тема лекции, ключевые тезисы, определения, аргументы, "
        "примеры, выводы и практическая ценность. Сохраняй фактическую точность, не выдумывай."
    )

    for payload in texts:
        raw_segs = payload.get("segments") if isinstance(payload, dict) else None
        segments = raw_segs if isinstance(raw_segs, list) else []
        transcription_block, has_multiple_speakers, speaker_list, _known_speakers = (
            _segments_prompt_meta(segments)
        )

        if has_multiple_speakers:
            user_prompt = (
                "Ниже транскрипт лекции/семинара: каждая строка — «идентификатор_спикера: текст реплики».\n"
                "Сначала дай общее подробное резюме лекции на русском языке. Обязательно отрази:\n"
                "- основную тему и цель лекции;\n"
                "- ключевые идеи и объяснения;\n"
                "- важные термины/понятия и их смысл;\n"
                "- примеры, выводы, рекомендации/домашние действия (если были).\n"
                "Затем для каждого спикера дай отдельное подробное резюме вклада этого спикера "
                "(что именно объяснял, какие вопросы задавал, какие выводы озвучил). "
                "Сохраняй фактическую точность, не придумывай детали и не дублируй одно и то же без нужды.\n\n"
                "Строго такой порядок и формат строк:\n"
                "ИТОГ: развёрнутое резюме лекции в целом\n"
                "(при необходимости продолжай на следующих строках; не начинай эти строки с идентификатора спикера из списка ниже)\n"
                "SPEAKER_00: развёрнутое резюме речи этого спикера\n"
                "SPEAKER_01: развёрнутое резюме речи этого спикера\n"
                "Идентификаторы спикеров — только из списка: "
                f"{speaker_list}"
                ". У каждого из них должна быть своя строка «идентификатор: текст»; длинный текст переноси на следующие строки "
                "без нового «идентификатор:» (они относятся к тому же спикеру).\n\n"
                "Транскрипт:\n"
                f"{transcription_block}\n"
            )
        else:
            user_prompt = (
                "Ниже транскрипт лекции: каждая строка — «идентификатор_спикера: текст реплики».\n"
                "Сделай развёрнутое итоговое резюме на русском языке. Обязательно включи:\n"
                "- тему и цель лекции;\n"
                "- ключевые тезисы и объяснения;\n"
                "- важные термины/понятия и их трактовку;\n"
                "- примеры, факты/цифры из исходника;\n"
                "- основные выводы и практические шаги/рекомендации (если упомянуты).\n"
                "Сохраняй фактическую точность, не придумывай детали.\n\n"
                "Формат ответа: первая строка начинается с «ИТОГ: », затем текст; при необходимости продолжай на следующих строках.\n\n"
                "Транскрипт:\n"
                f"{transcription_block}\n"
            )

        started = time.perf_counter()
        completion = await client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=gen_max_new_tokens,
            temperature=0.0,
            frequency_penalty=0.2,
            presence_penalty=0.1,
        )
        elapsed = time.perf_counter() - started
        usage = getattr(completion, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        if prompt_tokens <= 0:
            prompt_tokens = max(len(user_prompt.split()), 1)
        if completion_tokens <= 0:
            # Fallback for non-standard OpenAI-compatible servers without token usage.
            completion_tokens = max(len((completion.choices[0].message.content or "").split()), 1)
        observe_qwen_inference(
            duration_seconds=elapsed,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

        content = completion.choices[0].message.content or ""
        content = re.sub(
            r"<redacted_thinking>.*?</redacted_thinking>",
            "",
            content,
            flags=re.DOTALL,
        ).strip()

        summaries.append(content)

    return summaries
