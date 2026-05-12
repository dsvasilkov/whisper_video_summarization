from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import time
import unicodedata
import threading
from pathlib import Path
from typing import Any

import nltk
from openai import AsyncOpenAI

from whisper_video_summarization.llm import qa_rag
from whisper_video_summarization.llm import hierarchy_summarize, topic_graph_mindmap, topic_labels, unit_graph
from whisper_video_summarization.utils.observability import (
    bump_infer_accum_qwen_wall_seconds,
    infer_wall_totals_snapshot,
    infer_wall_tracking_begin,
    infer_wall_tracking_end,
    observe_qwen_inference,
)

logger = logging.getLogger(__name__)

_nltk_punkt_lock = threading.Lock()
_nltk_punkt_ready = False


def _ensure_nltk_punkt() -> None:
    """Ensure Punkt is available without doing per-call downloads."""
    global _nltk_punkt_ready
    if _nltk_punkt_ready:
        return
    with _nltk_punkt_lock:
        if _nltk_punkt_ready:
            return
        try:
            # Prefer local availability; avoids network.
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            # Best-effort: download once if missing.
            nltk.download("punkt_tab", quiet=True)
        _nltk_punkt_ready = True


def _vllm_llm_timeout_seconds() -> float:
    raw = os.getenv("VLLM_LLM_TIMEOUT_SECONDS", "").strip()
    if not raw:
        return 900.0
    try:
        return max(1.0, float(raw))
    except ValueError:
        return 900.0


def _get_vllm_openai_client() -> AsyncOpenAI:
    """Клиент OpenAI API к vLLM: базовый URL и ключ из окружения (все генерации в модуле идут через него)."""
    base_url = os.getenv("VLLM_LLM_BASE_URL", "http://localhost:8000/v1")
    api_key = os.getenv("VLLM_OPENAI_API_KEY", "EMPTY")
    # Важно: без timeout `chat.completions.create` может зависнуть навсегда при проблемах vLLM/сети,
    # что приводит к redelivery сообщений RabbitMQ и "вечным" задачам.
    return AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=_vllm_llm_timeout_seconds(),
        max_retries=int(os.getenv("VLLM_LLM_MAX_RETRIES", "1")),
    )


def _segment_time_bounds(seg: dict[str, Any]) -> tuple[float | None, float | None]:
    start = seg.get("start")
    end = seg.get("end")
    if start is None:
        start = seg.get("start_time")
    if end is None:
        end = seg.get("end_time")
    try:
        start_f = float(start) if start is not None else None
    except (TypeError, ValueError):
        start_f = None
    try:
        end_f = float(end) if end is not None else None
    except (TypeError, ValueError):
        end_f = None
    return start_f, end_f


def _segments_prompt_meta_with_spans(
    segments: list[Any],
) -> tuple[str, bool, str, frozenset[str], list[tuple[int, int, float | None, float | None]]]:
    """Как `_segments_prompt_meta`, плюс интервалы символов в строке и время (сек) по каждой склеенной реплике."""
    lines_meta: list[tuple[str, float | None, float | None]] = []
    speakers: set[str] = set()
    current: str | None = None
    buf: list[str] = []
    buf_times: list[tuple[float | None, float | None]] = []

    def flush() -> None:
        nonlocal current, buf, buf_times
        if current is None or not buf:
            return
        line = f"{current}: {' '.join(buf)}"
        t0s = [x[0] for x in buf_times if x[0] is not None]
        t1s = [x[1] for x in buf_times if x[1] is not None]
        t0 = min(t0s) if t0s else None
        t1 = max(t1s) if t1s else None
        lines_meta.append((line, t0, t1))
        buf = []
        buf_times = []

    for seg in segments:
        if not isinstance(seg, dict):
            continue
        sp = str(seg.get("speaker") or "Unknown").strip() or "Unknown"
        t = str(seg.get("text") or "").strip()
        if not t:
            continue
        if sp.lower() != "unknown":
            speakers.add(sp)
        t0, t1 = _segment_time_bounds(seg)
        if sp == current:
            buf.append(t)
            buf_times.append((t0, t1))
        else:
            flush()
            current = sp
            buf = [t]
            buf_times = [(t0, t1)]

    flush()

    block = "\n".join(m[0] for m in lines_meta)
    spans: list[tuple[int, int, float | None, float | None]] = []
    off = 0
    for i, (line, t0, t1) in enumerate(lines_meta):
        start = off
        end = off + len(line)
        spans.append((start, end, t0, t1))
        off = end + (1 if i < len(lines_meta) - 1 else 0)

    uniq = sorted(speakers)
    return block, len(uniq) > 1, ", ".join(uniq), frozenset(speakers), spans


def _segments_prompt_meta(
    segments: list[Any],
) -> tuple[str, bool, str, frozenset[str]]:
    """Склеивает сегменты в многострочный транскрипт «Спикер: текст», сливая подряд идущие реплики.

    Возвращает: полный текст; флаг «несколько различных спикеров»; строка имён через запятую;
    множество спикеров (без Unknown) — для финального промпта и метаданных.
    """
    block, multi, speakers, fset, _spans = _segments_prompt_meta_with_spans(segments)
    return block, multi, speakers, fset


def _format_rag_timestamp(sec: float | None) -> str:
    if sec is None:
        return "время не указано"
    s = max(0.0, float(sec))
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    r = int(s % 60)
    if h > 0:
        return f"{h}:{m:02d}:{r:02d}"
    return f"{m}:{r:02d}"


def _time_at_char_in_transcript_spans(
    spans: list[tuple[int, int, float | None, float | None]], pos: int
) -> float | None:
    for c0, c1, t0, _t1 in spans:
        if c0 <= pos < c1:
            return t0
    return None


def _timestamp_for_rag_chunk(
    *,
    transcription_block: str,
    frag: str,
    spans: list[tuple[int, int, float | None, float | None]],
    unit_text: str,
    semantic_units: list[Any],
) -> float | None:
    raw = str(frag or "").strip()
    if not raw:
        return None
    pos = transcription_block.find(raw)
    if pos >= 0:
        t = _time_at_char_in_transcript_spans(spans, pos)
        if t is not None:
            return t
    pos2 = unit_text.find(raw)
    if pos2 < 0:
        return None
    for u in semantic_units:
        if not isinstance(u, dict):
            continue
        try:
            cs = int(u.get("char_start", 0))
            ce = int(u.get("char_end", 0))
        except (TypeError, ValueError):
            continue
        if cs <= pos2 < ce:
            t0 = u.get("t0")
            return float(t0) if t0 is not None else None
    fend = pos2 + len(raw)
    best_t: float | None = None
    best_ov = 0
    for u in semantic_units:
        if not isinstance(u, dict):
            continue
        try:
            cs = int(u.get("char_start", 0))
            ce = int(u.get("char_end", 0))
        except (TypeError, ValueError):
            continue
        ov = max(0, min(fend, ce) - max(pos2, cs))
        if ov > best_ov:
            best_ov = ov
            t0 = u.get("t0")
            best_t = float(t0) if t0 is not None else None
    return best_t


def _normalize_segments(segments: list[Any]) -> list[dict[str, Any]]:
    """Приводит сегменты к виду speaker/text/start/end (float), поддерживает start_time/end_time."""
    out: list[dict[str, Any]] = []
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        start = seg.get("start")
        end = seg.get("end")
        if start is None:
            start = seg.get("start_time")
        if end is None:
            end = seg.get("end_time")
        try:
            start_f = float(start) if start is not None else None
        except (TypeError, ValueError):
            start_f = None
        try:
            end_f = float(end) if end is not None else None
        except (TypeError, ValueError):
            end_f = None
        out.append(
            {
                "speaker": str(seg.get("speaker") or "Unknown"),
                "text": str(seg.get("text") or ""),
                "start": start_f,
                "end": end_f,
            }
        )
    return out


def _extract_completion_message_text(completion: Any) -> str:
    """Текст ответа: при vLLM + --reasoning-parser Qwen3 поле content иногда пустое (см. vLLM #17357)."""
    try:
        msg = completion.choices[0].message
    except (IndexError, AttributeError, TypeError):
        return ""
    primary = getattr(msg, "content", None)
    if isinstance(primary, str) and primary.strip():
        return primary
    secondary = getattr(msg, "reasoning_content", None)
    if isinstance(secondary, str) and secondary.strip():
        return secondary
    extra = getattr(msg, "model_extra", None)
    if isinstance(extra, dict):
        for key in ("reasoning_content", "reasoning"):
            alt = extra.get(key)
            if isinstance(alt, str) and alt.strip():
                return alt
    return primary if isinstance(primary, str) else ""


def _truncate_utf8_chars(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "…"


def _max_lecture_body_chars(
    *,
    gen_max_new_tokens: int,
    system_prompt: str,
    user_prefix_without_body: str,
) -> int:
    """Ограничение длины «Текст лекции», чтобы промпт не переполнял KV vLLM."""
    raw = os.getenv("LECTURE_SUMMARY_MAX_INPUT_CHARS", "").strip()
    if raw:
        try:
            return max(4000, int(raw))
        except ValueError:
            pass
    model_len = int(os.getenv("VLLM_MAX_MODEL_LEN", "65536"))
    prompt_tokens_budget = max(1024, model_len - gen_max_new_tokens - 384)
    rough_chars = int(prompt_tokens_budget * 2.8)
    overhead = len(system_prompt) + len(user_prefix_without_body) + 120
    return max(12_000, min(rough_chars - overhead, 220_000))


async def _chat_completion_text(
    client: AsyncOpenAI,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float = 0.0,
    frequency_penalty: float = 0.2,
    presence_penalty: float = 0.1,
) -> str:
    """Один вызов chat.completions к vLLM; длительность и контекст в observe_qwen_inference."""
    started = time.perf_counter()
    # Дублируем таймаут поверх openai-клиента: иногда зависание происходит до применения http-timeout.
    req_timeout = _vllm_llm_timeout_seconds()
    completion = await asyncio.wait_for(
        client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        ),
        timeout=req_timeout,
    )
    elapsed = time.perf_counter() - started
    bump_infer_accum_qwen_wall_seconds(elapsed)

    usage = getattr(completion, "usage", None)
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    if prompt_tokens <= 0:
        prompt_tokens = max(len(user_prompt.split()), 1)
    raw_text = _extract_completion_message_text(completion)
    if completion_tokens <= 0:
        stripped = (raw_text or "").strip()
        completion_tokens = len(stripped.split()) if stripped else 0
    observe_qwen_inference(
        duration_seconds=elapsed,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )

    return (raw_text or "").strip()


def _sentence_split(text: str) -> list[str]:
    """Sentence segmentation через NLTK Punkt + нормализация normalize_sentence()."""
    raw = str(text or "").strip()
    if not raw:
        return []

    if os.getenv("SENTENCE_SPLIT_NFKC", "false").strip().lower() in {"1", "true", "yes", "on"}:
        cleaned_text = unicodedata.normalize("NFKC", raw)
    else:
        cleaned_text = raw

    _ensure_nltk_punkt()
    try:
        sents = nltk.sent_tokenize(cleaned_text)
    except Exception:
        sents = [cleaned_text]

    out: list[str] = []
    for s in sents:
        cleaned = normalize_sentence(str(s))
        if cleaned.strip():
            out.append(cleaned)
    return out or [normalize_sentence(cleaned_text)]


def normalize_sentence(text: str) -> str:
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _l2_norm(v: list[float]) -> list[float]:
    """L2-нормализация вектора (эмбеддинги микро-единиц для косинусов соседей)."""
    n = math.sqrt(sum(x * x for x in v)) + 1e-12
    return [x / n for x in v]


def _cosine(a: list[float], b: list[float]) -> float:
    """Скалярное произведение; для нормированных векторов — косинусная близость."""
    return float(sum(x * y for x, y in zip(a, b)))


def _presegment_units(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Первая стадия чанкинга: маркеры длинных пауз между сегментами, split на предложения, разнесение времени по предложениям.

    На выходе только элементы type=text с полем line («Спикер: предложение»).
    """
    pause_sec = float(os.getenv("SUMMARIZATION_PRESEG_PAUSE_SEC", "0.8"))
    units: list[dict[str, Any]] = []
    last_end: float | None = None
    for seg in segments:
        sp = str(seg.get("speaker") or "Unknown")
        raw = str(seg.get("text") or "").strip()
        if not raw:
            continue
        s0 = seg.get("start")
        s1 = seg.get("end")
        try:
            st = float(s0) if s0 is not None else None
        except (TypeError, ValueError):
            st = None
        try:
            en = float(s1) if s1 is not None else None
        except (TypeError, ValueError):
            en = None
        if last_end is not None and st is not None and (st - last_end) >= pause_sec:
            units.append({"type": "pause", "gap_sec": float(st - last_end)})
        sents = _sentence_split(raw)
        if not sents:
            continue
        dur = (en - st) if (st is not None and en is not None and en >= st) else None
        per = (dur / len(sents)) if dur and dur > 0 else None
        for i, sent in enumerate(sents):
            a = (st + i * per) if per is not None and st is not None else st
            b = (a + per) if per is not None and a is not None else en
            units.append(
                {
                    "type": "text",
                    "line": f"{sp}: {sent}",
                    "start": a,
                    "end": b,
                }
            )
        if en is not None:
            last_end = float(en)
        elif st is not None:
            last_end = float(st)
    return [u for u in units if u.get("type") == "text"]


def _apply_duration_char_constraints(
    units: list[dict[str, Any]],
    hard_boundaries: set[int],
) -> list[str]:
    """Склеивает микро-единицы в чанки с лимитами по символам и длительности; hard_boundaries — индексы обязательных разрезов.

    Короткий хвост может дописаться к предыдущему чанку; учитывается overlap между чанками (env).
    """
    min_chars = max(400, int(os.getenv("SUMMARIZATION_MIN_CHUNK_CHARS", "1200")))
    max_chars = max(min_chars, int(os.getenv("SUMMARIZATION_MAX_CHUNK_CHARS", "12000")))
    min_sec = max(0.0, float(os.getenv("SUMMARIZATION_MIN_CHUNK_SECONDS", "60")))
    max_sec = max(min_sec, float(os.getenv("SUMMARIZATION_MAX_CHUNK_SECONDS", "300")))
    overlap_ratio = min(0.2, max(0.0, float(os.getenv("SUMMARIZATION_CHUNK_OVERLAP_RATIO", "0.12"))))

    def _stats(block: list[dict[str, Any]]) -> tuple[int, float | None]:
        text = "\n".join(x["line"] for x in block)
        t0, t1 = block[0].get("start"), block[-1].get("end")
        dur = (float(t1) - float(t0)) if (t0 is not None and t1 is not None) else None
        return len(text), dur

    chunks: list[list[dict[str, Any]]] = []
    cur: list[dict[str, Any]] = []

    for i, u in enumerate(units):
        if not cur:
            cur = [u]
            continue
        cand = cur + [u]
        c_len, c_dur = _stats(cand)
        hard = i in hard_boundaries
        over_chars = c_len >= max_chars
        over_time = c_dur is not None and c_dur >= max_sec
        p_len, _p_dur = _stats(cur)
        if hard or over_chars or over_time:
            if p_len >= min_chars or hard or over_chars or over_time:
                chunks.append(cur)
                if overlap_ratio > 0 and cur:
                    keep = max(1, int(len(cur) * overlap_ratio))
                    cur = cur[-keep:] + [u]
                else:
                    cur = [u]
            else:
                cur.append(u)
            # если после жёсткого разреза кандидат всё ещё слишком длинный, продолжим на следующей итерации
            continue
        cur.append(u)

    if cur:
        chunks.append(cur)

    out: list[str] = []
    for ch in chunks:
        s = "\n".join(x["line"] for x in ch)
        if len(s) < min_chars and out:
            out[-1] = out[-1] + "\n" + s
        else:
            out.append(s)
    if not out and units:
        return ["\n".join(x["line"] for x in units)]
    return [x for x in out if str(x).strip()]


async def _single_pass_lecture_summary(
    *,
    client: AsyncOpenAI,
    model_name: str,
    system_prompt: str,
    chunk_texts: list[str],
    transcription_block: str,
    has_multiple_speakers: bool,
    speaker_list: str,
    gen_max_new_tokens: int,
    effective_context: int,
) -> str:
    """Один вызов LLM по полному тексту (склейка чанков или сырой транскрипт) — запасной путь без графа тем."""
    body = "\n\n---\n\n".join(str(c).strip() for c in chunk_texts if str(c).strip())
    if not body.strip():
        body = transcription_block.strip()
    speaker_line = f"Спикеры: {speaker_list}\n\n" if has_multiple_speakers and speaker_list else ""
    user_prefix = (
        "Сделай развёрнутый подробный конспект/обобщение лекции по тексту ниже. Ответ должен быть достаточно подробным, "
        "чтобы по нему можно было восстановить логику изложения и основные идеи без чтения исходной транскрипции.\n"
        "Форматируй ответ в Markdown и придерживайся структуры (разделы в таком порядке):\n"
        "## 1) Коротко о лекции\n"
        "- 3–6 буллетов: тема, цель, основной тезис/интуиция, чем это полезно.\n"
        "## 2) План/карта содержания\n"
        "- 6–12 пунктов по порядку появления в лекции (что за чем идёт).\n"
        "## 3) Подробный конспект по разделам\n"
        "- Для каждого раздела: что вводят, какую проблему решают, какие шаги/аргументы делают, к чему приходят.\n"
        "- Обязательно выписывай определения простыми словами и поясняй смысл ключевых терминов.\n"
        "- Если в тексте есть примеры/аналогии/интерпретации — включай их и объясняй, что они иллюстрируют.\n"
        "- Если есть перечисления, условия, допущения, оговорки, ограничения метода — перечисли их явно.\n"
        "## 4) Связи и выводы\n"
        "- Как связаны разделы между собой; какие выводы/идеи являются центральными.\n"
        "## 5) Практические takeaways\n"
        "- Что слушатель должен унести: 5–10 прикладных выводов/навыков/инструментов (если они есть в лекции).\n\n"
        "Требования:\n"
        "- Между абзацами и блоками оставляй пустую строку.\n"
        "- Подпункты оформляй заголовками ### или маркированным списком; не начинай строки с нумерации вида «2.1.».\n"
        "- Пиши фактически точно по исходному тексту; не выдумывай факты, которых нет в лекции.\n"
        "- Если что-то сформулировано неясно в транскрипте, так и отметь кратко, не додумывая.\n"
        "- Излагай обобщённо и словесно: без формул, математической символики и пошаговых выкладок; пересказывай смысл словами.\n"
        f"{speaker_line}"
        "Текст лекции:\n"
    )
    limit = _max_lecture_body_chars(
        gen_max_new_tokens=gen_max_new_tokens,
        system_prompt=system_prompt,
        user_prefix_without_body=user_prefix,
    )
    body = _truncate_utf8_chars(body, limit)
    user_prompt = f"{user_prefix}{body}\n"
    out = (
        await _chat_completion_text(
            client=client,
            model_name=model_name,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=min(gen_max_new_tokens, effective_context),
            temperature=0.0,
        )
    ).strip()
    if not out:
        logger.warning(
            "vLLM: final lecture summary returned empty (lecture body chars=%s, max_tokens=%s)",
            len(body),
            min(gen_max_new_tokens, effective_context),
        )
    return out


async def infer(
    model_checkpoint: Path,
    texts: list[dict[str, Any]],
    model_name: str,
    model_type: str,
    max_length: int,
    device: str,
    max_new_tokens: int | None = None,
    lecture_id: str | None = None,
) -> list[dict[str, Any]]:
    """Публичная суммаризация JSON с полем segments.

    Текущий пайплайн:
    - строит unit-graph (узлы = sentence/utterance units, рёбра = kNN(HNSW/FAISS)+temporal, веса semantic+temporal+speaker)
    - кластеризует граф Leiden в topic-communities (три уровня γ; см. UNIT_GRAPH_LEIDEN_RESOLUTIONS)
    - LLM размечает сообщества уровня UNIT_GRAPH_LABEL_LEVEL (по умолчанию `finest`=больше подтем из нижних сообществ)
    - при HIERARCHICAL_SUMMARY_ENABLED: восходящая суммаризация подтема→тема→лекция; итог лекции из неё, иначе один проход по чанкам

    Элемент списка: ``{"summary": str, "topic_graph": dict | None}``, где ``topic_graph`` — payload для mind map UI (``nodes``, ``links``).

    Модель только через vLLM (OpenAI API); model_checkpoint и device игнорируются.
    lecture_id в теле не используется. Поддерживается model_type qwen.
    """
    del model_checkpoint, device

    if model_type != "qwen":
        raise ValueError(f"Unsupported model_type: {model_type}. Expected 'qwen'.")

    client = _get_vllm_openai_client()
    gen_max_new_tokens = max_new_tokens if max_new_tokens is not None else 8192
    effective_context = max(max_length, int(os.getenv("LLM_CONTEXT_MAX_LENGTH", "8192")))
    summaries: list[dict[str, Any]] = []
    system_prompt = (
        "Ты академический помощник по суммаризации лекций и учебных выступлений. "
        "Отвечай содержательно и структурно: тема лекции, ключевые тезисы, определения, аргументы, "
        "примеры, выводы и практическая ценность. Сохраняй фактическую точность, не выдумывай. "
        "Итоговое резюме лекции излагай обобщённо обычным языком: без формул, символики и LaTeX; "
        "акцент на смысле и главных выводах, не на технике выкладок. "
        "Допускается обычный Markdown (заголовки, списки, выделение) для читаемости; без HTML. "
        "Разметка: разделяй смысловые блоки пустой строкой; для разделов используй ## или ###; "
        "не начинай строку с нумерации вида «2.1.» (цифра.цифра.) — пиши «### 2.1.» или список. "
        "Временные метки в тексте давай как «[м:сс]» в обычном тексте. "
        "Не добавляй рассуждения и теги <think>."
    )

    for payload in texts:
        _wall_toks = infer_wall_tracking_begin()
        try:
            raw_segs = payload.get("segments") if isinstance(payload, dict) else None
            segments = raw_segs if isinstance(raw_segs, list) else []
            transcription_block, has_multiple_speakers, speaker_list, _ = _segments_prompt_meta(segments)

            normalized = _normalize_segments(segments)
            units_raw = _presegment_units(normalized)
            embed_remote = bool(os.getenv("RAG_EMBEDDINGS_SERVE_URL", "").strip())
            logger.info(
                "unit_graph: building lecture_id=%s units=%s (embeddings_via_http=%s)",
                lecture_id,
                len(units_raw),
                embed_remote,
            )
            graph = unit_graph.build_unit_graph(units=units_raw, lecture_id=lecture_id)
            logger.info(
                "unit_graph: done nodes=%s communities=%s",
                len(graph.get("nodes") or []),
                len(graph.get("communities") or []),
            )
            unit_graph_payload = {k: v for k, v in graph.items() if k != "community_texts"}

            try:
                _hier_summary = []
                for h in unit_graph_payload.get("hierarchy") or []:
                    if not isinstance(h, dict):
                        continue
                    _r = float(h.get("resolution", 0.0))
                    _n = len(h.get("communities") or [])
                    _hier_summary.append(f"γ={_r:.3f}:{_n}")
                _t0_present = sum(
                    1 for n in (unit_graph_payload.get("nodes") or []) if isinstance(n, dict) and n.get("t0") is not None
                )
                _total_nodes = len(unit_graph_payload.get("nodes") or [])
                _last_t0 = max(
                    (
                        float(n.get("t0"))
                        for n in (unit_graph_payload.get("nodes") or [])
                        if isinstance(n, dict) and isinstance(n.get("t0"), (int, float))
                    ),
                    default=None,
                )
                logger.info(
                    "unit_graph: hierarchy levels [%s] | nodes_with_t0=%s/%s | last_t0=%s",
                    ", ".join(_hier_summary) or "-",
                    _t0_present,
                    _total_nodes,
                    f"{_last_t0:.1f}s" if _last_t0 is not None else "—",
                )
            except (ValueError, TypeError) as _e:
                logger.debug("unit_graph: hierarchy log skipped: %s", _e)

            # Optional LLM annotation layer (NOT part of graph construction/caching).
            n_comm = len(graph.get("community_texts") or [])
            logger.info("vLLM: topic labeling for %s communities", n_comm)
            label_meta = await topic_labels.label_communities(
                chat_completion_text=_chat_completion_text,
                client=client,
                model_name=model_name,
                community_texts=graph.get("community_texts") or [],
                effective_context=effective_context,
            )
            for c in unit_graph_payload.get("communities", []):
                cid = int(c.get("id", -1))
                meta = label_meta.get(cid)
                if meta:
                    c["name"] = meta.get("name")
                    c["summary"] = meta.get("summary")
                    c["keywords"] = meta.get("keywords")

            # Build chunk texts for summarizer (label if present).
            chunk_texts: list[str] = []
            comm_items = graph.get("community_texts") or []
            for it in comm_items:
                cid = int(it.get("id", -1))
                body = str(it.get("body") or "").strip()
                if cid < 0 or not body:
                    continue
                meta = label_meta.get(cid)
                header = f"[Topic {cid}]"
                if meta and meta.get("name"):
                    header = f"[{meta['name']}]"
                chunk_texts.append(f"{header}\n{body}".strip())
            if not chunk_texts:
                chunk_texts = [transcription_block]

            tier_summaries: dict[str, Any] | None = None
            if os.getenv("HIERARCHICAL_SUMMARY_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}:
                logger.info("vLLM: hierarchical summarization (leaves/themes/lecture)")
                tier_summaries = await hierarchy_summarize.hierarchical_bottom_up_summary(
                    chat_completion_text=_chat_completion_text,
                    client=client,
                    model_name=model_name,
                    unit_graph_payload=dict(unit_graph_payload),
                    community_texts=graph.get("community_texts") or [],
                    effective_context=effective_context,
                    gen_max_new_tokens=gen_max_new_tokens,
                )

            h_lec = (
                str(tier_summaries.get("lecture_summary") or "").strip()
                if isinstance(tier_summaries, dict)
                else ""
            )
            if h_lec:
                logger.info("vLLM: lecture summary from hierarchy (chunks fallback skipped)")
                summary_text = h_lec
            else:
                logger.info("vLLM: final lecture summary single-pass (chunks=%s)", len(chunk_texts))
                summary_text = await _single_pass_lecture_summary(
                    client=client,
                    model_name=model_name,
                    system_prompt=system_prompt,
                    chunk_texts=chunk_texts,
                    transcription_block=transcription_block,
                    has_multiple_speakers=has_multiple_speakers,
                    speaker_list=speaker_list,
                    gen_max_new_tokens=gen_max_new_tokens,
                    effective_context=effective_context,
                )

            topic_graph_payload = topic_graph_mindmap.unit_graph_to_mindmap_payload(
                {
                    "nodes": unit_graph_payload.get("nodes") or [],
                    "communities": unit_graph_payload.get("communities") or [],
                    "hierarchy": unit_graph_payload.get("hierarchy") or [],
                    "edges": unit_graph_payload.get("edges") or [],
                },
                tier_summaries=tier_summaries,
            )
            try:
                tg_nodes = (topic_graph_payload or {}).get("nodes") or []
                theme_spans = [
                    (
                        str(n.get("label") or "?"),
                        n.get("communityTimeStart"),
                        n.get("communityTimeEnd"),
                    )
                    for n in tg_nodes
                    if isinstance(n, dict) and n.get("kind") == "theme"
                ]
                lecture_node = next(
                    (n for n in tg_nodes if isinstance(n, dict) and n.get("kind") == "lecture"),
                    None,
                )
                lec_t0 = lecture_node.get("communityTimeStart") if lecture_node else None
                lec_t1 = lecture_node.get("communityTimeEnd") if lecture_node else None
                logger.info(
                    "topic_graph: themes=%s | lecture_span=%s..%s",
                    len(theme_spans),
                    f"{float(lec_t0):.1f}s" if isinstance(lec_t0, (int, float)) else "—",
                    f"{float(lec_t1):.1f}s" if isinstance(lec_t1, (int, float)) else "—",
                )
                for label, t0, t1 in theme_spans:
                    logger.info(
                        "topic_graph: theme '%s' span=%s..%s",
                        label,
                        f"{float(t0):.1f}s" if isinstance(t0, (int, float)) else "—",
                        f"{float(t1):.1f}s" if isinstance(t1, (int, float)) else "—",
                    )
            except (TypeError, ValueError) as _e:
                logger.debug("topic_graph: theme span log skipped: %s", _e)

            qw, em = infer_wall_totals_snapshot()
            summaries.append(
                {
                    "summary": summary_text,
                    "topic_graph": topic_graph_payload,
                    "_task_wall_seconds": {"qwen": qw, "embeddings": em},
                }
            )
        finally:
            infer_wall_tracking_end(_wall_toks)

    return summaries


async def answer_question_with_rag(
    *,
    lecture_id: str,
    transcription_json: dict[str, Any],
    question: str,
    model_name: str,
    max_length: int,
    max_new_tokens: int | None = None,
) -> str:
    """Ответ на вопрос по лекции: retrieval через qa_rag (Qdrant + LlamaIndex) в отдельном потоке, затем один вызов LLM по контексту.

    lecture_id задаёт имя коллекции в Qdrant. Пустой транскрипт → короткое сообщение об ошибке.
    """
    client = _get_vllm_openai_client()
    if max_new_tokens is not None:
        gen_max_new_tokens = max_new_tokens
    else:
        raw_def = os.getenv("RAG_QA_DEFAULT_MAX_NEW_TOKENS", "4096").strip()
        try:
            gen_max_new_tokens = max(512, int(raw_def))
        except ValueError:
            gen_max_new_tokens = 4096
    effective_context = max(max_length, int(os.getenv("LLM_CONTEXT_MAX_LENGTH", "8192")))

    raw_segs = transcription_json.get("segments") if isinstance(transcription_json, dict) else None
    segments = raw_segs if isinstance(raw_segs, list) else []
    transcription_block, _, _, _, time_spans = _segments_prompt_meta_with_spans(segments)
    if not transcription_block.strip():
        return "Не удалось ответить: пустой транскрипт."

    q = question.strip()
    # Одна дорожка «резюме лекции» смешивает топ релевантности с общим обзором и ухудшает ответы на короткие вопросы.
    queries = [q] if q else ["резюме лекции"]

    try:
        selected_chunk_ids, chunk_texts, _ = await asyncio.to_thread(
            qa_rag.retrieve_chunks_for_queries_sync,
            lecture_id,
            queries,
            transcription_block,
        )
    except qa_rag.RAGNotIndexedError:
        return (
            "Не удалось ответить: векторный индекс лекции ещё не сохранён в Qdrant "
            "(ожидайте задачу rag.index_transcript) или отключён RAG_INDEXING_ENABLED."
        )

    selected_ids = list(dict.fromkeys(selected_chunk_ids))
    normalized = _normalize_segments(segments)
    units_raw = _presegment_units(normalized)
    semantic_units, unit_text = unit_graph.build_semantic_units(units_raw)

    def _ctx_line(cid: int) -> str:
        body = chunk_texts[cid].strip() if 0 <= cid < len(chunk_texts) else ""
        if not body:
            return ""
        ts = _timestamp_for_rag_chunk(
            transcription_block=transcription_block,
            frag=body,
            spans=time_spans,
            unit_text=unit_text,
            semantic_units=semantic_units,
        )
        label = _format_rag_timestamp(ts)
        return f"Выдержка [{label}]:\n{body}"

    context_block = "\n\n".join(
        line for cid in selected_ids if (line := _ctx_line(cid))
    )
    if not context_block:
        context_block = transcription_block

    unit_spans = [(int(u["unit_id"]), int(u["char_start"]), int(u["char_end"])) for u in semantic_units]

    evidence_units: list[int] = []
    for cid in selected_ids:
        if not (0 <= cid < len(chunk_texts)):
            continue
        frag = str(chunk_texts[cid] or "").strip()
        if not frag:
            continue
        pos = transcription_block.find(frag)
        if pos < 0:
            # try on unit_text representation too
            pos = unit_text.find(frag)
            if pos < 0:
                continue
        a = pos
        b = pos + len(frag)
        for uid, s, e in unit_spans:
            if e <= a or s >= b:
                continue
            evidence_units.append(uid)
    evidence_units = sorted(set(evidence_units))
    evidence_block = ""
    if evidence_units:
        max_e = max(10, min(120, int(os.getenv("RAG_EVIDENCE_MAX_UNITS", "40"))))
        picked = evidence_units[:max_e]
        lines = []
        for uid in picked:
            if 0 <= uid < len(semantic_units):
                su = semantic_units[uid]
                ts = _format_rag_timestamp(su.get("t0"))
                lines.append(f"[{ts}] {su['line']}")
        if lines:
            evidence_block = (
                "Строки транскрипта с отметкой времени начала реплики (пересечение с найденными выдержками; "
                "в ответе ссылайся на те же отметки, например «[3:05]»):\n"
                + "\n".join(lines)
                + "\n\n"
            )

    system_prompt = (
        "Ты помогаешь отвечать на вопросы по записи лекции. Перед тобой — выдержки из транскрипта, "
        "подобранные поиском по чанкам; они могут быть неполными и не в хронологическом порядке.\n"
        "Правила ответа:\n"
        "— Используй только утверждения, которые прямо вытекают из приведённого текста (выдержки с меткой [м:сс] "
        "и строки с той же меткой времени). Не добавляй факты, определения, «планы лекции», формулы и примеры, "
        "которых нет в этих выдержках.\n"
        "— Запрещено обобщать в стиле программы курса (например «тема недели», «основные моменты, которые будут "
        "обсуждаться»), если в тексте этого нет дословно или почти дословно.\n"
        "— Если искомого нет в предоставленных выдержках — напиши одной фразой, что в них этого нет; "
        "не восполняй знание из памяти.\n"
        "— Каждое содержательное утверждение сопровождай указанием источника в скобках: отметка времени в формате "
        "как в тексте (например «[3:05]» или «[1:02:03]»), без номеров фрагментов.\n"
        "— Формат ответа: допускается обычный Markdown (заголовки ##/###, списки, выделение); без HTML. "
        "Между абзацами, подпунктами и блоками «Источник»/«Пояснение» оставляй пустую строку. "
        "Не начинай строку с «2.1.» (цифра.цифра.) — для нумерации подпунктов используй ### 2.1. или список.\n"
        "— Формулируй по-русски, конкретно; при возможности сохраняй терминологию спикера. Там, где в выдержках есть "
        "достаточно материала, развёрнуто поясняй ход мысли, определения, примеры и связи между фрагментами — "
        "без выдумывания. Краткая дословная цитата допустима, если она есть в тексте."
    )
    user_prompt = (
        f"Вопрос:\n{q}\n\n"
        f"{evidence_block}"
        "Выдержки из транскрипта лекции (отвечать только опираясь на них; заголовки «Выдержка […]» задают время):\n"
        f"{context_block}\n\n"
        "Сформулируй развёрнутый ответ на русском:\n"
        "1) Начни с ответа по сути (несколько предложений; для сложного вопроса — до одного короткого абзаца), "
        "без вводных «по лекции тема недели», если этого нет в тексте выше.\n"
        "2) Обязательно добавь детализацию: все релевантные к вопросу моменты из выдержек и строк с временем; "
        "структурируй подзаголовками или нумерованным/маркированным списком, где это уместно; поясняй термины и "
        "шаги рассуждения так, как они даны в транскрипте; привязывай каждый тезис к отметке «[м:сс]» "
        "(или «[ч:мм:сс]»), как в выдержках. Не сокращай ответ до минимума, если материала в выдержках хватает.\n"
        "3) Если в выдержках только частично затронут предмет вопроса — опиши только то, что есть, и явно укажи, "
        "чего рядом в этом отрезке нет."
    )
    return await _chat_completion_text(
        client=client,
        model_name=model_name,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=min(gen_max_new_tokens, effective_context),
    )
