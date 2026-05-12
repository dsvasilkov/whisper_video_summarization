from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any

from openai import AsyncOpenAI


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _label_from_text(text: str, cid: int, *, max_len: int = 56) -> str:
    one = (text or "").strip().replace("\n", " ")
    if not one:
        return f"Фрагмент {cid}"
    if len(one) <= max_len:
        return one
    return one[: max_len - 1].rstrip() + "…"


def _normalize_json_wrapping(t: str) -> str:
    t = (t or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9]*\s*", "", t)
        t = re.sub(r"\s*```\s*$", "", t).strip()
    t = t.replace("\u201c", '"').replace("\u201d", '"')
    t = t.replace("\u00ab", '"').replace("\u00bb", '"')
    return t


def _repair_json_string_backslashes(t: str) -> str:
    """Внутри JSON-строк удваивает \\ там, где это не стандартный escape JSON (например LaTeX \\sigma)."""
    out: list[str] = []
    i = 0
    n = len(t)
    in_string = False
    while i < n:
        ch = t[i]
        if not in_string:
            out.append(ch)
            if ch == '"':
                bs = 0
                k = i - 1
                while k >= 0 and t[k] == "\\":
                    bs += 1
                    k -= 1
                if bs % 2 == 0:
                    in_string = True
            i += 1
            continue
        if ch == '"':
            bs = 0
            k = i - 1
            while k >= 0 and t[k] == "\\":
                bs += 1
                k -= 1
            if bs % 2 == 0:
                in_string = False
            out.append(ch)
            i += 1
            continue
        if ch == "\\" and i + 1 < n:
            nxt = t[i + 1]
            if nxt in '"\\/bfnrt':
                out.append(ch)
                out.append(nxt)
                i += 2
                continue
            if nxt == "u" and i + 6 <= n:
                hexd = t[i + 2 : i + 6]
                if len(hexd) == 4 and all(c in "0123456789abcdefABCDEF" for c in hexd):
                    out.append(t[i : i + 6])
                    i += 6
                    continue
            out.append("\\\\")
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _parse_topic_json(raw: str) -> dict[str, Any] | None:
    """Первый объект JSON; терпим LaTeX с одиночным \\ в строках (невалидный чистый JSON)."""
    if not raw or not isinstance(raw, str):
        return None
    t = _normalize_json_wrapping(raw)

    def _try_load(s: str) -> dict[str, Any] | None:
        try:
            obj = json.loads(s)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    if (obj := _try_load(t)) is not None:
        return obj
    fixed = _repair_json_string_backslashes(t)
    if (obj := _try_load(fixed)) is not None:
        return obj

    decoder = json.JSONDecoder()
    for i, ch in enumerate(fixed):
        if ch != "{":
            continue
        try:
            obj, _end = decoder.raw_decode(fixed, i)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return None


async def label_communities(
    *,
    chat_completion_text: Any,
    client: AsyncOpenAI,
    model_name: str,
    community_texts: list[dict[str, Any]],
    effective_context: int,
) -> dict[int, dict[str, Any]]:
    """LLM naming layer (optional): returns {community_id: {name, summary, keywords}}."""
    if os.getenv("UNIT_GRAPH_TOPIC_LABELS_ENABLED", "true").strip().lower() not in {"1", "true", "yes", "on"}:
        return {}
    if not community_texts:
        return {}

    max_chars = max(1200, min(32000, int(os.getenv("UNIT_GRAPH_TOPIC_LABEL_MAX_CHARS", "12000"))))
    max_resp = max(512, min(8192, int(os.getenv("UNIT_GRAPH_TOPIC_LABEL_MAX_TOKENS", "4096"))))
    sem = asyncio.Semaphore(max(1, min(12, int(os.getenv("UNIT_GRAPH_TOPIC_LABEL_CONCURRENCY", "6")))))
    json_max_attempts = max(1, min(8, _env_int("UNIT_GRAPH_TOPIC_LABEL_JSON_MAX_ATTEMPTS", 3)))

    def _meta_from_valid_topic_json(obj: dict[str, Any], cid: int) -> dict[str, Any] | None:
        """Только если в JSON есть непустой topic_summary (без подстановки сырого ответа)."""
        summ = str(obj.get("topic_summary") or "").strip()
        if not summ:
            return None
        name = str(obj.get("topic_name") or "").strip()
        kws = obj.get("keywords")
        if not isinstance(kws, list):
            kws = []
        keywords = [str(x).strip() for x in kws if str(x).strip()][:16]
        if not name:
            name = _label_from_text(summ, cid)
        return {"name": name, "summary": summ, "keywords": keywords}

    def _meta_fallback_raw(raw_stripped: str, cid: int) -> dict[str, Any] | None:
        if not raw_stripped:
            return None
        return {
            "name": _label_from_text(raw_stripped, cid),
            "summary": raw_stripped,
            "keywords": [],
        }

    async def _one(item: dict[str, Any]) -> tuple[int, dict[str, Any] | None]:
        cid = int(item.get("id", -1))
        body = str(item.get("body") or "").strip()
        if cid < 0 or not body:
            return cid, None
        sample = body if len(body) <= max_chars else body[: max_chars - 1] + "…"
        user_base = (
            "Дан фрагмент лекции — полный текст одного сообщества (реплики со спикерами). "
            "Дай JSON строго вида:\n"
            '{"topic_name": str, "topic_summary": str, "keywords": [str, ...]}\n'
            "topic_name: одна короткая точная подпись о том, ЧТО делается в фрагменте, "
            "а не «Обсуждение распределений»). Уникально среди фрагментов лекции.\n"
            "topic_summary: изложение фактов и действий ИЗ текста без общих окольных фраз. "
            "НЕ использовать формулировки вроде: «Лекция рассматривает», «Далее обсуждается», «Спикер объясняет, что», "
            "«В данном фрагменте» — сразу по сути. "
            "Указывай: условия (если они явно есть в тексте), конкретные выражения, шаг подстановки, "
            "что сокращается/что остаётся, получившиеся формулы и обозначения; "
            "если есть числовые коэффициенты или примеры — перенеси их как в транскрипте. Несколько коротких абзацев по шагам, если нужно.\n"
            "keywords: 5–14 коротких меток из содержания (термины, объекты, операции).\n"
            "Допускается Markdown в полях topic_name и topic_summary.\n\n"
            f"TEXT:\n{sample}\n"
        )
        raw = ""
        system_full = (
            "Ты помогаешь структурировать лекцию. Отвечай только JSON. "
            "Пиши максимально конкретно: формулы, условия, подстановки, обозначения и числа — как во фрагменте; "
            "не добавляй фактов от себя."
        )
        system_retry = (
            "Исправление формата: ответ — только один JSON с ключами topic_name, topic_summary, keywords "
            "(массив строк); та же схема, что в первом задании; topic_summary непустой."
        )
        for attempt in range(json_max_attempts):
            if attempt == 0:
                user = user_base
                sys_prompt = system_full
            else:
                prev = (raw or "").strip()
                snip = prev[:2500] + ("…" if len(prev) > 2500 else "")
                user = (
                    "Исправь черновик ниже до валидного JSON: ровно один объект с ключами "
                    "topic_name, topic_summary, keywords (массив строк); topic_summary — непустая строка. "
                    "Экранируй переводы строк внутри строк JSON. Без markdown-ограждений ```, без пояснений.\n\n"
                    "Черновик:\n"
                    + snip
                )
                sys_prompt = system_retry
            async with sem:
                raw = await chat_completion_text(
                    client=client,
                    model_name=model_name,
                    system_prompt=sys_prompt,
                    user_prompt=user,
                    max_tokens=min(max_resp, effective_context),
                    temperature=0.0,
                )
            obj = _parse_topic_json(raw or "")
            if isinstance(obj, dict):
                meta = _meta_from_valid_topic_json(obj, cid)
                if meta:
                    return cid, meta
        return cid, _meta_fallback_raw((raw or "").strip(), cid)

    pairs = await asyncio.gather(*(_one(it) for it in community_texts))
    return {cid: meta for cid, meta in pairs if meta and cid >= 0}

