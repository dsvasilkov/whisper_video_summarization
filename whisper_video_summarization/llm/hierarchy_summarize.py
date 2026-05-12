"""Восходящая суммаризация: подтемы (листья по сообществам нижнего уровня) → темы → лекция.

По умолчанию текст подтемы берётся из LLM-разметки ``topic_labels`` (сырой текст сообщества там же).
При ``HIERARCHICAL_SUMMARY_LEAF_LLM=true`` — дополнительный проход: развёрнутое резюме по разметке + полному ``body``.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any

from openai import AsyncOpenAI

from whisper_video_summarization.llm import topic_graph_mindmap

_SYSTEM = (
    "Ты академический помощник по суммаризации лекций. Пиши на русском. "
    "Опирайся только на переданные фрагменты: не придумывай факты и не переформулируй содержание в общие формулы «про это говорится». "
    "Избегай мета-текста: не начинай с «Лекция рассматривает», «Далее обсуждается», «Спикер утверждает». "
    "Сформулируй конкретно: условия, определения, шаги, формулы, обозначения и выводы в том виде, как они следуют из текста. "
    "Для связности допускается обычный Markdown (заголовки ##/###, списки, **выделение**); без HTML и без служебных XML-тегов. "
    "Между абзацами и смысловыми блоками оставляй пустую строку; не начинай строку с нумерации «2.1.» (цифра.цифра.) — "
    "для подпунктов используй ### 2.1. или маркированный список."
)


def _parse_json_object(raw: str) -> dict[str, Any] | None:
    """Первый валидный JSON-объект в тексте (устойчиво к тексту до/после и к ```-блокам)."""
    if not raw or not isinstance(raw, str):
        return None
    t = raw.strip()
    if not t:
        return None
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9]*\s*", "", t)
        t = re.sub(r"\s*```\s*$", "", t).strip()

    decoder = json.JSONDecoder()
    for start in range(len(t)):
        if t[start] != "{":
            continue
        try:
            obj, _end = decoder.raw_decode(t, start)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return None


def _merge_summary_chunks(chunks: list[str]) -> str:
    out: list[str] = []
    seen: set[str] = set()
    for piece in chunks:
        x = (piece or "").strip()
        if not x:
            continue
        nk = x.lower()
        if nk in seen:
            continue
        seen.add(nk)
        out.append(x)
    return "\n\n".join(out)


def _normalized_nested_summary(
    obj: dict[str, Any],
    *,
    title_key: str,
    summary_key: str,
    summary_numbered_re: re.Pattern[str],
) -> tuple[str, str]:
    """Собирает title и одно текстовое поле из summary_key + summary_2 … (модели иногда режут так)."""
    title = str(obj.get(title_key) or "").strip()
    parts: list[str] = []

    prim = obj.get(summary_key)
    if isinstance(prim, str) and prim.strip():
        parts.append(prim.strip())

    numbered: list[tuple[int, str]] = []
    for k, val in obj.items():
        ks = str(k)
        if ks == summary_key:
            continue
        m = summary_numbered_re.fullmatch(ks)
        if not m:
            continue
        if isinstance(val, str) and val.strip():
            numbered.append((int(m.group(1)), val.strip()))
    numbered.sort(key=lambda x: x[0])
    for _, text in numbered:
        parts.append(text)

    body = _merge_summary_chunks(parts)
    return title, body


_THEME_SUM_NUM = re.compile(r"theme_summary_(\d+)")
_LEC_SUM_NUM = re.compile(r"lecture_summary_(\d+)")


def _normalized_theme_llm_blob(obj: dict[str, Any] | None) -> tuple[str, str]:
    if not obj:
        return "", ""
    return _normalized_nested_summary(obj, title_key="theme_title", summary_key="theme_summary", summary_numbered_re=_THEME_SUM_NUM)


def _normalized_lecture_llm_blob(obj: dict[str, Any] | None) -> tuple[str, str]:
    if not obj:
        return "", ""
    return _normalized_nested_summary(
        obj, title_key="lecture_title", summary_key="lecture_summary", summary_numbered_re=_LEC_SUM_NUM
    )


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _truncate(s: str, max_chars: int) -> str:
    t = (s or "").strip()
    if len(t) <= max_chars:
        return t
    return t[: max_chars - 1] + "…"


def _body_by_community(community_texts: list[dict[str, Any]]) -> dict[int, str]:
    out: dict[int, str] = {}
    for it in community_texts:
        if not isinstance(it, dict):
            continue
        try:
            cid = int(it.get("id", -1))
        except (TypeError, ValueError):
            continue
        if cid < 0:
            continue
        b = str(it.get("body") or "").strip()
        if b:
            out[cid] = b
    return out


async def hierarchical_bottom_up_summary(
    *,
    chat_completion_text: Any,
    client: AsyncOpenAI,
    model_name: str,
    unit_graph_payload: dict[str, Any],
    community_texts: list[dict[str, Any]],
    effective_context: int,
    gen_max_new_tokens: int,
) -> dict[str, Any] | None:
    """``lecture_title``, ``lecture_summary``, ``theme_titles`` ``{ti:str}``, ``themes``, ``subtopics`` или None."""
    parsed = topic_graph_mindmap._parse_labeled_communities(unit_graph_payload)
    if not parsed:
        return None
    by_id, items = parsed
    hierarchy = unit_graph_payload.get("hierarchy") or []
    if not isinstance(hierarchy, list):
        hierarchy = []

    ctx = topic_graph_mindmap.try_leiden_topics_context(unit_graph_payload, by_id, items, hierarchy)
    if ctx is None:
        return None

    item_by_micro = ctx["item_by_micro"]
    micros_by_theme = ctx["micros_by_theme"]
    theme_order = ctx["theme_order"]
    bodies = _body_by_community(community_texts if isinstance(community_texts, list) else [])

    max_in = max(4000, _env_int("HIERARCHICAL_SUMMARY_MAX_INPUT_CHARS", 28000))
    mt_sub = min(effective_context, _env_int("HIERARCHICAL_SUMMARY_SUB_MAX_TOKENS", 2048))
    mt_theme = min(effective_context, _env_int("HIERARCHICAL_SUMMARY_THEME_MAX_TOKENS", 4096))
    mt_lecture = min(effective_context, gen_max_new_tokens, _env_int("HIERARCHICAL_SUMMARY_LECTURE_MAX_TOKENS", 8192))
    conc = max(1, min(12, _env_int("HIERARCHICAL_SUMMARY_CONCURRENCY", 6)))
    sem = asyncio.Semaphore(conc)
    leaf_llm = os.getenv("HIERARCHICAL_SUMMARY_LEAF_LLM", "false").strip().lower() in {"1", "true", "yes", "on"}

    all_leaf_ids = sorted(
        item_by_micro.keys(),
        key=lambda mm: topic_graph_mindmap._first_time(item_by_micro[mm]["ids"], by_id),
    )

    async def _one_leaf_llm(mic_id: int) -> tuple[int, str]:
        it = item_by_micro.get(mic_id)
        if not it:
            return mic_id, ""
        nm = str(it.get("name") or "").strip() or f"Фрагмент {mic_id}"
        sm = str(it.get("summary") or "").strip()
        if not sm and mic_id in bodies:
            sm = _truncate(bodies[mic_id], max_in)
        if not sm:
            sm = topic_graph_mindmap._snippet(it.get("ids") or [], by_id)
        raw_comm = _truncate(bodies.get(mic_id, ""), max_in) if mic_id in bodies else ""
        block = _truncate(
            f"{nm}\n\nПересказ из разметки:\n{sm}"
            + (f"\n\nИсходный текст:\n{raw_comm}" if raw_comm else ""),
            max_in,
        )
        if not block.strip():
            return mic_id, ""
        user = (
            "Ниже подтема лекции: пересказ и при наличии — исходные реплики. "
            "Сожми в связный текст максимально конкретно: что именно делают (подставляют, упрощают, доказывают), "
            "с какими обозначениями и при каких условиях; сохраняй формулы и числа из текста. "
            "Без фраз «лекция рассматривает», «далее обсуждается», «спикер объясняет». "
            "Допускается Markdown в ответе.\n\n"
            + block
        )
        async with sem:
            raw = await chat_completion_text(
                client=client,
                model_name=model_name,
                system_prompt=_SYSTEM,
                user_prompt=user,
                max_tokens=mt_sub,
                temperature=0.0,
            )
        return mic_id, (raw or "").strip()

    def _leaf_without_llm(mic_id: int) -> tuple[int, str]:
        it = item_by_micro.get(mic_id)
        if not it:
            return mic_id, ""
        nm = str(it.get("name") or "").strip() or f"Фрагмент {mic_id}"
        sm = str(it.get("summary") or "").strip()
        if not sm and mic_id in bodies:
            sm = _truncate(bodies[mic_id], max_in)
        if not sm:
            sm = topic_graph_mindmap._snippet(it.get("ids") or [], by_id)
        return mic_id, _truncate(f"{nm}\n\n{sm}".strip(), max_in)

    if leaf_llm:
        sub_pairs = await asyncio.gather(*(_one_leaf_llm(mic_id) for mic_id in all_leaf_ids))
        subtopics: dict[int, str] = {mid: txt for mid, txt in sub_pairs if txt}
    else:
        subtopics = {mid: txt for mid, txt in (_leaf_without_llm(m) for m in all_leaf_ids) if txt}

    json_max_attempts = max(1, min(8, _env_int("HIERARCHICAL_SUMMARY_JSON_MAX_ATTEMPTS", 3)))

    async def _one_theme(ti: int) -> tuple[int, str, str]:
        mids = micros_by_theme.get(ti, [])
        parts: list[str] = []
        for mic_id in mids:
            st = subtopics.get(mic_id) or ""
            if not st:
                continue
            lbl = topic_graph_mindmap._macro_label_from_summary(st, mic_id)
            parts.append(f"Подтема ({mic_id}): {lbl}\n{st}")
        block = _truncate("\n\n".join(parts), max_in)
        if not block.strip():
            return ti, "", ""
        user_base = (
            "Объединяются конспекты подтем одной темы (блок ниже). "
            "Верни один JSON: только theme_title и theme_summary; без лишних ключей, без текста до/после JSON. "
            "theme_title: короткий заголовок сути темы, без формул. "
            "theme_summary: обзор темы; внутри строки допускается Markdown. "
            "о чём эта тема, какие идеи и выводы связывают подтемы, к чему приходят. "
            "Не вставляй формулы и математическую символику — только словесно. Факты только из текста ниже.\n\n"
            '{"theme_title":"…","theme_summary":"абзац1\\n\\nабзац2"}\n\n'
            + block
        )
        raw = ""
        stripped = ""
        title, body = "", ""
        theme_system_full = (
            _SYSTEM
            + " Это резюме одной темы (не всей лекции)"
            + "Обобщённый словесный стиль, без формул и LaTeX. Только один JSON-объект."
        )
        theme_system_retry = (
            _SYSTEM
            + " Исправление формата: ответ — только один JSON с ключами theme_title и theme_summary "
            "(та же схема, что в первом задании); theme_summary непустой; без текста до/после JSON."
        )
        for attempt in range(json_max_attempts):
            if attempt == 0:
                user = _truncate(user_base, max_in)
                sys_theme = theme_system_full
            else:
                prev = (raw or "").strip()
                snip = prev[:4000] + ("…" if len(prev) > 4000 else "")
                user = _truncate(
                    "Исправь черновик ниже до валидного JSON: ровно один объект с ключами "
                    "theme_title и theme_summary (theme_summary — непустая строка). "
                    "Экранируй переводы строк внутри строк JSON. Без markdown-ограждений ```, без пояснений.\n\n"
                    "Черновик:\n"
                    + snip,
                    max_in,
                )
                sys_theme = theme_system_retry
            async with sem:
                raw = await chat_completion_text(
                    client=client,
                    model_name=model_name,
                    system_prompt=sys_theme,
                    user_prompt=user,
                    max_tokens=mt_theme,
                    temperature=0.0,
                )
            stripped = (raw or "").strip()
            obj = _parse_json_object(stripped)
            title, body = _normalized_theme_llm_blob(obj if isinstance(obj, dict) else None)
            if (body or "").strip():
                if not (title or "").strip():
                    title = topic_graph_mindmap._macro_label_from_summary(body, ti)
                return ti, body, title
        if stripped:
            body = stripped
            title = topic_graph_mindmap._macro_label_from_summary(body, ti)
            return ti, body, title
        return ti, "", ""

    theme_results = await asyncio.gather(*(_one_theme(ti) for ti in theme_order))
    themes: dict[int, str] = {ti: txt for ti, txt, _ttl in theme_results if txt}
    theme_titles: dict[int, str] = {
        ti: ttl for ti, txt, ttl in theme_results if txt and str(ttl).strip()
    }

    theme_blocks: list[str] = []
    for ti in theme_order:
        tt = themes.get(ti) or ""
        if not tt:
            continue
        head = str(theme_titles.get(ti) or "").strip() or topic_graph_mindmap._macro_label_from_summary(tt, ti)
        theme_blocks.append(f"Тема: {head}\n{tt}")
    lec_body = _truncate("\n\n".join(theme_blocks), max_in)
    if not lec_body.strip():
        return {
            "lecture_summary": "",
            "lecture_title": "",
            "themes": themes,
            "theme_titles": theme_titles,
            "subtopics": subtopics,
        }

    user_prefix = (
        "Ниже — обзорные резюме крупных тем лекции. "
        "Ответ строго: один JSON с ключами lecture_title и lecture_summary; без текста до/после JSON. "
        "lecture_title: короткое обобщающее название всей лекции по смыслу, без формул. "
        "lecture_summary: цельный обзор всей лекции обычным языком: основная линия выступления, ключевые идеи; "
        "внутри строки допускается Markdown. "
        "Формулируй обобщённо: без формул, символики/LaTeX и перечисления переменных; не воспроизводи выкладки — "
        "перескажи смысл и результаты простыми словами, на уровне общего конспекта. "
        "Чему учат и к каким выводам приходят; логично свяжи темы между собой. "
        "Избегай пустых метаформулировок. Только содержание из текста ниже.\n\n"
        '{"lecture_title":"…","lecture_summary":"…"}\n\n'
        "Резюме тем:\n"
    )
    user_lecture_base = _truncate(user_prefix + lec_body, max(12000, max_in))
    lecture_system_full = (
        _SYSTEM
        + " В lecture_title и lecture_summary — обобщённый словесный обзор без формул и LaTeX; излагай суть простым языком. "
        "Ответ — только один JSON."
    )
    lecture_system_retry = (
        _SYSTEM
        + " Исправление формата: ответ — только один JSON с ключами lecture_title и lecture_summary "
        "(та же схема, что в первом задании); lecture_summary непустой; без текста до/после JSON."
    )
    raw_lec = ""
    lec_strip = ""
    lecture_title, lecture_summary = "", ""
    for attempt in range(json_max_attempts):
        if attempt == 0:
            user_lecture = _truncate(user_lecture_base, max(12000, max_in))
            sys_lecture = lecture_system_full
        else:
            prev = (raw_lec or "").strip()
            snip = prev[:4000] + ("…" if len(prev) > 4000 else "")
            user_lecture = _truncate(
                "Исправь черновик ниже до валидного JSON: ровно один объект с ключами "
                "lecture_title и lecture_summary (lecture_summary — непустая строка). "
                "Экранируй переводы строк внутри строк JSON. Без markdown-ограждений ```, без пояснений.\n\n"
                "Черновик:\n"
                + snip,
                max(12000, max_in),
            )
            sys_lecture = lecture_system_retry
        raw_lec = await chat_completion_text(
            client=client,
            model_name=model_name,
            system_prompt=sys_lecture,
            user_prompt=user_lecture,
            max_tokens=min(effective_context, max(2048, mt_lecture)),
            temperature=0.0,
        )
        lec_strip = (raw_lec or "").strip()
        lec_obj = _parse_json_object(lec_strip)
        lecture_title, lecture_summary = _normalized_lecture_llm_blob(lec_obj if isinstance(lec_obj, dict) else None)
        if (lecture_summary or "").strip():
            if not (lecture_title or "").strip():
                lecture_title = topic_graph_mindmap._macro_label_from_summary(lecture_summary, 0)
            break
    if not (lecture_summary or "").strip():
        lecture_summary = lec_strip
    if not (lecture_title or "").strip() and (lecture_summary or "").strip():
        lecture_title = topic_graph_mindmap._macro_label_from_summary(lecture_summary, 0)

    return {
        "lecture_summary": lecture_summary,
        "lecture_title": lecture_title,
        "themes": themes,
        "theme_titles": theme_titles,
        "subtopics": subtopics,
    }
