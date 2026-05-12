"""Unit-graph → JSON для mind map UI: ``nodes`` + ``links``."""

from __future__ import annotations

import os
from typing import Any


def _grid(i: int, cols: int, w: float = 300.0, h: float = 200.0) -> dict[str, float]:
    cols = max(1, cols)
    return {"x": float((i % cols) * w), "y": float((i // cols) * h)}


def _grid_layout(nodes: list[dict[str, Any]], cols: int | None = None) -> list[dict[str, Any]]:
    if not nodes or any(isinstance(n.get("position"), dict) for n in nodes):
        return nodes
    n = len(nodes)
    c = cols if cols is not None else min(4, max(2, int(n**0.5 + 0.999)))
    return [{**nd, "position": _grid(i, c)} for i, nd in enumerate(nodes)]


def _unit_map(nodes: list[Any]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for u in nodes:
        if not isinstance(u, dict):
            continue
        rid = u.get("id")
        try:
            uid = int(rid)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        out[uid] = u
    return out


def _first_time(node_ids: list[int], by_id: dict[int, dict[str, Any]]) -> float:
    t = float("inf")
    for k in node_ids:
        u = by_id.get(k)
        if not u:
            continue
        try:
            v = float(u["timestamp"]) if u.get("timestamp") is not None else float("nan")  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if v == v:
            t = min(t, v)
    return 0.0 if t == float("inf") else t


def _snippet(ids: list[int], by_id: dict[int, dict[str, Any]]) -> str:
    parts = []
    for k in ids[:4]:
        u = by_id.get(k)
        if not u:
            continue
        s = str(u.get("line") or "").strip() or str(u.get("text") or "").strip()
        if s:
            parts.append(s)
    body = " ".join(parts)
    return body if len(body) <= 240 else body[:237] + "…"


def _community_transcript(member_ids: list[Any], by_id: dict[int, dict[str, Any]]) -> str:
    """Хронологическая склейка реплик сообщества (для UI: фактический текст vs резюме)."""
    ids: list[int] = []
    for x in member_ids or []:
        try:
            ids.append(int(x))
        except (TypeError, ValueError):
            continue
    ids.sort(key=lambda uid: float(_first_time([uid], by_id)))
    lines: list[str] = []
    for uid in ids:
        u = by_id.get(uid)
        if not u:
            continue
        line = str(u.get("line") or "").strip() or str(u.get("text") or "").strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def _safe_float_time(v: Any) -> float | None:
    if v is None:
        return None
    try:
        x = float(v)
        return x if x == x else None
    except (TypeError, ValueError):
        return None


def _community_time_span(member_ids: list[Any], by_id: dict[int, dict[str, Any]]) -> tuple[float | None, float | None]:
    """Границы блока сообщества (секунды) ТОЛЬКО по реальным t0/t1; ``timestamp`` без t0 — это unit_id (индекс), а не время."""
    starts: list[float] = []
    ends: list[float] = []
    for x in member_ids or []:
        try:
            uid = int(x)
        except (TypeError, ValueError):
            continue
        u = by_id.get(uid)
        if not u:
            continue
        t0 = _safe_float_time(u.get("t0"))
        t1 = _safe_float_time(u.get("t1"))
        st = t0
        en = t1 if t1 is not None else t0
        if st is not None:
            starts.append(st)
        if en is not None:
            ends.append(en)
        elif st is not None:
            ends.append(st)
    if not starts:
        return None, None
    t_lo = float(min(starts))
    t_hi = float(max(ends)) if ends else float(max(starts))
    return t_lo, t_hi


def _theme_unit_ids(
    ti: int,
    th_comms: list[Any],
    micros_by_theme: dict[int, list[int]],
    item_by_micro: dict[int, dict[str, Any]],
) -> list[int]:
    """Все unit-id, отнесённые к теме: объединение узлов сообщества темы и юнитов всех её листьев."""
    out: set[int] = set()
    if 0 <= ti < len(th_comms):
        for uid in _community_member_ids(th_comms[ti]):
            out.add(int(uid))
    for mic_id in micros_by_theme.get(ti, []) or []:
        it = item_by_micro.get(int(mic_id))
        if not it:
            continue
        for uid in (it.get("ids") or []):
            try:
                out.add(int(uid))
            except (TypeError, ValueError):
                continue
    return sorted(out)


def _lecture_time_span(by_id: dict[int, dict[str, Any]]) -> tuple[float | None, float | None]:
    """Полный временной диапазон лекции по всем юнитам графа."""
    return _community_time_span(list(by_id.keys()), by_id)


def _community_member_ids(c: Any) -> list[int]:
    if not isinstance(c, dict):
        return []
    subs = c.get("nodes")
    if not isinstance(subs, list):
        return []
    out: list[int] = []
    for x in subs:
        try:
            out.append(int(x))
        except (TypeError, ValueError):
            continue
    return out


def _parse_labeled_communities(raw: dict[str, Any]) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]] | None:
    rn, rc = raw.get("nodes"), raw.get("communities")
    if not isinstance(rn, list) or not isinstance(rc, list) or not rc:
        return None
    by_id = _unit_map([x for x in rn if isinstance(x, dict)])
    items: list[dict[str, Any]] = []
    for c in rc:
        if not isinstance(c, dict):
            continue
        try:
            cid = int(c["id"])  # type: ignore[arg-type]
        except (KeyError, TypeError, ValueError):
            continue
        ids = _community_member_ids(c)
        if cid < 0 or not ids:
            continue
        nm, sm = c.get("name"), c.get("summary")
        kraw = c.get("keywords")
        kw_list: list[str] = []
        if isinstance(kraw, list):
            kw_list = [str(x).strip() for x in kraw if str(x).strip()][:24]
        items.append({"id": cid, "ids": ids, "name": nm, "summary": sm, "keywords": kw_list})
    if not items:
        return None
    items.sort(key=lambda it: _first_time(it["ids"], by_id))
    return by_id, items


def _coarser_hierarchy_level(hierarchy: list[dict[str, Any]], n_fine: int) -> dict[str, Any] | None:
    """Уровень с меньшим числом сообществ чем fined (более крупные блоки)."""
    if not hierarchy or n_fine < 2:
        return None
    candidates: list[dict[str, Any]] = []
    for h in hierarchy:
        if not isinstance(h, dict):
            continue
        comms = h.get("communities")
        if not isinstance(comms, list):
            continue
        nc = len(comms)
        if 1 < nc < n_fine:
            candidates.append(h)
    if not candidates:
        return None
    return min(candidates, key=lambda h: len(h.get("communities") or []))


def _assign_fines_to_coarse_macroids(
    fine_items: list[dict[str, Any]],
    coarse_level: dict[str, Any],
) -> list[int]:
    coarse_comms = coarse_level.get("communities")
    if not isinstance(coarse_comms, list):
        return [0] * len(fine_items)
    coarse_sets = [set(_community_member_ids(x)) for x in coarse_comms]
    if not coarse_sets:
        return [0] * len(fine_items)
    parents: list[int] = []
    for it in fine_items:
        fset = set(it["ids"])
        best = max(range(len(coarse_sets)), key=lambda ci: len(fset & coarse_sets[ci]))
        parents.append(int(best))
    return parents


def _unit_to_fine_id(items: list[dict[str, Any]]) -> dict[int, int]:
    m: dict[int, int] = {}
    for it in items:
        cid = int(it["id"])
        for uid in it["ids"]:
            m[int(uid)] = cid
    return m


def _aggregate_related_pairs(
    edges: list[Any],
    unit_to_fine: dict[int, int],
    *,
    k: int,
) -> list[tuple[int, int, float]]:
    acc: dict[tuple[int, int], float] = {}
    for e in edges:
        if not isinstance(e, dict):
            continue
        try:
            s, t = int(e["source"]), int(e["target"])
        except (KeyError, TypeError, ValueError):
            continue
        fs = unit_to_fine.get(s)
        ft = unit_to_fine.get(t)
        if fs is None or ft is None or fs == ft:
            continue
        a, b = (fs, ft) if fs <= ft else (ft, fs)
        w = float(e.get("weight", 0.0) or 0.0)
        acc[(a, b)] = acc.get((a, b), 0.0) + w
    top = sorted(acc.items(), key=lambda x: -x[1])[: max(0, k)]
    return [(a, b, w) for (a, b), w in top]


def _append_global_timeline(items: list[dict[str, Any]], links: list[dict[str, Any]]) -> None:
    """Одна хронология по всей лекции: связывает темы между макроблоками (связный граф тем)."""
    for a, b in zip(items, items[1:]):
        ia, ib = int(a["id"]), int(b["id"])
        links.append({"source": f"c-{ia}", "target": f"c-{ib}", "type": "timeline"})


def _macro_label_from_summary(summary: str, index: int) -> str:
    s = (summary or "").strip().replace("\n", " ")
    if len(s) > 56:
        s = s[:53].rstrip() + "…"
    return s if s else f"Блок {index + 1}"


def _item_indices_for_micro_ids(items: list[dict[str, Any]], micro_ids: set[int]) -> list[int]:
    return [i for i, it in enumerate(items) if int(it.get("id", -1)) in micro_ids]


def _aggregate_fine_summaries(child_iis: list[int], items: list[dict[str, Any]], *, max_chars: int = 720) -> str:
    """Склейка суммаризаций подтем (из LLM) для макроблока."""
    parts: list[str] = []
    for ii in child_iis:
        it = items[ii]
        sm = it.get("summary")
        if isinstance(sm, str) and sm.strip():
            parts.append(sm.strip().replace("\n", " "))
    if not parts:
        return ""
    body = " ".join(parts)
    if len(body) <= max_chars:
        return body
    return body[: max_chars - 1] + "…"


def _macro_block_text(
    child_iis: list[int],
    items: list[dict[str, Any]],
    mids: list[int],
    by_id: dict[int, dict[str, Any]],
    macro_index: int,
) -> tuple[str, str]:
    """(label, summary) для макроблока: приоритет LLM-текстам дочерних тем, иначе сырой сниппет по юнитам."""
    summ = _aggregate_fine_summaries(child_iis, items)
    if not summ:
        summ = _snippet(mids, by_id)
    lbl: str | None = None
    if len(child_iis) == 1:
        it0 = items[child_iis[0]]
        n0 = str(it0.get("name") or "").strip()
        if n0:
            lbl = n0 if len(n0) <= 56 else n0[:53].rstrip() + "…"
    if not lbl:
        lbl = _macro_label_from_summary(summ, macro_index)
    return lbl, summ


def _append_macro_timeline(macro_mids: list[str], links: list[dict[str, Any]]) -> None:
    """Связь крупных блоков слева по хронологии лекции (рядом с parent на микротемы)."""
    for a, b in zip(macro_mids, macro_mids[1:]):
        links.append({"source": a, "target": b, "type": "timeline"})


def _append_theme_chain_timeline(theme_ids: list[str], links: list[dict[str, Any]]) -> None:
    for a, b in zip(theme_ids, theme_ids[1:]):
        links.append({"source": a, "target": b, "type": "timeline"})


def _mindmap_label_level() -> str:
    """Совпадает с unit_graph: `finest` — листья = самый высокий γ; `mid` — средний слой."""
    raw = os.getenv("UNIT_GRAPH_LABEL_LEVEL", "finest").strip().lower()
    return "finest" if raw == "finest" else "mid"


def _unique_label(text: str, seen_lower: dict[str, int]) -> str:
    base = (text or "").strip() or "Тема"
    k = base.lower()
    n = seen_lower.get(k, 0) + 1
    seen_lower[k] = n
    return base if n == 1 else f"{base} ({n})"


def _parent_map_lookup(pm: Any, idx: int) -> int | None:
    if not isinstance(pm, dict) or not pm:
        return None
    v = pm.get(idx)
    if v is None:
        v = pm.get(str(idx))
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def try_leiden_topics_context(
    raw: dict[str, Any],
    by_id: dict[int, dict[str, Any]],
    items: list[dict[str, Any]],
    hierarchy_raw: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Иерархия Leiden → лекция→темы→листья (листьевой уровень = `finest`=микро или `mid`=середина γ)."""
    if len(hierarchy_raw) < 3:
        return None
    h_sorted = sorted(
        [x for x in hierarchy_raw if isinstance(x, dict)],
        key=lambda x: float(x.get("resolution", 0.0)),
    )
    if len(h_sorted) < 3:
        return None
    h_theme, h_sub, h_micro = h_sorted[-3], h_sorted[-2], h_sorted[-1]
    th_comms = h_theme.get("communities")
    sub_comms = h_sub.get("communities")
    mi_comms = h_micro.get("communities")
    if not isinstance(th_comms, list) or not isinstance(sub_comms, list) or not isinstance(mi_comms, list):
        return None
    if not th_comms or not sub_comms or not mi_comms:
        return None

    pm_sub_to_th = h_sub.get("parent_map")
    pm_mi_to_sub = h_micro.get("parent_map")
    if not isinstance(pm_sub_to_th, dict):
        return None

    label_lv = _mindmap_label_level()
    if label_lv == "mid":
        item_by_micro: dict[int, dict[str, Any]] = {}
        for it in items:
            try:
                lid = int(it["id"])
            except (KeyError, TypeError, ValueError):
                continue
            if lid < 0 or lid >= len(sub_comms):
                return None
            item_by_micro[lid] = it
        if not item_by_micro:
            return None
        n_th = len(th_comms)
        theme_order = sorted(range(n_th), key=lambda ti: _first_time(_community_member_ids(th_comms[ti]), by_id))
        micros_by_theme: dict[int, list[int]] = {int(t): [] for t in theme_order}
        for leaf_id in item_by_micro:
            tj = _parent_map_lookup(pm_sub_to_th, leaf_id)
            if tj is None or tj < 0 or tj >= n_th:
                return None
            if tj in micros_by_theme:
                micros_by_theme[tj].append(int(leaf_id))
        for tj in micros_by_theme:
            micros_by_theme[tj].sort(
                key=lambda mm: _first_time(item_by_micro[mm]["ids"], by_id),
            )
        return {
            "th_comms": th_comms,
            "item_by_micro": item_by_micro,
            "theme_order": theme_order,
            "micros_by_theme": micros_by_theme,
        }

    if not isinstance(pm_mi_to_sub, dict):
        return None

    item_by_micro = {}
    for it in items:
        try:
            mid = int(it["id"])
        except (KeyError, TypeError, ValueError):
            continue
        if mid < 0 or mid >= len(mi_comms):
            return None
        item_by_micro[mid] = it

    if not item_by_micro:
        return None

    n_sub = len(sub_comms)
    n_th = len(th_comms)

    micro_to_sub: dict[int, int] = {}
    for mic_id in item_by_micro:
        s = _parent_map_lookup(pm_mi_to_sub, mic_id)
        if s is None or s < 0 or s >= n_sub:
            return None
        micro_to_sub[mic_id] = s

    sub_to_theme: dict[int, int] = {}
    for sj in range(n_sub):
        t = _parent_map_lookup(pm_sub_to_th, sj)
        if t is None or t < 0 or t >= n_th:
            return None
        sub_to_theme[sj] = t

    theme_order = sorted(range(n_th), key=lambda ti: _first_time(_community_member_ids(th_comms[ti]), by_id))

    micros_by_theme = {int(t): [] for t in theme_order}
    for mic_id in item_by_micro:
        sj = micro_to_sub.get(mic_id, -1)
        tj = sub_to_theme.get(sj, -2)
        if tj in micros_by_theme:
            micros_by_theme[tj].append(int(mic_id))
    for tj in micros_by_theme:
        micros_by_theme[tj].sort(
            key=lambda mm: _first_time(item_by_micro[mm]["ids"], by_id),
        )

    return {
        "th_comms": th_comms,
        "item_by_micro": item_by_micro,
        "theme_order": theme_order,
        "micros_by_theme": micros_by_theme,
    }


def _mindmap_lecture_theme_sub_three(
    raw: dict[str, Any],
    by_id: dict[int, dict[str, Any]],
    items: list[dict[str, Any]],
    hierarchy_raw: list[dict[str, Any]],
    tier_summaries: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]] | None:
    """Лекция → темы → подтемы (листы — сообщества уровня `UNIT_GRAPH_LABEL_LEVEL`: mid или finest)."""
    ctx = try_leiden_topics_context(raw, by_id, items, hierarchy_raw)
    if ctx is None:
        return None

    th_comms = ctx["th_comms"]
    item_by_micro = ctx["item_by_micro"]
    theme_order = ctx["theme_order"]
    micros_by_theme = ctx["micros_by_theme"]

    ts_leaf: dict[int, str] = {}
    ts_th: dict[int, str] = {}
    ts_theme_titles: dict[int, str] = {}
    ts_lec = ""
    ts_lec_title = ""
    if tier_summaries and isinstance(tier_summaries, dict):
        raw_sub = tier_summaries.get("subtopics") or {}
        if isinstance(raw_sub, dict):
            for k, v in raw_sub.items():
                try:
                    ts_leaf[int(k)] = str(v).strip()
                except (TypeError, ValueError):
                    continue
        raw_th = tier_summaries.get("themes") or {}
        if isinstance(raw_th, dict):
            for k, v in raw_th.items():
                try:
                    ts_th[int(k)] = str(v).strip()
                except (TypeError, ValueError):
                    continue
        raw_tt = tier_summaries.get("theme_titles") or {}
        if isinstance(raw_tt, dict):
            for k, v in raw_tt.items():
                try:
                    ts_theme_titles[int(k)] = str(v).strip()
                except (TypeError, ValueError):
                    continue
        ts_lec = str(tier_summaries.get("lecture_summary") or "").strip()
        ts_lec_title = str(tier_summaries.get("lecture_title") or "").strip()

    nodes: list[dict[str, Any]] = []
    links: list[dict[str, Any]] = []

    LEC_X, TH_X, SUB_X = 8.0, 200.0, 520.0
    TH_STEP, LEAF_GAP = 200.0, 72.0
    seen_theme_lc: dict[str, int] = {}
    seen_leaf_lc: dict[str, int] = {}

    lecture_id = "lecture-root"
    lec_lbl = ts_lec_title if ts_lec_title else "Лекция"
    lec_t_lo, lec_t_hi = _lecture_time_span(by_id)
    nodes.append(
        {
            "id": lecture_id,
            "label": lec_lbl,
            "summary": ts_lec
            or (
                "Структура по многоуровневой кластеризации Leiden на kNN-графе (FAISS HNSW + семантика/время/спикер)."
            ),
            "community": 0,
            "kind": "lecture",
            "parentId": None,
            "communityTimeStart": lec_t_lo,
            "communityTimeEnd": lec_t_hi,
            "position": {"x": LEC_X, "y": 120.0},
        }
    )

    theme_ids_timeline: list[str] = []
    theme_centers: dict[int, float] = {}

    for ti_idx, ti in enumerate(theme_order):
        y_th = 36.0 + float(ti_idx) * TH_STEP
        theme_centers[ti] = y_th
        tid = f"theme-{ti}"
        theme_ids_timeline.append(tid)

        micro_ids_here = list(micros_by_theme.get(ti, []))
        mids_for_snip: list[int] = []
        for m in micro_ids_here:
            mids_for_snip.extend(item_by_micro[m]["ids"][:2])
        mids_for_snip = mids_for_snip[:8]
        t_summ = str(ts_th.get(ti, "") or "").strip()
        if not t_summ:
            t_summ = _aggregate_fine_summaries(
                _item_indices_for_micro_ids(items, set(micro_ids_here)),
                items,
                max_chars=640,
            )
        if not t_summ:
            th_el = th_comms[ti] if ti < len(th_comms) else None
            t_summ = _snippet(mids_for_snip, by_id) if mids_for_snip else (
                _snippet(_community_member_ids(th_el), by_id) if isinstance(th_el, dict) else ""
            )
        t_lbl = str(ts_theme_titles.get(ti, "") or "").strip() or _macro_label_from_summary(t_summ, ti_idx)
        t_lbl = _unique_label(t_lbl, seen_theme_lc)

        theme_unit_ids = _theme_unit_ids(int(ti), th_comms, micros_by_theme, item_by_micro)
        th_t_lo, th_t_hi = _community_time_span(theme_unit_ids, by_id)

        nodes.append(
            {
                "id": tid,
                "label": t_lbl,
                "summary": t_summ,
                "community": ti,
                "kind": "theme",
                "parentId": lecture_id,
                "communityTimeStart": th_t_lo,
                "communityTimeEnd": th_t_hi,
                "position": {"x": TH_X, "y": y_th},
            }
        )
        links.append({"source": lecture_id, "target": tid, "type": "parent"})

    _append_theme_chain_timeline(theme_ids_timeline, links)

    for ti in theme_order:
        tid = f"theme-{ti}"
        y_th = theme_centers[ti]
        mics = micros_by_theme.get(ti, [])
        nm = len(mics)
        for j, mic_id in enumerate(mics):
            offset = (j - (nm - 1) / 2.0) * LEAF_GAP if nm > 1 else 0.0
            y_l = y_th + offset
            it = item_by_micro[mic_id]
            uu: list[int] = list(item_by_micro[mic_id].get("ids") or [])[:4]
            lf_summ = str(ts_leaf.get(mic_id, "") or "").strip()
            if not lf_summ:
                lf_summ = _aggregate_fine_summaries(
                    _item_indices_for_micro_ids(items, {mic_id}),
                    items,
                    max_chars=560,
                )
            if not lf_summ:
                lf_summ = _snippet(uu, by_id) if uu else ""
            lbl = (
                str(it["name"]).strip()
                if isinstance(it["name"], str) and str(it["name"]).strip()
                else _macro_label_from_summary(lf_summ, j)
            )
            lbl = _unique_label(lbl, seen_leaf_lc)
            sm_raw = it["summary"]
            summary = (
                str(sm_raw).strip()
                if isinstance(sm_raw, str) and str(sm_raw).strip()
                else lf_summ or _snippet(it["ids"], by_id)
            )
            kws = it.get("keywords") if isinstance(it.get("keywords"), list) else []
            kw_safe = [str(x).strip() for x in kws if str(x).strip()][:24]
            fid = f"c-{mic_id}"
            mids_list = item_by_micro[mic_id].get("ids") or []
            comm_body = _community_transcript(mids_list, by_id)
            t_lo, t_hi = _community_time_span(mids_list, by_id)
            nodes.append(
                {
                    "id": fid,
                    "label": lbl,
                    "summary": summary,
                    "communityBody": comm_body,
                    "communityTimeStart": t_lo,
                    "communityTimeEnd": t_hi,
                    "keywords": kw_safe,
                    "community": mic_id,
                    "kind": "subtopic",
                    "parentId": tid,
                    "position": {"x": SUB_X, "y": y_l},
                }
            )
            links.append({"source": tid, "target": fid, "type": "parent"})

    return nodes, links


def _from_communities(raw: dict[str, Any], tier_summaries: dict[str, Any] | None = None) -> dict[str, Any] | None:
    parsed = _parse_labeled_communities(raw)
    if not parsed:
        return None
    by_id, items = parsed
    hierarchy = raw.get("hierarchy") or []
    if not isinstance(hierarchy, list):
        hierarchy = []
    edges = raw.get("edges") or []
    if not isinstance(edges, list):
        edges = []

    max_related = max(0, int(os.getenv("TOPIC_GRAPH_MAX_RELATED", "12")))
    unit_to_fine = _unit_to_fine_id(items)
    related = _aggregate_related_pairs(edges, unit_to_fine, k=max_related)

    if os.getenv("TOPIC_GRAPH_FOUR_TIER", "true").strip().lower() in {"1", "true", "yes", "on"}:
        three_tier = _mindmap_lecture_theme_sub_three(raw, by_id, items, hierarchy, tier_summaries)
        if three_tier is not None:
            nodes_ft, links_ft = three_tier
            _append_global_timeline(items, links_ft)
            for fa, fb, w in related:
                links_ft.append(
                    {
                        "source": f"c-{fa}",
                        "target": f"c-{fb}",
                        "type": "related",
                        "weight": round(float(w), 4),
                    }
                )
            return {"nodes": nodes_ft, "links": links_ft}

    coarse_lvl = _coarser_hierarchy_level(hierarchy, len(items))
    coarse_comms = coarse_lvl.get("communities") if isinstance(coarse_lvl, dict) else None
    use_hierarchy = (
        isinstance(coarse_lvl, dict)
        and isinstance(coarse_comms, list)
        and len(coarse_comms) >= 2
        and len(items) >= 2
    )

    nodes: list[dict[str, Any]] = []
    links: list[dict[str, Any]] = []

    MACRO_X, LEAF_X = 36.0, 420.0
    MACRO_STEP = 240.0
    LEAF_DY = 92.0

    if use_hierarchy and coarse_comms is not None:
        parent_of_fine = _assign_fines_to_coarse_macroids(items, coarse_lvl)  # type: ignore[arg-type]
        macro_order = sorted(
            range(len(coarse_comms)),
            key=lambda ci: _first_time(_community_member_ids(coarse_comms[ci]), by_id),  # type: ignore[index]
        )
        children_by_macro: dict[int, list[int]] = {int(ci): [] for ci in macro_order}
        for fi, it in enumerate(items):
            pci = parent_of_fine[fi]
            children_by_macro.setdefault(int(pci), []).append(fi)
        for ci in macro_order:
            iis = children_by_macro.get(int(ci), [])
            iis.sort(key=lambda ii: _first_time(items[ii]["ids"], by_id))
            children_by_macro[int(ci)] = iis

        macro_meta: list[tuple[int, str, float]] = []
        macro_mids_order: list[str] = []
        for mi, ci in enumerate(macro_order):
            y_m = 40.0 + float(mi) * MACRO_STEP
            mids = _community_member_ids(coarse_comms[ci])
            child_iis = children_by_macro.get(int(ci), [])
            lbl, summ = _macro_block_text(child_iis, items, mids, by_id, mi)
            mid = f"macro-{int(ci)}"
            macro_mids_order.append(mid)
            nodes.append(
                {
                    "id": mid,
                    "label": lbl,
                    "summary": summ,
                    "community": int(ci),
                    "kind": "macro",
                    "parentId": None,
                    "position": {"x": MACRO_X, "y": y_m},
                }
            )
            macro_meta.append((ci, mid, y_m))

        _append_macro_timeline(macro_mids_order, links)

        for _ci, mid, y_m in macro_meta:
            child_iis = children_by_macro.get(int(_ci), [])
            nch = len(child_iis)
            for j, ii in enumerate(child_iis):
                it = items[ii]
                offset = (j - (nch - 1) / 2.0) * LEAF_DY if nch > 1 else 0.0
                y = y_m + offset
                cid = int(it["id"])
                fid = f"c-{cid}"
                lbl = (
                    str(it["name"]).strip()
                    if isinstance(it["name"], str) and str(it["name"]).strip()
                    else f"Тема {cid + 1}"
                )
                sm = it["summary"]
                summary = (
                    str(sm).strip() if isinstance(sm, str) and str(sm).strip() else _snippet(it["ids"], by_id)
                )
                kws = it.get("keywords") if isinstance(it.get("keywords"), list) else []
                kw_safe = [str(x).strip() for x in kws if str(x).strip()][:24]
                nodes.append(
                    {
                        "id": fid,
                        "label": lbl,
                        "summary": summary,
                        "keywords": kw_safe,
                        "community": cid,
                        "kind": "topic",
                        "parentId": mid,
                        "position": {"x": LEAF_X, "y": y},
                    }
                )
                links.append({"source": mid, "target": fid, "type": "parent"})
    else:
        ncol = min(4, max(2, int(len(items) ** 0.5 + 0.999)))
        for i, it in enumerate(items):
            lbl = (
                str(it["name"]).strip()
                if isinstance(it["name"], str) and str(it["name"]).strip()
                else f"Тема {it['id'] + 1}"
            )
            sm = it["summary"]
            summary = (
                str(sm).strip() if isinstance(sm, str) and str(sm).strip() else _snippet(it["ids"], by_id)
            )
            cid = int(it["id"])
            ikws = it.get("keywords") if isinstance(it.get("keywords"), list) else []
            kw_flat = [str(x).strip() for x in ikws if str(x).strip()][:24]
            nodes.append(
                {
                    "id": f"c-{cid}",
                    "label": lbl,
                    "summary": summary,
                    "keywords": kw_flat,
                    "community": cid,
                    "kind": "topic",
                    "parentId": None,
                    "position": _grid(i, ncol),
                }
            )

    _append_global_timeline(items, links)

    for fa, fb, w in related:
        links.append(
            {
                "source": f"c-{fa}",
                "target": f"c-{fb}",
                "type": "related",
                "weight": round(float(w), 4),
            }
        )

    return {"nodes": nodes, "links": links}


def _from_units(raw_nodes: list[Any]) -> dict[str, Any]:
    pairs: list[tuple[float, int, dict[str, Any]]] = []
    for j, x in enumerate(raw_nodes):
        if isinstance(x, dict):
            try:
                ts = float(x["timestamp"]) if x.get("timestamp") is not None else float(j)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                ts = float(j)
            pairs.append((ts if ts == ts else float(j), j, x))
    pairs.sort(key=lambda p: p[0])
    nodes: list[dict[str, Any]] = []
    for i, (_, _, u) in enumerate(pairs):
        try:
            sid = str(int(u["id"]))  # type: ignore[arg-type]
        except (KeyError, TypeError, ValueError):
            sid = f"u-{i}"
        line, text = str(u.get("line") or "").strip(), str(u.get("text") or "").strip()
        raw = text or line
        summary = raw if len(raw) <= 240 else raw[:237] + "…"
        label = line if len(line) <= 72 else line[:69] + "…"
        if not label:
            label = f"Фрагмент {i + 1}"
        nodes.append(
            {
                "id": sid,
                "label": label,
                "summary": summary,
                "community": i % 12,
                "kind": "topic",
                "parentId": None,
            }
        )
    nodes = _grid_layout(nodes)
    ln = len(nodes)
    links = (
        [{"source": nodes[i]["id"], "target": nodes[i + 1]["id"], "type": "follows"} for i in range(ln - 1)]
        if ln > 1
        else []
    )
    return {"nodes": nodes, "links": links}


def unit_graph_to_mindmap_payload(
    unit_graph: dict[str, Any] | None,
    tier_summaries: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if not isinstance(unit_graph, dict):
        return None
    raw_nodes = unit_graph.get("nodes")
    if not isinstance(raw_nodes, list) or not raw_nodes:
        return None
    cm = _from_communities(unit_graph, tier_summaries=tier_summaries)
    return cm if cm else _from_units(raw_nodes)
