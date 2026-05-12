from __future__ import annotations

import hashlib
import math
import os
import re
import threading
import time
from typing import Any, TypedDict

import faiss
import igraph as ig
import leidenalg
import numpy as np

from whisper_video_summarization.llm import qa_rag
from whisper_video_summarization.utils.observability import observe_embeddings_batch


UNIT_GRAPH_CACHE_VERSION = "v7"


class SemanticUnit(TypedDict, total=False):
    unit_id: int
    speaker: str
    t0: float | None
    t1: float | None
    text: str
    line: str
    char_start: int
    char_end: int
    embedding_key: str
    community_id: int | None
    chunk_id: int | None


class UnitGraphBuildResult(TypedDict, total=False):
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    communities: list[dict[str, Any]]
    semantic_units: list[SemanticUnit]
    unit_text: str
    hierarchy: list[dict[str, Any]]
    # order-preserving texts of communities at target resolution
    community_texts: list[dict[str, Any]]  # {id:int, t0:float, body:str}
    cache_key: str


_embed_cache_lock = threading.Lock()
_embed_cache: dict[str, list[float]] = {}

_graph_cache_lock = threading.Lock()
_graph_cache: dict[str, UnitGraphBuildResult] = {}


def _env_bool(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _unit_graph_label_level() -> str:
    """На каком уровне γ строить `communities` / LLM-подписи: `finest` (максимум подтем) или `mid` (крупнее блоки)."""
    raw = os.getenv("UNIT_GRAPH_LABEL_LEVEL", "finest").strip().lower()
    if raw == "finest":
        return "finest"
    return "mid"


def stable_text_hash(text: str) -> str:
    t = re.sub(r"\s+", " ", str(text or "")).strip()
    return hashlib.sha1(t.encode("utf-8", errors="ignore")).hexdigest()


def stable_embedding_key(text: str) -> str:
    t = re.sub(r"\s+", " ", str(text or "")).strip()
    return f"u{len(t)}_{stable_text_hash(t)[:16]}"


def speaker_from_unit_line(line: str) -> str:
    m = re.match(r"^\s*([^:]{1,80})\s*:\s*(.*)$", str(line or ""))
    if not m:
        return "Unknown"
    sp = (m.group(1) or "").strip()
    return sp or "Unknown"


def text_from_unit_line(line: str) -> str:
    m = re.match(r"^\s*([^:]{1,80})\s*:\s*(.*)$", str(line or ""))
    if not m:
        return str(line or "").strip()
    return str(m.group(2) or "").strip()


def build_semantic_units(units: list[dict[str, Any]]) -> tuple[list[SemanticUnit], str]:
    out: list[SemanticUnit] = []
    pieces: list[str] = []
    cursor = 0
    for i, u in enumerate(units):
        line = str(u.get("line") or "").strip()
        sp = speaker_from_unit_line(line)
        txt = text_from_unit_line(line)
        t0 = u.get("start")
        t1 = u.get("end")
        try:
            t0f = float(t0) if t0 is not None else None
        except Exception:
            t0f = None
        try:
            t1f = float(t1) if t1 is not None else None
        except Exception:
            t1f = None
        piece = line if line else f"{sp}: {txt}".strip()
        pieces.append(piece)
        start = cursor
        end = cursor + len(piece)
        cursor = end + 1
        out.append(
            {
                "unit_id": int(i),
                "speaker": sp,
                "t0": t0f,
                "t1": t1f,
                "text": txt,
                "line": piece,
                "char_start": int(start),
                "char_end": int(end),
                "embedding_key": stable_embedding_key(piece),
                "community_id": None,
                "chunk_id": None,
            }
        )
    return out, "\n".join(pieces)


def _get_embed_model_name() -> str:
    return os.getenv("RAG_EMBEDDING_MODEL_NAME", "BAAI/bge-m3").strip() or "BAAI/bge-m3"


def embed_texts_with_cache(texts: list[str], keys: list[str]) -> list[list[float]]:
    """Embed texts using shared HF embedder, with in-process cache keyed by (model,texthash)."""
    if not texts:
        return []
    model_name = _get_embed_model_name()
    # If keys aren't provided, derive from text.
    if len(keys) != len(texts):
        keys = [stable_embedding_key(t) for t in texts]

    cache_keys = [f"{model_name}:{k}" for k in keys]
    out: list[list[float] | None] = [None] * len(texts)
    missing_texts: list[str] = []
    missing_idx: list[int] = []
    with _embed_cache_lock:
        for i, ck in enumerate(cache_keys):
            v = _embed_cache.get(ck)
            if v is not None:
                out[i] = list(v)
            else:
                missing_texts.append(texts[i])
                missing_idx.append(i)

    if missing_texts:
        m = qa_rag.get_embed_model()
        t0 = time.perf_counter()
        batch = m.get_text_embedding_batch(missing_texts)
        observe_embeddings_batch(
            duration_seconds=time.perf_counter() - t0,
            stage="unit_graph_faiss",
        )
        for local_i, i in enumerate(missing_idx):
            vec = list(batch[local_i])
            out[i] = vec
            with _embed_cache_lock:
                _embed_cache[cache_keys[i]] = vec

        max_items = max(0, int(os.getenv("EMBEDDING_CACHE_MAX", "40000")))
        if max_items > 0:
            with _embed_cache_lock:
                while len(_embed_cache) > max_items:
                    _embed_cache.pop(next(iter(_embed_cache)))

    return [x if x is not None else [] for x in out]


def _temporal_proximity(a_ts: float, b_ts: float, tau: float) -> float:
    d = abs(float(a_ts) - float(b_ts))
    t = max(1e-6, float(tau))
    return float(math.exp(-d / t))


def build_unit_knn_graph(*, nodes: list[dict[str, Any]], embs_norm: np.ndarray) -> dict[str, Any]:
    n, dim = embs_norm.shape
    if n <= 1:
        return {"nodes": nodes, "edges": []}

    # Чуть плотнее kNN → Leiden лучше отделяет темы / подтемы / микротемы на разных γ.
    top_k = max(2, int(os.getenv("UNIT_GRAPH_TOP_K", "10")))
    hnsw_m = max(8, int(os.getenv("UNIT_GRAPH_HNSW_M", "40")))
    ef_search = max(16, int(os.getenv("UNIT_GRAPH_HNSW_EF_SEARCH", "80")))
    min_sim = float(os.getenv("UNIT_GRAPH_MIN_SEMANTIC_SIM", "0.48"))

    w_sem = float(os.getenv("UNIT_GRAPH_W_SEMANTIC", "0.55"))
    w_tmp = float(os.getenv("UNIT_GRAPH_W_TEMPORAL", "0.2"))
    w_spk = float(os.getenv("UNIT_GRAPH_W_SPEAKER", "0.1"))
    tau = float(os.getenv("UNIT_GRAPH_TEMPORAL_TAU", "90.0"))
    same_speaker_boost = float(os.getenv("UNIT_GRAPH_SAME_SPEAKER_BOOST", "0.1"))

    wsum = abs(w_sem) + abs(w_tmp) + abs(w_spk)
    if wsum <= 1e-9:
        w_sem, w_tmp, w_spk = 1.0, 0.0, 0.0
    else:
        w_sem, w_tmp, w_spk = w_sem / wsum, w_tmp / wsum, w_spk / wsum

    index = faiss.IndexHNSWFlat(dim, hnsw_m)
    index.hnsw.efSearch = ef_search
    index.metric_type = faiss.METRIC_INNER_PRODUCT
    index.add(embs_norm.astype(np.float32, copy=False))

    D, I = index.search(embs_norm.astype(np.float32, copy=False), top_k + 1)

    edges_map: dict[tuple[int, int], dict[str, Any]] = {}

    def _add_edge(i: int, j: int, semantic: float, force: bool = False) -> None:
        if i == j:
            return
        a, b = (i, j) if i < j else (j, i)
        if not force and semantic < min_sim:
            return
        a_ts = float(nodes[a]["timestamp"])
        b_ts = float(nodes[b]["timestamp"])
        temporal = _temporal_proximity(a_ts, b_ts, tau)
        same = str(nodes[a]["speaker"]) == str(nodes[b]["speaker"])
        speaker = (same_speaker_boost if same else 0.0)
        score = w_sem * float(semantic) + w_tmp * float(temporal) + w_spk * float(speaker)
        cur = edges_map.get((a, b))
        if cur is None or score > float(cur.get("weight", 0.0)):
            edges_map[(a, b)] = {
                "source": int(a),
                "target": int(b),
                "weight": float(score),
                "semantic": float(semantic),
                "temporal": float(temporal),
                "speaker": float(speaker),
            }

    for i in range(n):
        for j, sim in zip(I[i], D[i]):
            j = int(j)
            if j < 0 or j >= n or j == i:
                continue
            _add_edge(i, j, float(sim), force=False)

    for i in range(n - 1):
        sem = float(np.dot(embs_norm[i], embs_norm[i + 1]))
        _add_edge(i, i + 1, sem, force=True)

    return {"nodes": nodes, "edges": list(edges_map.values())}


def leiden_communities(*, n_nodes: int, edges: list[dict[str, Any]], resolution: float) -> list[list[int]]:
    if n_nodes <= 0:
        return []
    if not edges:
        return [[i] for i in range(n_nodes)]
    tuples = [(int(e["source"]), int(e["target"])) for e in edges]
    weights = [float(e.get("weight", 0.0)) for e in edges]
    g = ig.Graph(n=n_nodes, edges=tuples, directed=False)
    g.es["weight"] = weights
    part = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=float(resolution),
        seed=42,
    )
    comms: dict[int, list[int]] = {}
    for vid, cid in enumerate(part.membership):
        comms.setdefault(int(cid), []).append(int(vid))
    return [comms[k] for k in sorted(comms.keys())]


def build_multires_hierarchy(
    *,
    n_nodes: int,
    edges: list[dict[str, Any]],
    resolutions: list[float],
) -> list[dict[str, Any]]:
    res_list = sorted(set(float(r) for r in resolutions)) or [1.0]
    hierarchy: list[dict[str, Any]] = []
    prev: list[list[int]] | None = None
    for r in res_list:
        level = leiden_communities(n_nodes=n_nodes, edges=edges, resolution=r)
        payload: dict[str, Any] = {
            "resolution": float(r),
            "communities": [{"id": i, "nodes": c} for i, c in enumerate(level)],
        }
        if prev is not None:
            prev_sets = [set(c) for c in prev]
            cur_sets = [set(c) for c in level]
            parent_map: dict[int, int] = {}
            for ci, cset in enumerate(cur_sets):
                best_pi = 0
                best_j = -1.0
                for pi, pset in enumerate(prev_sets):
                    inter = len(cset & pset)
                    uni = len(cset | pset) or 1
                    j = inter / uni
                    if j > best_j:
                        best_j = j
                        best_pi = pi
                parent_map[ci] = int(best_pi)
            payload["parent_map"] = parent_map
        hierarchy.append(payload)
        prev = level
    return hierarchy


def _graph_params_fingerprint() -> str:
    keys = [
        "UNIT_GRAPH_TOP_K",
        "UNIT_GRAPH_HNSW_M",
        "UNIT_GRAPH_HNSW_EF_SEARCH",
        "UNIT_GRAPH_MIN_SEMANTIC_SIM",
        "UNIT_GRAPH_W_SEMANTIC",
        "UNIT_GRAPH_W_TEMPORAL",
        "UNIT_GRAPH_W_SPEAKER",
        "UNIT_GRAPH_TEMPORAL_TAU",
        "UNIT_GRAPH_SAME_SPEAKER_BOOST",
        "UNIT_GRAPH_LEIDEN_RESOLUTIONS",
        "UNIT_GRAPH_LEIDEN_TARGET_RESOLUTION",
        "UNIT_GRAPH_LEIDEN_LABEL_FINEST",
        "UNIT_GRAPH_LABEL_LEVEL",
    ]
    parts = [f"{k}={os.getenv(k,'')}" for k in keys]
    parts.append(f"embed={_get_embed_model_name()}")
    parts.append(f"v={UNIT_GRAPH_CACHE_VERSION}")
    return stable_text_hash("|".join(parts))[:16]


def make_graph_cache_key(*, lecture_id: str | None, unit_text: str) -> str:
    lid = (lecture_id or "").strip() or "-"
    return f"{lid}:{stable_text_hash(unit_text)[:16]}:{_graph_params_fingerprint()}"


def build_unit_graph(
    *,
    units: list[dict[str, Any]],
    lecture_id: str | None = None,
) -> UnitGraphBuildResult:
    semantic_units, unit_text = build_semantic_units(units)
    cache_key = make_graph_cache_key(lecture_id=lecture_id, unit_text=unit_text)

    with _graph_cache_lock:
        cached = _graph_cache.get(cache_key)
    if cached:
        return dict(cached)

    nodes: list[dict[str, Any]] = [
        {
            "id": int(u["unit_id"]),
            "speaker": str(u["speaker"]),
            "timestamp": float(u["t0"]) if u.get("t0") is not None else float(u["unit_id"]),
            "t0": u.get("t0"),
            "t1": u.get("t1"),
            "text": str(u.get("text") or ""),
            "line": str(u.get("line") or ""),
            "char_start": int(u.get("char_start", 0)),
            "char_end": int(u.get("char_end", 0)),
            "embedding_key": str(u.get("embedding_key") or ""),
        }
        for u in semantic_units
    ]
    lines = [str(u["line"]) for u in semantic_units]
    keys = [str(u["embedding_key"]) for u in semantic_units]
    embs = embed_texts_with_cache(lines, keys)
    embs_np = np.asarray(embs, dtype=np.float32)
    if embs_np.ndim != 2 or embs_np.shape[0] != len(nodes) or embs_np.shape[0] == 0:
        out: UnitGraphBuildResult = {
            "nodes": nodes,
            "edges": [],
            "communities": [],
            "semantic_units": semantic_units,
            "unit_text": unit_text,
            "hierarchy": [],
            "community_texts": [],
            "cache_key": cache_key,
        }
        return out
    norms = np.linalg.norm(embs_np, axis=1, keepdims=True) + 1e-12
    embs_norm = embs_np / norms

    graph = build_unit_knn_graph(nodes=nodes, embs_norm=embs_norm)
    edges = graph["edges"]

    # Три уровня γ (sorted по возрастанию: темы→подтемы по слоям; выше последний γ → дробнее листья).
    res_raw = os.getenv("UNIT_GRAPH_LEIDEN_RESOLUTIONS", "0.10,0.42,1.85")
    res_list: list[float] = []
    for p in res_raw.split(","):
        p = p.strip()
        if not p:
            continue
        try:
            res_list.append(float(p))
        except ValueError:
            continue
    res_list = res_list or [1.0]
    hierarchy = build_multires_hierarchy(n_nodes=len(nodes), edges=edges, resolutions=res_list)

    avail_levels = sorted(
        [x for x in hierarchy if isinstance(x, dict)],
        key=lambda x: float(x.get("resolution", 0.0)),
    )
    lvl = _unit_graph_label_level()
    if lvl == "mid" and len(avail_levels) >= 2:
        target_level = avail_levels[-2]
    elif avail_levels:
        target_level = avail_levels[-1]
    elif hierarchy:
        target_level = hierarchy[-1]
    else:
        target_level = None
    comms = [c["nodes"] for c in (target_level["communities"] if target_level else [])]

    unit_to_comm: dict[int, int] = {}
    for cid, node_ids in enumerate(comms):
        for uid in node_ids:
            unit_to_comm[int(uid)] = int(cid)
    for u in semantic_units:
        cid = unit_to_comm.get(int(u["unit_id"]))
        u["community_id"] = cid
        u["chunk_id"] = cid

    communities = [{"id": i, "nodes": c} for i, c in enumerate(comms)]

    comm_items: list[dict[str, Any]] = []
    for cid, node_ids in enumerate(comms):
        if not node_ids:
            continue
        node_ids = sorted(node_ids, key=lambda k: float(nodes[k]["timestamp"]))
        body = "\n".join(nodes[k]["line"] for k in node_ids if str(nodes[k]["line"]).strip())
        if not body.strip():
            continue
        t0 = min(float(nodes[k]["timestamp"]) for k in node_ids)
        comm_items.append({"id": int(cid), "t0": float(t0), "body": body})
    comm_items.sort(key=lambda x: x["t0"])

    out = {
        "nodes": nodes,
        "edges": edges,
        "communities": communities,
        "semantic_units": semantic_units,
        "unit_text": unit_text,
        "hierarchy": hierarchy,
        "community_texts": comm_items,
        "cache_key": cache_key,
    }

    max_items = max(0, int(os.getenv("UNIT_GRAPH_CACHE_MAX", "16")))
    if max_items > 0:
        with _graph_cache_lock:
            _graph_cache[cache_key] = dict(out)
            while len(_graph_cache) > max_items:
                _graph_cache.pop(next(iter(_graph_cache)))
    return out

