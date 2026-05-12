from __future__ import annotations

import math
import os
import re
import threading
import time
from collections import defaultdict
from typing import Any

from qdrant_client import QdrantClient

from whisper_video_summarization.utils.observability import (
    observe_embeddings_batch,
    observe_rag_retrieval,
)

_qdrant_singleton: QdrantClient | None = None
_qdrant_singleton_lock = threading.Lock()

_llama_lock = threading.Lock()
_llama_bundle: dict[str, Any] | None = None
_embed_model_singleton: Any = None
_mock_llm_singleton: Any = None


def _embeddings_serve_url() -> str | None:
    raw = os.getenv("RAG_EMBEDDINGS_SERVE_URL", "").strip().rstrip("/")
    return raw or None


def _openai_embeddings_api_base(serve_url: str) -> str:
    base = serve_url.strip().rstrip("/")
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    return base


def get_qdrant_client() -> QdrantClient:
    url = os.getenv("QDRANT_URL", "http://localhost:6333")
    api_key = os.getenv("QDRANT_API_KEY")
    timeout = float(os.getenv("QDRANT_TIMEOUT_SECONDS", "30"))
    if url == ":memory:":
        return QdrantClient(path=":memory:")
    return QdrantClient(url=url, api_key=api_key, timeout=timeout)


def get_qdrant_client_singleton() -> QdrantClient:
    global _qdrant_singleton
    if _qdrant_singleton is not None:
        return _qdrant_singleton
    with _qdrant_singleton_lock:
        if _qdrant_singleton is None:
            _qdrant_singleton = get_qdrant_client()
    return _qdrant_singleton


def qdrant_collection_base() -> str:
    return os.getenv("QDRANT_COLLECTION_NAME", "lectures").strip() or "lectures"


def collection_for_lecture(lecture_id: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", lecture_id)[:72]
    return f"{qdrant_collection_base()}_{safe}"


def is_hierarchical_chunking() -> bool:
    return os.getenv("RAG_USE_HIERARCHICAL_CHUNKING", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def get_llama_bundle() -> dict[str, Any]:
    global _llama_bundle
    if _llama_bundle is not None:
        return _llama_bundle
    with _llama_lock:
        if _llama_bundle is not None:
            return _llama_bundle
        from llama_index.core import Document, QueryBundle, Settings, StorageContext, VectorStoreIndex
        from llama_index.core.llms.mock import MockLLM
        from llama_index.core.node_parser import HierarchicalNodeParser, SentenceSplitter
        from llama_index.core.retrievers import QueryFusionRetriever
        from llama_index.retrievers.bm25 import BM25Retriever
        from llama_index.vector_stores.qdrant import QdrantVectorStore

        _llama_bundle = {
            "Document": Document,
            "MockLLM": MockLLM,
            "QueryBundle": QueryBundle,
            "Settings": Settings,
            "StorageContext": StorageContext,
            "VectorStoreIndex": VectorStoreIndex,
            "HierarchicalNodeParser": HierarchicalNodeParser,
            "SentenceSplitter": SentenceSplitter,
            "QueryFusionRetriever": QueryFusionRetriever,
            "BM25Retriever": BM25Retriever,
            "QdrantVectorStore": QdrantVectorStore,
        }
        return _llama_bundle


def get_embed_model() -> Any:
    global _embed_model_singleton
    if _embed_model_singleton is not None:
        return _embed_model_singleton
    with _llama_lock:
        if _embed_model_singleton is not None:
            return _embed_model_singleton
        serve_url = _embeddings_serve_url()
        if not serve_url:
            raise RuntimeError(
                "RAG_EMBEDDINGS_SERVE_URL is not set. Local HuggingFace embedding models are disabled; "
                "point this to your OpenAI-compatible embeddings HTTP service (e.g. Ray Serve /embeddings)."
            )
        embed_model_name = os.getenv("RAG_EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
        from llama_index.embeddings.openai import OpenAIEmbedding, OpenAIEmbeddingModelType

        api_base = _openai_embeddings_api_base(serve_url)
        api_key = os.getenv("RAG_EMBEDDINGS_OPENAI_API_KEY", "local-embeddings")
        timeout = float(os.getenv("RAG_EMBEDDINGS_SERVE_TIMEOUT_SECONDS", "300"))
        _embed_model_singleton = OpenAIEmbedding(
            model=OpenAIEmbeddingModelType.TEXT_EMBED_3_SMALL,
            model_name=embed_model_name,
            api_key=api_key,
            api_base=api_base,
            timeout=timeout,
        )
        return _embed_model_singleton


def get_mock_llm() -> Any:
    global _mock_llm_singleton
    if _mock_llm_singleton is not None:
        return _mock_llm_singleton
    with _llama_lock:
        if _mock_llm_singleton is not None:
            return _mock_llm_singleton
        _mock_llm_singleton = get_llama_bundle()["MockLLM"]()
        return _mock_llm_singleton


def configure_llama_settings() -> None:
    L = get_llama_bundle()
    L["Settings"].llm = get_mock_llm()
    L["Settings"].embed_model = get_embed_model()


def dot_vec(a: list[float], b: list[float]) -> float:
    return float(sum(x * y for x, y in zip(a, b)))


def normalize_vec(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) + 1e-12
    return [x / n for x in v]


def mmr_select_chunk_ids(
    query_emb: list[float],
    candidate_ids: list[int],
    id_to_emb: dict[int, list[float]],
    top_k: int,
    lambda_mult: float | None = None,
) -> list[int]:
    if not candidate_ids:
        return []
    lam = lambda_mult if lambda_mult is not None else float(os.getenv("RAG_MMR_LAMBDA", "0.7"))
    qn = normalize_vec(query_emb)
    remaining = [cid for cid in candidate_ids if cid in id_to_emb]
    if not remaining:
        return candidate_ids[:top_k]
    selected: list[int] = []
    while len(selected) < top_k and remaining:
        best_id: int | None = None
        best_score = -1e9
        for cid in remaining:
            dn = normalize_vec(id_to_emb[cid])
            sim_q = dot_vec(qn, dn)
            if not selected:
                mmr = sim_q
            else:
                sim_d = max(dot_vec(dn, normalize_vec(id_to_emb[s])) for s in selected)
                mmr = lam * sim_q - (1.0 - lam) * sim_d
            if mmr > best_score:
                best_score = mmr
                best_id = cid
        if best_id is None:
            break
        selected.append(best_id)
        remaining.remove(best_id)
    return selected


def semantic_neighbor_ids(
    chunk_id: int,
    id_to_emb: dict[int, list[float]],
    n_chunks: int,
    k: int,
) -> list[int]:
    if k <= 0 or chunk_id not in id_to_emb:
        return []
    base = normalize_vec(id_to_emb[chunk_id])
    sims: list[tuple[int, float]] = []
    for j in range(n_chunks):
        if j == chunk_id or j not in id_to_emb:
            continue
        sims.append((j, dot_vec(base, normalize_vec(id_to_emb[j]))))
    sims.sort(key=lambda x: -x[1])
    return [j for j, _ in sims[:k]]


def coerce_qdrant_vector(vector: Any) -> list[float]:
    if vector is None:
        return []
    if isinstance(vector, dict):
        if not vector:
            return []
        if len(vector) == 1:
            v = next(iter(vector.values()))
            return [float(x) for x in list(v)]
        for key in ("text-dense", ""):
            if key in vector and vector[key] is not None:
                return [float(x) for x in list(vector[key])]
        v = next(iter(vector.values()))
        return [float(x) for x in list(v)]
    return [float(x) for x in list(vector)]


def qdrant_collection_exists(client: QdrantClient, name: str) -> bool:
    try:
        return any(c.name == name for c in client.get_collections().collections)
    except Exception:
        return False


class RAGNotIndexedError(RuntimeError):
    """Для lecture_id ещё нет коллекции Qdrant (индекс RAG не построен)."""


def rag_indexing_enabled() -> bool:
    return os.getenv("RAG_INDEXING_ENABLED", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _build_chunk_nodes(full_transcript: str, lecture_id: str) -> tuple[list[Any], list[str], int]:
    L = get_llama_bundle()
    chunk_size = int(os.getenv("RAG_CHUNK_SIZE", "512"))
    chunk_overlap = int(os.getenv("RAG_CHUNK_OVERLAP", "50"))
    child_chunk = int(os.getenv("RAG_CHILD_CHUNK_SIZE", "150"))
    Document = L["Document"]
    HierarchicalNodeParser = L["HierarchicalNodeParser"]
    SentenceSplitter = L["SentenceSplitter"]

    documents = [Document(text=full_transcript, metadata={"lecture_id": lecture_id})]
    if is_hierarchical_chunking():
        parser = HierarchicalNodeParser.from_defaults(chunk_sizes=[chunk_size, child_chunk])
    else:
        parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    nodes = parser.get_nodes_from_documents(documents)
    for i, node in enumerate(nodes):
        node.metadata["chunk_id"] = i

    n = len(nodes)
    texts = [""] * n
    for node in nodes:
        cid = int(node.metadata.get("chunk_id", -1))
        if 0 <= cid < n:
            texts[cid] = node.get_content()
    return nodes, texts, n


def _attach_embeddings_to_nodes(nodes: list[Any], texts: list[str], n: int) -> dict[int, list[float]]:
    embed_model = get_embed_model()
    id_to_emb: dict[int, list[float]] = {}
    contents = [texts[i] for i in range(n) if texts[i].strip()]
    if contents:
        idx_map = [i for i in range(n) if texts[i].strip()]
        t0 = time.perf_counter()
        batch_embs = embed_model.get_text_embedding_batch(contents)
        observe_embeddings_batch(duration_seconds=time.perf_counter() - t0, stage="rag_index")
        for local_i, global_i in enumerate(idx_map):
            id_to_emb[global_i] = list(batch_embs[local_i])

    dim = len(next(iter(id_to_emb.values()))) if id_to_emb else 0
    zero = [0.0] * dim if dim else []
    for node in nodes:
        cid = int(node.metadata.get("chunk_id", -1))
        if 0 <= cid < n and texts[cid].strip():
            emb = id_to_emb.get(cid)
            if emb is not None:
                node.embedding = emb
        elif dim:
            node.embedding = list(zero)
    return id_to_emb


def index_full_transcript_to_qdrant_sync(full_transcript: str, lecture_id: str) -> int:
    """Перезаписывает коллекцию Qdrant для lecture_id текущим текстом транскрипта. Возвращает число чанков."""
    if not rag_indexing_enabled():
        return 0

    raw = str(full_transcript or "").strip()
    if not raw:
        return 0

    L = get_llama_bundle()
    configure_llama_settings()
    StorageContext = L["StorageContext"]
    VectorStoreIndex = L["VectorStoreIndex"]
    QdrantVectorStore = L["QdrantVectorStore"]

    nodes, texts, n = _build_chunk_nodes(raw, lecture_id)
    if not n:
        return 0

    _attach_embeddings_to_nodes(nodes, texts, n)

    qclient = get_qdrant_client_singleton()
    collection_name = collection_for_lecture(lecture_id)
    try:
        if qdrant_collection_exists(qclient, collection_name):
            qclient.delete_collection(collection_name)
    except Exception:
        pass

    vector_store = QdrantVectorStore(client=qclient, collection_name=collection_name)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        show_progress=False,
    )
    return n


def scroll_lecture_chunks_from_qdrant_sync(
    lecture_id: str,
) -> tuple[int, list[str], dict[int, list[float]], list[Any]]:
    """Читает все точки коллекции, восстанавливает тексты чанков и узлы для BM25."""
    from llama_index.core.schema import TextNode
    from llama_index.core.vector_stores.utils import metadata_dict_to_node

    client = get_qdrant_client_singleton()
    collection_name = collection_for_lecture(lecture_id)
    if not qdrant_collection_exists(client, collection_name):
        raise RAGNotIndexedError(collection_name)

    nodes_by_cid: dict[int, Any] = {}
    id_to_emb: dict[int, list[float]] = {}
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection_name,
            limit=512,
            offset=offset,
            with_vectors=True,
            with_payload=True,
        )
        for rec in points:
            pl = dict(rec.payload or {})
            vec = coerce_qdrant_vector(getattr(rec, "vector", None))
            try:
                node = metadata_dict_to_node(pl)
            except Exception:
                continue
            cid_raw = node.metadata.get("chunk_id")
            if cid_raw is None:
                continue
            try:
                cid = int(cid_raw)
            except (TypeError, ValueError):
                continue
            nodes_by_cid[cid] = node
            if vec:
                id_to_emb[cid] = vec
        if offset is None:
            break

    if not nodes_by_cid:
        raise RAGNotIndexedError(collection_name)

    max_cid = max(nodes_by_cid.keys())
    n = max_cid + 1
    dim = len(next(iter(id_to_emb.values()))) if id_to_emb else 0
    zero = [0.0] * dim if dim else []
    texts: list[str] = [""] * n
    bm25_nodes: list[Any] = []
    for i in range(n):
        stored = nodes_by_cid.get(i)
        if stored is not None:
            t = stored.get_content()
            texts[i] = t
            emb = id_to_emb.get(i) if str(t).strip() else None
            bm25_nodes.append(
                TextNode(
                    text=t,
                    metadata={"chunk_id": i},
                    embedding=list(emb) if emb else (list(zero) if dim else None),
                )
            )
        else:
            bm25_nodes.append(
                TextNode(
                    text="",
                    metadata={"chunk_id": i},
                    embedding=list(zero) if dim else None,
                )
            )
    return n, texts, id_to_emb, bm25_nodes


def list_chunk_embeddings_from_qdrant_sync(lecture_id: str) -> list[dict[str, Any]]:
    """Сырые эмбеддинги по chunk_id из Qdrant (для задач worker)."""
    _n, _texts, id_to_emb, _bm25_nodes = scroll_lecture_chunks_from_qdrant_sync(lecture_id)
    rows: list[dict[str, Any]] = []
    for cid in sorted(id_to_emb.keys()):
        rows.append({"chunk_id": cid, "embedding": id_to_emb[cid]})
    return rows


def retrieve_chunk_ids_stored_sync(
    lecture_id: str,
    queries: list[str],
) -> tuple[list[int], list[str], dict[int, list[int]]]:
    """Поиск по уже сохранённой коллекции (без перестроения индекса)."""
    t0 = time.perf_counter()
    try:
        L = get_llama_bundle()
        configure_llama_settings()

        overfetch_k = max(5, int(os.getenv("RAG_OVERFETCH_K", "30")))
        top_k = max(1, int(os.getenv("RAG_TOP_K", "8")))
        mmr_pool = max(top_k, int(os.getenv("RAG_MMR_CANDIDATE_POOL", str(max(top_k, overfetch_k // 2)))))
        mode = os.getenv("RAG_RETRIEVAL_MODE", "hybrid").strip().lower()
        sem_neighbor_k = max(0, int(os.getenv("RAG_SEMANTIC_NEIGHBOR_K", "2")))
        use_semantic_neighbors = os.getenv("RAG_USE_SEMANTIC_NEIGHBORS", "true").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

        QueryBundle = L["QueryBundle"]
        VectorStoreIndex = L["VectorStoreIndex"]
        QueryFusionRetriever = L["QueryFusionRetriever"]
        BM25Retriever = L["BM25Retriever"]
        QdrantVectorStore = L["QdrantVectorStore"]

        embed_model = get_embed_model()
        client = get_qdrant_client_singleton()
        collection_name = collection_for_lecture(lecture_id)

        n, texts, id_to_emb, bm25_nodes = scroll_lecture_chunks_from_qdrant_sync(lecture_id)

        vector_store = QdrantVectorStore(client=client, collection_name=collection_name)
        index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
        dense_retriever = index.as_retriever(similarity_top_k=overfetch_k)

        if mode == "hybrid":
            bm25_retriever = BM25Retriever.from_defaults(
                nodes=list(bm25_nodes),
                similarity_top_k=overfetch_k,
            )
            retriever = QueryFusionRetriever(
                retrievers=[dense_retriever, bm25_retriever],
                similarity_top_k=overfetch_k,
                num_queries=1,
                mode="reciprocal_rerank",
                use_async=False,
            )
        else:
            retriever = dense_retriever

        merged: dict[int, float] = defaultdict(float)
        qlist = [q for q in queries if str(q).strip()] or ["резюме лекции"]
        for q in qlist:
            bundle = QueryBundle(str(q))
            retrieved = retriever.retrieve(bundle)
            for nws in retrieved:
                node = nws.node
                cid_meta = node.metadata.get("chunk_id")
                if cid_meta is None:
                    continue
                i = int(cid_meta)
                sc = float(nws.score or 0.0)
                merged[i] += sc

        ordered_by_score = sorted(merged.keys(), key=lambda i: merged[i], reverse=True)
        candidate_ids = ordered_by_score[: max(mmr_pool, top_k)]
        if not candidate_ids:
            candidate_ids = list(range(min(top_k, n)))

        main_q = " \n".join(qlist)
        try:
            qe = getattr(embed_model, "get_query_embedding", None)
            query_emb = list(qe(main_q)) if callable(qe) else list(embed_model.get_text_embedding(main_q))
        except Exception:
            query_emb = list(embed_model.get_text_embedding(main_q))
        mmr_selected = mmr_select_chunk_ids(query_emb, candidate_ids, id_to_emb, top_k)
        selected = list(mmr_selected if mmr_selected else candidate_ids[:top_k])

        sem_neighbors: dict[int, list[int]] = {}
        if use_semantic_neighbors and sem_neighbor_k > 0 and id_to_emb:
            for cid in selected:
                sem_neighbors[cid] = semantic_neighbor_ids(cid, id_to_emb, n, sem_neighbor_k)

        return selected, texts, sem_neighbors
    finally:
        observe_rag_retrieval(duration_seconds=time.perf_counter() - t0)


def retrieve_chunks_for_queries_sync(
    lecture_id: str,
    queries: list[str],
    fallback_transcript: str | None = None,
) -> tuple[list[int], list[str], dict[int, list[int]]]:
    """Поиск по Qdrant с опциональным одноразовым индексом из fallback-транскрипта (локальный режим без worker)."""
    try:
        return retrieve_chunk_ids_stored_sync(lecture_id, queries)
    except RAGNotIndexedError:
        fb = str(fallback_transcript or "").strip()
        allow = os.getenv("RAG_QA_FALLBACK_REINDEX", "true").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if allow and fb and rag_indexing_enabled():
            index_full_transcript_to_qdrant_sync(fb, lecture_id)
            return retrieve_chunk_ids_stored_sync(lecture_id, queries)
        raise
