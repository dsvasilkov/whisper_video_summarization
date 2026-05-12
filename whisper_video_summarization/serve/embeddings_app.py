from __future__ import annotations

import logging
import os
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ray import serve

logger = logging.getLogger(__name__)

api = FastAPI(title="Embeddings (Ray Serve)")


def _env(name: str, default: str) -> str:
    return str(os.getenv(name, default)).strip()


class OpenAIEmbeddingsRequest(BaseModel):
    model: str = ""
    input: str | list[str] | None = None


class OpenAIEmbeddingItem(BaseModel):
    object: Literal["embedding"] = "embedding"
    embedding: list[float]
    index: int


class OpenAIEmbeddingsUsage(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0


class OpenAIEmbeddingsResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[OpenAIEmbeddingItem]
    model: str
    usage: OpenAIEmbeddingsUsage


@serve.deployment(
    ray_actor_options={
        "num_cpus": float(_env("EMBED_NUM_CPUS", "2") or "2"),
        "num_gpus": float(_env("EMBED_NUM_GPUS", "0.5") or "0.5"),
    },
    max_ongoing_requests=int(_env("EMBED_MAX_ONGOING_REQUESTS", "8") or "8"),
)
@serve.ingress(api)
class BgeM3Embedder:
    def __init__(self) -> None:
        self._model_name = _env("RAG_EMBEDDING_MODEL_NAME", "BAAI/bge-m3") or "BAAI/bge-m3"
        self._trust_remote_code = _env("RAG_EMBEDDING_TRUST_REMOTE_CODE", "true").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        logger.info("Loading embedding model %s", self._model_name)
        self._embed_model: Any = HuggingFaceEmbedding(
            model_name=self._model_name,
            trust_remote_code=bool(self._trust_remote_code),
        )
        logger.info("embeddings ready: model=%s", self._model_name)

    @api.get("/health")
    def health(self) -> dict[str, str]:
        return {"status": "ok", "model": self._model_name}

    @api.post("/v1/embeddings", response_model=OpenAIEmbeddingsResponse)
    def openai_embeddings(self, body: OpenAIEmbeddingsRequest) -> OpenAIEmbeddingsResponse:
        """OpenAI-compatible embeddings API for llama-index OpenAIEmbedding clients."""
        raw = body.input
        if raw is None:
            raise HTTPException(status_code=400, detail="missing required field: input")
        if isinstance(raw, str):
            texts = [raw.replace("\n", " ")]
        else:
            texts = [str(t).replace("\n", " ") for t in raw]
        model = (body.model or "").strip() or self._model_name
        if not texts:
            return OpenAIEmbeddingsResponse(
                data=[],
                model=model,
                usage=OpenAIEmbeddingsUsage(prompt_tokens=0, total_tokens=0),
            )
        try:
            embs = self._embed_model.get_text_embedding_batch(texts)
        except Exception as exc:
            logger.exception("OpenAI embeddings failed: %s", exc)
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        data = [
            OpenAIEmbeddingItem(embedding=list(vec), index=i) for i, vec in enumerate(embs)
        ]
        approx_tokens = sum(len(t.split()) for t in texts)
        return OpenAIEmbeddingsResponse(
            data=data,
            model=model,
            usage=OpenAIEmbeddingsUsage(prompt_tokens=approx_tokens, total_tokens=approx_tokens),
        )


app = BgeM3Embedder.bind()

