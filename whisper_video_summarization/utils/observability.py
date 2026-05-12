import atexit
import contextvars
import datetime as _dt
import errno
import fcntl
import logging
import os
import re
import threading
import time
from typing import Any

import psutil

from whisper_video_summarization.utils.prometheus_multiproc import (
    ensure_multiproc_dir,
    registry_for_export,
)

ensure_multiproc_dir()

from prometheus_client import Counter, Gauge, Histogram, start_http_server

logger = logging.getLogger(__name__)

_api_infer_requests_total = Counter(
    "inference_api_requests_total",
    "Total API requests for inference endpoints.",
    ["path", "method"],
)

_inference_duration_seconds = Histogram(
    "inference_duration_seconds",
    "Inference request duration in seconds.",
    ["model"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 40, 60, 120),
)

_inference_tokens_total = Counter(
    "inference_tokens_total",
    "Generated tokens during inference.",
    ["model"],
)

_inference_tokens_per_second = Histogram(
    "inference_tokens_per_second",
    "Generated tokens per second.",
    ["model"],
    buckets=(0.1, 0.5, 1, 2, 5, 10, 20, 40, 80, 160, 320),
)

_inference_context_length = Histogram(
    "inference_context_length",
    "Input context length seen by inference model.",
    ["model", "unit"],
    buckets=(1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768),
)

_inference_cpu_percent = Gauge(
    "inference_cpu_percent",
    "Current host CPU utilization sampled during inference.",
    ["model"],
    multiprocess_mode="max",
)

_inference_ram_bytes = Gauge(
    "inference_ram_used_bytes",
    "Current host RAM used bytes sampled during inference.",
    ["model"],
    multiprocess_mode="max",
)

_inference_gpu_util_percent = Gauge(
    "inference_gpu_utilization_percent",
    "Current GPU utilization sampled during inference.",
    ["model", "gpu_index"],
    multiprocess_mode="max",
)

_inference_gpu_memory_used_bytes = Gauge(
    "inference_gpu_memory_used_bytes",
    "Current GPU memory used sampled during inference.",
    ["model", "gpu_index"],
    multiprocess_mode="max",
)

_inference_gpu_memory_total_bytes = Gauge(
    "inference_gpu_memory_total_bytes",
    "Total GPU memory in bytes (NVML mem.total), per physical GPU index.",
    ["gpu_index"],
    multiprocess_mode="max",
)

# --- Component-oriented names (dashboards / alerts) ---

_pyannote_processing_seconds = Histogram(
    "pyannote_processing_seconds",
    "Wall-clock seconds for pyannote HTTP diarization (client-side).",
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 40, 60, 120, 600),
)

_api_http_request_duration_seconds = Histogram(
    "api_http_request_duration_seconds",
    "API HTTP request duration in seconds (mounted /api app).",
    ["method", "route"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60),
)

_task_total_seconds = Histogram(
    "task_total_seconds",
    "End-to-end time from inference task creation (DB row insert) until terminal status "
    "(COMPLETED/FAILED), labeled by task_type and final status.",
    ["task_type", "status"],
    buckets=(0.5, 1, 2, 5, 10, 20, 40, 60, 120, 300, 600, 1200, 1800, 3600, 7200),
)

_task_component_wall_seconds = Histogram(
    "task_component_wall_seconds",
    "Total wall seconds for one InferenceTask in a single component (one observe at terminal). "
    "Values come from transcription JSON _meta and LLM infer accumulation; not RAG index worker.",
    ["component", "task_type", "status"],
    buckets=(0.5, 1, 2, 5, 10, 20, 40, 60, 120, 300, 600, 1200, 1800, 3600, 7200),
)

_infer_wall_active: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_infer_wall_active", default=False
)

# NOTE: We intentionally store a MUTABLE object in a contextvar.
# asyncio creates new Tasks by copying the current context. For immutable types (float),
# concurrent Tasks would accumulate into their own copies and the parent would only see
# a partial total. A shared mutable dict keeps a single accumulator reference across Tasks.
_infer_wall_accum: contextvars.ContextVar[dict[str, float] | None] = contextvars.ContextVar(
    "_infer_wall_accum", default=None
)

_gpu_utilization_percent = Gauge(
    "gpu_utilization_percent",
    "GPU utilization percent from NVML (0–100), sampled with other inference metrics.",
    ["gpu_index"],
    multiprocess_mode="max",
)

_rag_retrieval_duration_seconds = Histogram(
    "rag_retrieval_duration_seconds",
    "RAG chunk retrieval duration (Qdrant + retrievers + MMR) in seconds.",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 40, 60, 120),
)

_embeddings_batch_processing_seconds = Histogram(
    "embeddings_batch_processing_seconds",
    "Wall-clock seconds for a single remote embeddings batch call.",
    ["stage"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 40, 60, 120, 300),
)

_worker_metrics_started = False
_worker_metrics_lock = threading.Lock()
_worker_metrics_lock_fd: int | None = None
_nvml_tried = False
_nvml_ok = False
_pynvml: Any = None


def observe_api_infer_request(path: str, method: str) -> None:
    _api_infer_requests_total.labels(path=path, method=method).inc()


def observe_qwen_inference(
    *,
    duration_seconds: float,
    prompt_tokens: int,
    completion_tokens: int,
) -> None:
    _ = completion_tokens
    _observe_common("qwen", duration_seconds=duration_seconds)
    _inference_context_length.labels(model="qwen", unit="tokens").observe(max(prompt_tokens, 0))


def observe_whisper_inference(
    *,
    duration_seconds: float,
    input_seconds: float | None = None,
    output_tokens: int | None = None,
) -> None:
    _observe_common("whisper", duration_seconds=duration_seconds)
    if input_seconds is not None and input_seconds > 0:
        _inference_context_length.labels(model="whisper", unit="seconds").observe(input_seconds)
    if output_tokens is not None and output_tokens > 0:
        _inference_tokens_total.labels(model="whisper").inc(output_tokens)
        if duration_seconds > 0:
            _inference_tokens_per_second.labels(model="whisper").observe(output_tokens / duration_seconds)


def observe_pyannote_diarization(*, duration_seconds: float) -> None:
    _pyannote_processing_seconds.observe(max(duration_seconds, 0.0))


def observe_rag_retrieval(*, duration_seconds: float) -> None:
    _rag_retrieval_duration_seconds.observe(max(duration_seconds, 0.0))


def observe_embeddings_batch(*, duration_seconds: float, stage: str = "other") -> None:
    d = max(duration_seconds, 0.0)
    _embeddings_batch_processing_seconds.labels(stage=stage).observe(d)
    if not _infer_wall_active.get() or d <= 0:
        return
    acc = _infer_wall_accum.get()
    if acc is None:
        return
    acc["embeddings"] = float(acc.get("embeddings", 0.0)) + d


def infer_wall_tracking_begin() -> tuple[Any, Any, Any]:
    """Start per-``infer()`` accumulation of Qwen + embeddings wall time (reset counters)."""
    return (
        _infer_wall_active.set(True),
        _infer_wall_accum.set({"qwen": 0.0, "embeddings": 0.0}),
        None,
    )


def infer_wall_tracking_end(tokens: tuple[Any, Any, Any]) -> None:
    tok_active, tok_acc, _tok_unused = tokens
    if tok_acc is not None:
        _infer_wall_accum.reset(tok_acc)
    _infer_wall_active.reset(tok_active)


def bump_infer_accum_qwen_wall_seconds(seconds: float) -> None:
    if seconds <= 0 or not _infer_wall_active.get():
        return
    acc = _infer_wall_accum.get()
    if acc is None:
        return
    acc["qwen"] = float(acc.get("qwen", 0.0)) + float(seconds)


def infer_wall_totals_snapshot() -> tuple[float, float]:
    acc = _infer_wall_accum.get() or {}
    return float(acc.get("qwen", 0.0)), float(acc.get("embeddings", 0.0))


_TASK_ID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
    re.IGNORECASE,
)


def _api_route_label(request: Any) -> str:
    route = request.scope.get("route")
    if route is not None:
        p = getattr(route, "path", None)
        if isinstance(p, str) and p:
            return p
    return _TASK_ID_RE.sub("{id}", request.url.path)


def register_api_http_metrics_middleware(app: Any) -> None:
    from starlette.requests import Request

    @app.middleware("http")
    async def _prometheus_http_metrics(request: Request, call_next):
        t0 = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - t0
        route = _api_route_label(request)
        _api_http_request_duration_seconds.labels(
            method=request.method,
            route=route,
        ).observe(elapsed)
        return response


_TERMINAL_TASK_STATUSES = {"completed", "failed"}


def observe_inference_task_terminal(row: Any) -> None:
    """Record end-to-end time from row.created_at to row.updated_at for terminal status.

    Uses DB ``updated_at`` (set on the status commit) as the end instant so duration
    matches the task row, not wall clock after post-commit work. Falls back to UTC now
    if ``updated_at`` is missing or earlier than ``created_at``.

    Safe to call on any status change; only emits the histogram for COMPLETED/FAILED.
    """
    status = getattr(row, "status", None)
    status_val = getattr(status, "value", status)
    if not isinstance(status_val, str) or status_val.lower() not in _TERMINAL_TASK_STATUSES:
        return
    created_at = getattr(row, "created_at", None)
    if not isinstance(created_at, _dt.datetime):
        return
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=_dt.timezone.utc)
    end = getattr(row, "updated_at", None)
    if isinstance(end, _dt.datetime):
        if end.tzinfo is None:
            end = end.replace(tzinfo=_dt.timezone.utc)
        if end >= created_at:
            elapsed = (end - created_at).total_seconds()
        else:
            elapsed = (_dt.datetime.now(_dt.timezone.utc) - created_at).total_seconds()
    else:
        elapsed = (_dt.datetime.now(_dt.timezone.utc) - created_at).total_seconds()
    if elapsed < 0:
        return
    task_type = getattr(row, "task_type", None)
    task_type_val = getattr(task_type, "value", task_type) or "unknown"
    _task_total_seconds.labels(
        task_type=str(task_type_val),
        status=status_val.lower(),
    ).observe(elapsed)
    _observe_task_component_walls_from_row(row, task_type_val, status_val.lower())


def _meta_wall_seconds(meta: Any, key: str) -> float:
    """Parse seconds from JSONB _meta (asyncpg/SQLAlchemy may return Decimal, numpy scalar, str)."""
    if not isinstance(meta, dict):
        return 0.0
    v = meta.get(key)
    if v is None or isinstance(v, bool):
        return 0.0
    try:
        return max(float(v), 0.0)
    except (TypeError, ValueError):
        return 0.0


def _observe_task_component_walls_from_row(row: Any, task_type_val: str, status: str) -> None:
    raw = getattr(row, "result_transcription_json", None)
    meta: dict[str, Any] = {}
    if isinstance(raw, dict):
        m = raw.get("_meta")
        if isinstance(m, dict):
            meta = m
    tt = str(task_type_val)
    st = str(status).lower()
    pairs = (
        ("whisper", "task_wall_whisper_seconds"),
        ("pyannote", "task_wall_pyannote_seconds"),
        ("qwen", "task_wall_qwen_seconds"),
        ("embeddings", "task_wall_embeddings_seconds"),
    )
    for component, mkey in pairs:
        _task_component_wall_seconds.labels(
            component=component,
            task_type=tt,
            status=st,
        ).observe(_meta_wall_seconds(meta, mkey))


def start_worker_metrics_server() -> None:
    """Start metrics HTTP once per machine: extra Uvicorn/Celery workers skip (flock)."""
    global _worker_metrics_started, _worker_metrics_lock_fd

    with _worker_metrics_lock:
        if _worker_metrics_started:
            return
        port = int(os.getenv("WORKER_METRICS_PORT", "9101"))
        lock_path = os.getenv(
            "WORKER_METRICS_LOCK_PATH",
            "/tmp/whisper_video_summarization_worker_metrics.lock",
        )
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            logger.info(
                "Worker metrics lock held by another process; skipping bind on :%s",
                port,
            )
            return
        try:
            start_http_server(port, registry=registry_for_export())
        except OSError as exc:
            if exc.errno == errno.EADDRINUSE:
                logger.warning(
                    "Worker metrics port %s already in use; skipping duplicate server",
                    port,
                )
                os.close(fd)
                return
            os.close(fd)
            raise
        _worker_metrics_lock_fd = fd
        _worker_metrics_started = True
        logger.info("Prometheus worker metrics server started on :%s", port)


def _mark_process_dead() -> None:
    if not os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
        return
    try:
        from prometheus_client import multiprocess

        multiprocess.mark_process_dead(os.getpid())
    except Exception:
        logger.debug("prometheus mark_process_dead failed", exc_info=True)


atexit.register(_mark_process_dead)


def _observe_common(model: str, *, duration_seconds: float) -> None:
    _inference_duration_seconds.labels(model=model).observe(max(duration_seconds, 0.0))
    _inference_cpu_percent.labels(model=model).set(max(psutil.cpu_percent(interval=None), 0.0))
    _inference_ram_bytes.labels(model=model).set(float(psutil.virtual_memory().used))
    _observe_gpu(model)


def _observe_gpu(model: str) -> None:
    if not _init_nvml():
        return
    assert _pynvml is not None
    try:
        count = _pynvml.nvmlDeviceGetCount()
        for idx in range(count):
            handle = _pynvml.nvmlDeviceGetHandleByIndex(idx)
            util = _pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem = _pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_pct = float(util.gpu)
            _inference_gpu_util_percent.labels(model=model, gpu_index=str(idx)).set(gpu_pct)
            _gpu_utilization_percent.labels(gpu_index=str(idx)).set(gpu_pct)
            _inference_gpu_memory_used_bytes.labels(model=model, gpu_index=str(idx)).set(float(mem.used))
            _inference_gpu_memory_total_bytes.labels(gpu_index=str(idx)).set(float(mem.total))
    except Exception:
        logger.exception("Failed to collect NVML GPU metrics")


def _init_nvml() -> bool:
    global _nvml_tried, _nvml_ok, _pynvml
    if _nvml_tried:
        return _nvml_ok
    _nvml_tried = True
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        _pynvml = pynvml
        _nvml_ok = True
    except Exception:
        _nvml_ok = False
        logger.warning("NVML unavailable, GPU metrics disabled")
    return _nvml_ok
