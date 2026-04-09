import atexit
import errno
import fcntl
import logging
import os
import threading
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
    _observe_common("qwen", duration_seconds=duration_seconds)
    _inference_context_length.labels(model="qwen", unit="tokens").observe(max(prompt_tokens, 0))
    _inference_tokens_total.labels(model="qwen").inc(max(completion_tokens, 0))
    if duration_seconds > 0 and completion_tokens > 0:
        _inference_tokens_per_second.labels(model="qwen").observe(completion_tokens / duration_seconds)


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
            _inference_gpu_util_percent.labels(model=model, gpu_index=str(idx)).set(float(util.gpu))
            _inference_gpu_memory_used_bytes.labels(model=model, gpu_index=str(idx)).set(float(mem.used))
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
