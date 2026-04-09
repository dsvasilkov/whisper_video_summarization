"""PROMETHEUS_MULTIPROC_DIR + merged registry for multi-worker Uvicorn / prefork."""

import os
from pathlib import Path


def ensure_multiproc_dir() -> None:
    d = os.environ.get("PROMETHEUS_MULTIPROC_DIR")
    if not d:
        return
    Path(d).mkdir(parents=True, exist_ok=True)


def registry_for_export():
    """REGISTRY in single-process mode; MultiProcessCollector when multiproc dir is set."""
    from prometheus_client import REGISTRY, CollectorRegistry
    from prometheus_client import multiprocess

    if not os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
        return REGISTRY
    ensure_multiproc_dir()
    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry)
    return registry
