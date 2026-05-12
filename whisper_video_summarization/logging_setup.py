"""Uvicorn leaves app loggers without handlers; root is often WARNING — INFO would be dropped."""

from __future__ import annotations

import logging
import os
import sys

_done = False


def ensure_package_logging() -> None:
    global _done
    if _done:
        return
    _done = True

    raw = (os.getenv("LOG_LEVEL") or "INFO").strip().upper()
    level = getattr(logging, raw, None)
    if not isinstance(level, int):
        level = logging.INFO

    pkg = logging.getLogger("whisper_video_summarization")
    pkg.setLevel(level)
    if pkg.handlers:
        return

    h = logging.StreamHandler(sys.stderr)
    h.setLevel(level)
    h.setFormatter(
        logging.Formatter(
            os.getenv("WHISPER_LOG_FORMAT", "%(levelname)s %(name)s: %(message)s"),
        ),
    )
    pkg.addHandler(h)
    pkg.propagate = False
