import json
import logging
import os
from typing import Any

import redis.asyncio as redis

logger = logging.getLogger(__name__)


def task_events_channel(task_id: str) -> str:
    return f"task:{task_id}"


def task_events_redis_url() -> str | None:
    raw = os.getenv("TASK_EVENTS_REDIS_URL", "").strip()
    if raw:
        return raw
    rb = os.getenv("CELERY_RESULT_BACKEND", "").strip()
    if rb.startswith("redis://") or rb.startswith("rediss://"):
        return rb
    return None


def task_events_redis_client() -> redis.Redis | None:
    url = task_events_redis_url()
    if not url:
        return None
    return redis.from_url(url, decode_responses=True)


async def publish_task_event(task_id: str, payload: dict[str, Any]) -> None:
    client = task_events_redis_client()
    if client is None:
        return
    try:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        n = await client.publish(task_events_channel(task_id), body)
        if n:
            logger.info("task event pub task_id=%s receivers=%s", task_id, n)
        else:
            logger.debug("task event pub task_id=%s (no subscribers)", task_id)
    except Exception:
        logger.exception("publish_task_event failed task_id=%s", task_id)
    finally:
        try:
            await client.aclose()
        except Exception:
            pass
