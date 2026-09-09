"""Arq worker settings.

Start the worker with:

    arq app.worker.WorkerSettings

Workers register a heartbeat key (``platform:worker:<uuid>``) in Redis so the
platform can monitor how many workers are alive and what they are doing.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
import uuid

from arq.connections import RedisSettings

from app.core.config import settings
from app.jobs.jobs import pull_model, run_compression

_WORKER_UUID = str(uuid.uuid4())
_HEARTBEAT_KEY = f"platform:worker:{_WORKER_UUID}"


async def _heartbeat_loop(ctx: dict) -> None:
    """Refresh the worker heartbeat key while the worker is alive."""
    payload = json.dumps(
        {
            "id": _WORKER_UUID,
            "pid": os.getpid(),
            "started_at": ctx.get("started_at"),
            "queue": ctx.get("queue_name", "default"),
        }
    )
    try:
        while not ctx.get("_stop", False):
            await ctx["redis"].psetex(_HEARTBEAT_KEY, 15000, payload)
            await asyncio.sleep(5)
    except asyncio.CancelledError:
        pass
    except Exception:
        pass


class WorkerSettings:
    functions = [pull_model, run_compression]
    redis_settings = RedisSettings.from_dsn(str(settings.REDIS_URL))
    # CPU-only servers: run one compression job at a time to avoid CPU thrash.
    max_jobs = int(os.environ.get("WORKER_MAX_JOBS", "1"))
    # Longest allowed job: 12 hours (LLM INT4 export can take a long time).
    job_timeout = 12 * 60 * 60
    keep_result = 60 * 60 * 24

    # Arq reads these straight from the class namespace and calls them with
    # only ``ctx`` (no ``self``), so they must not be instance methods.
    async def on_startup(ctx: dict) -> None:  # type: ignore[no-untyped-def]
        from app.services import configure_runtime, resolve_backend

        compute = await asyncio.to_thread(configure_runtime, resolve_backend())
        ctx["started_at"] = time.time()
        ctx["_stop"] = False
        ctx["compute"] = compute
        # Visible in `docker logs <worker>` even without a logging config.
        print(f"[worker] compute backend={compute}", flush=True)
        ctx["heartbeat_task"] = asyncio.create_task(_heartbeat_loop(ctx))

    async def on_shutdown(ctx: dict) -> None:  # type: ignore[no-untyped-def]
        ctx["_stop"] = True
        task = ctx.get("heartbeat_task")
        if task is not None:
            task.cancel()
        try:
            await ctx["redis"].delete(_HEARTBEAT_KEY)
        except Exception:
            pass

