"""Lazy Redis / Arq pool used by the API to enqueue jobs."""
from __future__ import annotations

from arq import create_pool
from arq.connections import ArqRedis, RedisSettings

from app.core.config import settings

_pool: ArqRedis | None = None


async def get_redis_pool() -> ArqRedis:
    global _pool
    if _pool is None:
        _pool = await create_pool(
            RedisSettings.from_dsn(str(settings.REDIS_URL))
        )
    return _pool


async def enqueue_job(name: str, *args: object) -> None:
    """Enqueue a job by name on the default Arq queue."""
    pool = await get_redis_pool()
    await pool.enqueue_job(name, *args)


async def close_redis_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.aclose()
        _pool = None
