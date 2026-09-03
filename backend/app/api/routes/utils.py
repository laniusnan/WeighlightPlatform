from typing import Any

from fastapi import APIRouter, Depends, Response
from pydantic.networks import EmailStr

from app.api.deps import SessionDep, get_current_active_superuser
from app.models import Message
from app.utils import generate_test_email, send_email

router = APIRouter(prefix="/utils", tags=["utils"])


@router.post(
    "/test-email/",
    dependencies=[Depends(get_current_active_superuser)],
    status_code=201,
)
def test_email(email_to: EmailStr) -> Message:
    """
    Test emails.
    """
    email_data = generate_test_email(email_to=email_to)
    send_email(
        email_to=email_to,
        subject=email_data.subject,
        html_content=email_data.html_content,
    )
    return Message(message="Test email sent")


@router.get("/workers/")
async def workers() -> dict[str, Any]:
    """Worker fleet status: live workers (heartbeat), queue depth, running jobs."""
    import json as _json
    import time as _time

    from app.core.redis import get_redis_pool

    try:
        pool = await get_redis_pool()
        keys = list(await pool.keys("platform:worker:*"))
        worker_list: list[dict[str, Any]] = []
        for key in keys:
            raw = await pool.get(key)
            if raw:
                try:
                    info = _json.loads(raw)
                    info["uptime_s"] = round(
                        _time.time() - float(info.get("started_at") or 0)
                    )
                    worker_list.append(info)
                except Exception:
                    continue
        queue_depth = int(await pool.llen("arq:queue:default") or 0)
        running = len(list(await pool.keys("arq:in-progress:*")))
    except Exception:
        return {
            "redis": "down",
            "workers": [],
            "worker_count": 0,
            "queue_depth": 0,
            "running_jobs": 0,
        }

    return {
        "redis": "up",
        "workers": worker_list,
        "worker_count": len(worker_list),
        "queue_depth": queue_depth,
        "running_jobs": running,
    }


@router.get("/health-check/")
async def health_check() -> bool:
    return True


@router.get("/platform-status/")
async def platform_status(
    session: SessionDep,
) -> dict[str, Any]:
    """Lightweight platform status: Redis connectivity + resource counts."""
    from sqlmodel import func, select

    from app.core.redis import get_redis_pool
    from app.models import CompressionTask, Dataset, Model, TaskStatus

    redis_status = "up"
    try:
        pool = await get_redis_pool()
        await pool.ping()
    except Exception:
        redis_status = "down"

    def _count(model_cls: type) -> int:
        return int(session.exec(select(func.count()).select_from(model_cls)).one())

    def _count_tasks(status: str) -> int:
        return int(
            session.exec(
                select(func.count())
                .select_from(CompressionTask)
                .where(CompressionTask.status == status)
            ).one()
        )

    storage_bytes = int(
        session.exec(
            select(func.coalesce(func.sum(Model.size_bytes), 0))
        ).one()
    ) + int(
        session.exec(
            select(func.coalesce(func.sum(Dataset.size_bytes), 0))
        ).one()
    )

    return {
        "redis": redis_status,
        "tasks": {
            status.value: _count_tasks(status.value)
            for status in TaskStatus
        },
        "models": _count(Model),
        "datasets": _count(Dataset),
        "storage_bytes": storage_bytes,
    }


@router.get("/metrics/")
async def metrics(session: SessionDep) -> Response:
    """Prometheus-style text metrics for the platform (no extra deps)."""
    from sqlmodel import func, select

    from app.models import CompressionTask, Dataset, Model, TaskStatus

    lines = [
        "# HELP platform_tasks_total Compression tasks by status",
        "# TYPE platform_tasks_total gauge",
    ]
    for status in TaskStatus:
        count = int(
            session.exec(
                select(func.count())
                .select_from(CompressionTask)
                .where(CompressionTask.status == status.value)
            ).one()
        )
        lines.append(f'platform_tasks_total{{status="{status.value}"}} {count}')
    models = int(
        session.exec(select(func.count()).select_from(Model)).one()
    )
    datasets = int(
        session.exec(select(func.count()).select_from(Dataset)).one()
    )
    lines.append("# HELP platform_models_total Registered models")
    lines.append("# TYPE platform_models_total gauge")
    lines.append(f"platform_models_total {models}")
    lines.append("# HELP platform_datasets_total Registered datasets")
    lines.append("# TYPE platform_datasets_total gauge")
    lines.append(f"platform_datasets_total {datasets}")
    return Response(
        content="\n".join(lines) + "\n",
        media_type="text/plain; version=0.0.4",
    )
