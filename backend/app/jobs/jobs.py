"""Arq task functions executed by the worker.

These are thin async wrappers that run the CPU-bound sync work in a thread so
the Arq event loop stays responsive.
"""
from __future__ import annotations

import asyncio

from sqlmodel import Session

from app.core.db import engine
from app.jobs.compression import (
    _is_modelscope_url,
    run_compression_task,
)
from app.jobs.download import download_file
from app.jobs.storage import original_model_path, safe_extension
from app.models import Model, ModelStatus


def _pull_model_sync(model_id: object) -> None:
    with Session(engine) as session:
        model = session.get(Model, model_id)
        if model is None:
            return
        try:
            url = (model.source_url or "").strip()
            if model.framework == "huggingface" and url and (
                "://" not in url or _is_modelscope_url(url)
            ):
                # HuggingFace repo id or ModelScope repo id/URL: no local
                # download, the compression job fetches the weights on demand
                # (huggingface.co -> hf-mirror.com -> ModelScope).
                model.status = ModelStatus.ready.value
                session.add(model)
                session.commit()
                return
            if not url:
                raise ValueError("source_url is empty")

            extension = safe_extension(url.split("?")[0].split("/")[-1])
            dest = original_model_path(model.id, extension)
            download_file(url, dest)
            model.file_path = str(dest)
            model.size_bytes = dest.stat().st_size
            model.status = ModelStatus.ready.value
            model.error_message = None
            session.add(model)
            session.commit()
        except Exception as exc:
            model.status = ModelStatus.failed.value
            model.error_message = f"{type(exc).__name__}: {exc}"[:2048]
            session.add(model)
            session.commit()


async def pull_model(_ctx: object, model_id: object) -> None:
    """Download a model from a URL (or mark a HF repo id as ready)."""
    await asyncio.to_thread(_pull_model_sync, model_id)


def _mark_compression_failed(task_id: object, message: str) -> None:
    """Best-effort failure bookkeeping when the service layer raises.

    The CPU path never reaches here (run_compression_task owns its exception
    handling); this exists for backend implementations that are not ready yet
    (e.g. the GPU skeleton) so tasks fail cleanly instead of hanging.
    """
    from datetime import UTC, datetime

    from app.models import TaskStatus

    with Session(engine) as session:
        task = session.get(Model, task_id)
        if task is None:
            return
        task.status = TaskStatus.failed.value
        task.error_message = message[:2048]
        task.finished_at = datetime.now(UTC)
        session.add(task)
        session.commit()


async def run_compression(_ctx: object, task_id: object) -> None:
    """Run a compression task through the selected compute backend.

    COMPUTE_BACKEND=cpu (default) delegates to the existing pipeline
    (``run_compression_task``) unchanged. COMPUTE_BACKEND=gpu uses the GPU
    service skeleton, which currently marks the task as failed with a clear
    message until implementations land.
    """
    import logging

    logger = logging.getLogger("app.jobs.run_compression")
    try:
        from app.services import NotImplementedBackendError, get_service_bundle

        bundle = get_service_bundle()
        await asyncio.to_thread(bundle.runner.run, task_id)
    except NotImplementedBackendError as exc:
        logger.critical("compression skipped (backend not ready): %s", exc)
        _mark_compression_failed(task_id, str(exc))
    except Exception as exc:  # defensive: never let a job die silently
        logger.exception("compression failed on backend: %s", exc)
        _mark_compression_failed(task_id, f"{type(exc).__name__}: {exc}")
