"""CPU compression runner.

This is a thin adapter: the real implementation is the existing, untouched
``app.jobs.compression.run_compression_task`` pipeline (session bookkeeping,
progress, eval/benchmark hooks). Keeping the delegation here means GPU
implementations can later share the exact same task semantics.
"""
from __future__ import annotations

import logging
from typing import Any

from app.services.base import (
    ALL_COMPRESSION_METHODS,
    CompressionService,
)

logger = logging.getLogger("app.services.compression_cpu")


def run_cpu_compression(task_id: Any) -> None:
    """Delegate a task to the existing CPU pipeline."""
    from app.jobs.compression import run_compression_task

    logger.info("cpu runner executing task_id=%s", task_id)
    run_compression_task(task_id)


class CpuCompressionRunner(CompressionService):
    """Runs any compression method on CPU via the existing pipeline."""

    name = "compression.cpu"
    backend = "cpu"
    supported_methods = ALL_COMPRESSION_METHODS

    def run(self, task_id: Any, method: str | None = None) -> None:
        if method is not None and not self.supports(method):
            raise ValueError(
                f"method {method!r} is not supported by {self.name}"
            )
        run_cpu_compression(task_id)


def get_cpu_runner() -> CpuCompressionRunner:
    return CpuCompressionRunner()
