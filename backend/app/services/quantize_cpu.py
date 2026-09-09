"""Domain adapters: quantization + distillation (CPU implementations).

These narrow :class:`~app.services.compression_cpu.CpuCompressionRunner` to a
method family and expose the stable per-domain API used by the factory:
``QuantizeService.quantize(task_id)`` and ``DistillService.distill(task_id)``.
"""
from __future__ import annotations

import logging
from typing import Any

from app.services.base import QUANTIZE_METHODS
from app.services.compression_cpu import CpuCompressionRunner

logger = logging.getLogger("app.services.quantize_cpu")


class QuantizeCpuService(CpuCompressionRunner):
    """CPU quantization/conversion jobs (reuses the existing pipeline)."""

    name = "quantize.cpu"
    supported_methods = QUANTIZE_METHODS

    def quantize(self, task_id: Any) -> None:
        """Quantize/convert a task on CPU.

        The canonical method is read from the CompressionTask row by the
        shared pipeline, so no method argument is needed here.
        """
        self.run(task_id)


