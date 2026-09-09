"""CPU prune / sparsify / distill jobs (reuses the existing pipeline)."""
from __future__ import annotations

import logging
from typing import Any

from app.services.base import TRAIN_METHODS
from app.services.compression_cpu import CpuCompressionRunner

logger = logging.getLogger("app.services.distill_cpu")


class DistillCpuService(CpuCompressionRunner):
    name = "distill.cpu"
    supported_methods = TRAIN_METHODS

    def distill(self, task_id: Any) -> None:
        """Run a prune/sparsify/distill task on CPU."""
        self.run(task_id)
