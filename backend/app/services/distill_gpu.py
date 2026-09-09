"""GPU distillation/pruning service - SKELETON.

TODO(gpu): CUDA implementations of ``distill`` (knowledge distillation),
``prune`` (structured filter pruning) and ``sparsify`` (magnitude pruning).
All three are training-style loops with forward/backward passes -> natural GPU
winners. Design notes:
  * keep per-device batch splits from GPU_DEVICE_IDS when using data-parallel;
  * keep eval/benchmark steps on the same device; fp16 autocast optional.
"""
from __future__ import annotations

import logging
from typing import Any

from app.services.base import TRAIN_METHODS, NotImplementedBackendError
from app.services.compression_gpu import GpuCompressionRunner

logger = logging.getLogger("app.services.distill_gpu")


class DistillGpuService(GpuCompressionRunner):
    name = "distill.gpu"
    supported_methods = TRAIN_METHODS

    def distill(self, task_id: Any) -> None:
        raise NotImplementedBackendError(
            "GPU distillation service is a skeleton; set COMPUTE_BACKEND=cpu."
        )
