"""GPU compression runner - SKELETON (not implemented yet).

Migration status
----------------
* Keep ``COMPUTE_BACKEND=cpu`` in production until a method is flipped below.
* Enable methods one by one by implementing the corresponding ``_run_*`` and
  removing it from ``NOT_IMPLEMENTED_YET``.

Per-method GPU strategy (see docs/gpu-migration-plan.md for the full matrix):
* ``llm_quantize``      -> optimum-intel / NNCF with OpenVINO ``GPU`` plugin or
                           torch CUDA calibration; stateful IR on GPU later.
* ``quantize_onnx(_static)`` -> ONNX Runtime ``CUDAExecutionProvider`` static
                           quantization + calibration mini-batches on GPU.
* ``openvino_int8``     -> NNCF on ``GPU`` device (INT8 CV models).
* ``qat``               -> FX QAT training on CUDA.
* ``prune/sparsify/distill`` -> CUDA training loops.
* ``export_onnx/export_gguf`` -> conversion is memory/IO bound; usually stays
                           on CPU, but fp16 CUDA trace helps large LLMs.
"""
from __future__ import annotations

import logging
from typing import Any, ClassVar

from app.services.base import (
    ALL_COMPRESSION_METHODS,
    CompressionService,
    NotImplementedBackendError,
)

logger = logging.getLogger("app.services.compression_gpu")

NOT_IMPLEMENTED_YET: frozenset[str] = frozenset(ALL_COMPRESSION_METHODS)


class GpuCompressionRunner(CompressionService):
    """Placeholder for GPU-backed compression/quantization jobs."""

    name = "compression.gpu"
    backend = "gpu"
    supported_methods = ALL_COMPRESSION_METHODS
    #: methods that are actually wired to a GPU implementation
    implemented_methods: ClassVar[frozenset[str]] = frozenset()

    def run(self, task_id: Any, method: str | None = None) -> None:
        if method is not None and method not in NOT_IMPLEMENTED_YET:
            raise NotImplementedBackendError(
                f"gpu backend not implemented yet for method={method!r}; "
                "keep COMPUTE_BACKEND=cpu until it is enabled."
            )
        raise NotImplementedBackendError(
            "gpu compression backend is a skeleton. "
            "Set COMPUTE_BACKEND=cpu (default) for now."
        )


def get_gpu_runner() -> GpuCompressionRunner:
    return GpuCompressionRunner()
