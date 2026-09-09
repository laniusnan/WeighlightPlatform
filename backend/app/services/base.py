"""Compute-backend abstraction layer (GPU migration prep).

Design
------
* ``COMPUTE_BACKEND=cpu|gpu`` selects which implementations the factory builds.
* All abstraction lives under :mod:`app.services`; nothing in the API/routes
  changes.  Existing CPU job logic stays in ``app.jobs.compression`` and is
  *reused* verbatim by the CPU implementations below (single source of truth).
* GPU implementations are skeletons raising ``NotImplementedError`` today; when
  a method is ready it is enabled per-method so ``cpu`` continues to work
  untouched.
"""
from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import Any, ClassVar

from app.core.config import settings

logger = logging.getLogger("app.services")

# Method families (values mirror app.models.CompressionMethod).
ALL_COMPRESSION_METHODS: frozenset[str] = frozenset(
    {
        "export_onnx",
        "quantize_onnx",
        "quantize_onnx_static",
        "openvino_int8",
        "prune",
        "sparsify",
        "distill",
        "qat",
        "export_gguf",
        "llm_quantize",
    }
)
# Conversion / weight-compression / quantization-oriented jobs.
QUANTIZE_METHODS: frozenset[str] = frozenset(
    {
        "export_onnx",
        "quantize_onnx",
        "quantize_onnx_static",
        "openvino_int8",
        "qat",
        "export_gguf",
        "llm_quantize",
    }
)
# Iterative / training-style jobs (pruning, sparsification, distillation).
TRAIN_METHODS: frozenset[str] = frozenset({"prune", "sparsify", "distill"})


def resolve_backend() -> str:
    """Return the normalized compute backend (``cpu`` or ``gpu``)."""
    backend = str(settings.COMPUTE_BACKEND or "cpu").strip().lower()
    if backend not in {"cpu", "gpu"}:
        raise ValueError(
            f"COMPUTE_BACKEND must be 'cpu' or 'gpu', got: {backend!r}"
        )
    return backend


def resolve_devices(backend: str) -> list[str]:
    """Effective device ids visible to this process.

    * ``cpu``   -> ["cpu"]
    * ``gpu``   -> intersection of GPU_DEVICE_IDS and CUDA_VISIBLE_DEVICES
                   (when CUDA_VISIBLE_DEVICES is set, it is authoritative).
    """
    if backend == "cpu":
        return ["cpu"]
    visible = str(os.environ.get("CUDA_VISIBLE_DEVICES") or settings.CUDA_VISIBLE_DEVICES or "")
    requested = [d.strip() for d in str(settings.GPU_DEVICE_IDS).split(",") if d.strip()]
    if not visible or visible.strip() in {"", "-1"}:
        # No CUDA_VISIBLE_DEVICES restriction -> assume requested ids are visible.
        return requested or ["0"]
    visible_ids = [d.strip() for d in visible.split(",") if d.strip()]
    return [d for d in requested if d in visible_ids] or ["0"]


def configure_runtime(backend: str) -> dict[str, str]:
    """Apply backend-specific runtime configuration and log the result.

    Called by the Arq worker at startup and by ``check_backend.py``. CPU is a
    no-op (keeps the current behaviour exactly); GPU configures PyTorch to use
    CUDA devices from ``GPU_DEVICE_IDS``.
    """
    if backend == "gpu":
        devices = resolve_devices("gpu")
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", ",".join(devices))
        try:
            import torch

            if torch.cuda.is_available():
                torch.set_default_device("cuda")
                logger.info(
                    "compute backend=gpu | cuda=%s | devices=%s",
                    torch.version.cuda,
                    devices,
                )
            else:
                logger.warning(
                    "COMPUTE_BACKEND=gpu but torch.cuda.is_available()=False; "
                    "falling back to cpu inside the service layer."
                )
        except Exception as exc:  # pragma: no cover - environment specific
            logger.warning("could not configure CUDA runtime: %s", exc)
    else:
        logger.info("compute backend=cpu")
    return {
        "backend": backend,
        "devices": ",".join(resolve_devices(backend)),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }


# --------------------------------------------------------------------------- #
# Abstract interfaces
# --------------------------------------------------------------------------- #
class CompressionService(ABC):
    """Base contract for CPU/GPU compression job runners."""

    name: ClassVar[str]
    supported_methods: ClassVar[frozenset[str]] = ALL_COMPRESSION_METHODS
    backend: ClassVar[str] = "cpu"

    @abstractmethod
    def run(self, task_id: Any, method: str | None = None) -> None:
        """Run a compression/quantization task identified by ``task_id``.

        ``method`` is used for introspection/validation only; the canonical
        method lives on the ``CompressionTask`` row.
        """

    def supports(self, method: str) -> bool:
        return method in self.supported_methods


class InferenceService(ABC):
    """Base contract for on-demand model inference."""

    name: ClassVar[str]
    backend: ClassVar[str] = "cpu"

    @abstractmethod
    def run_predict(self, artifact: Any, inputs: Any, **kwargs: Any) -> Any:
        """Single synchronous prediction (non-LLM artifacts)."""

    @abstractmethod
    def chat_completion(
        self,
        artifact: Any,
        prompt: str,
        *,
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        top_p: float = 1.0,
        min_p: float = 0.0,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """LLM text completion (non-streaming) on an artifact directory."""


class NotImplementedBackendError(NotImplementedError):
    """Raised when a GPU (or CPU) implementation is not ready yet."""
