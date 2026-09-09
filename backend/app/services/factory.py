"""Service factory: returns CPU/GPU service bundles based on COMPUTE_BACKEND.

Usage::

    from app.services import get_service_bundle
    bundle = get_service_bundle()          # cached
    bundle.runner.run(task_id)             # Arq worker entry
    bundle.quantize.quantize(task_id)
    bundle.inference.chat_completion(...)

No API route touches this module yet; the worker and future GPU paths do.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import ClassVar

from app.core.config import settings
from app.services.base import (
    CompressionService,
    InferenceService,
    resolve_backend,
    resolve_devices,
)

logger = logging.getLogger("app.services.factory")


@dataclass
class ServiceBundle:
    """Backend-selected service instances for the current process."""

    backend: str
    devices: list[str]
    runner: CompressionService
    quantize: CompressionService
    distill: CompressionService
    inference: InferenceService

    def describe(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "devices": self.devices,
            "runner": self.runner.name,
            "quantize": self.quantize.name,
            "distill": self.distill.name,
            "inference": self.inference.name,
            "cuda_visible_devices": settings.CUDA_VISIBLE_DEVICES,
        }


class ServiceFactory:
    """Build and cache a :class:`ServiceBundle` for one backend."""

    _cache: ClassVar[dict[str, ServiceBundle]] = {}

    @classmethod
    def get(cls, backend: str | None = None) -> ServiceBundle:
        backend = (backend or resolve_backend()).lower()
        if backend in cls._cache:
            return cls._cache[backend]

        if backend == "gpu":
            # Lazy imports keep CPU processes free of GPU packages.
            from app.services.compression_gpu import GpuCompressionRunner
            from app.services.distill_gpu import DistillGpuService
            from app.services.inference_gpu import InferenceGpuService
            from app.services.quantize_gpu import QuantizeGpuService

            runner = GpuCompressionRunner()
            quantize: CompressionService = QuantizeGpuService()
            distill: CompressionService = DistillGpuService()
            inference: InferenceService = InferenceGpuService()
        else:
            from app.services.compression_cpu import CpuCompressionRunner
            from app.services.distill_cpu import DistillCpuService
            from app.services.inference_cpu import InferenceCpuService
            from app.services.quantize_cpu import QuantizeCpuService

            runner = CpuCompressionRunner()
            quantize = QuantizeCpuService()
            distill = DistillCpuService()
            inference = InferenceCpuService()

        bundle = ServiceBundle(
            backend=backend,
            devices=resolve_devices(backend),
            runner=runner,
            quantize=quantize,
            distill=distill,
            inference=inference,
        )
        cls._cache[backend] = bundle
        logger.info("service bundle ready: %s", bundle.describe())
        return bundle


def get_service_bundle(backend: str | None = None) -> ServiceBundle:
    return ServiceFactory.get(backend)
