"""Compute service layer (backend-agnostic).

Public API::

    from app.services import get_service_bundle, configure_runtime

    cfg = configure_runtime()            # returns {"backend", "devices", ...}
    bundle = get_service_bundle()        # CPU by default
    bundle.runner.run(task_id)
"""
from app.services.base import (
    ALL_COMPRESSION_METHODS,
    QUANTIZE_METHODS,
    TRAIN_METHODS,
    InferenceService,
    CompressionService,
    NotImplementedBackendError,
    configure_runtime,
    resolve_backend,
    resolve_devices,
)
from app.services.factory import ServiceBundle, ServiceFactory, get_service_bundle

__all__ = [
    "ALL_COMPRESSION_METHODS",
    "QUANTIZE_METHODS",
    "TRAIN_METHODS",
    "InferenceService",
    "CompressionService",
    "NotImplementedBackendError",
    "ServiceBundle",
    "ServiceFactory",
    "configure_runtime",
    "resolve_backend",
    "resolve_devices",
    "get_service_bundle",
]
