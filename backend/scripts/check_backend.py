#!/usr/bin/env python
"""Debug helper: report the configured compute backend and service instances.

Usage::

    python backend/scripts/check_backend.py                # uses .env defaults
    COMPUTE_BACKEND=gpu python backend/scripts/check_backend.py
    python backend/scripts/check_backend.py --list-methods

Read-only: never touches the database, the queue or any model files.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Allow running straight from the repo root or from backend/.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # backend/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

# Config requires these; harmless defaults when .env is missing.
for _key, _val in {
    "PROJECT_NAME": "check",
    "SECRET_KEY": "check-secret-key-0123456789abcdef0123456789abcdef",
    "FIRST_SUPERUSER": "check@example.com",
    "FIRST_SUPERUSER_PASSWORD": "checkpass123",
    "DATABASE_URL": "postgresql://postgres:pw@localhost:5432/app",
}.items():
    os.environ.setdefault(_key, _val)


def _probe(lib: str) -> str:
    try:
        module = __import__(lib)
        return getattr(module, "__version__", "installed")
    except Exception as exc:  # pragma: no cover
        return f"not available ({type(exc).__name__})"


def main() -> int:
    from app.core.config import settings
    from app.services import (
        QUANTIZE_METHODS,
        TRAIN_METHODS,
        configure_runtime,
        get_service_bundle,
        resolve_backend,
        resolve_devices,
    )

    backend = resolve_backend()
    runtime = configure_runtime(backend)
    bundle = get_service_bundle(backend)

    print("=" * 64)
    print("compute-backend check")
    print("=" * 64)
    print(f"COMPUTE_BACKEND        : {settings.COMPUTE_BACKEND}")
    print(f"GPU_DEVICE_IDS         : {settings.GPU_DEVICE_IDS}")
    print(f"CUDA_VISIBLE_DEVICES   : {settings.CUDA_VISIBLE_DEVICES!r} (env: {os.environ.get('CUDA_VISIBLE_DEVICES')!r})")
    print(f"effective backend      : {runtime['backend']}")
    print(f"effective devices      : {runtime['devices']}")
    print(f"torch                  : {_probe('torch')}")
    print(f"openvino               : {_probe('openvino')}")
    print(f"onnxruntime            : {_probe('onnxruntime')}")
    if backend == "gpu":
        try:
            import torch

            print(
                f"torch.cuda.available   : {torch.cuda.is_available()} "
                f"(count={torch.cuda.device_count()})"
            )
        except Exception as exc:
            print(f"torch.cuda.available   : probe failed ({exc})")
    print("-" * 64)
    print("service instances")
    for label, obj in [
        ("runner   (all methods)", bundle.runner),
        ("quantize", bundle.quantize),
        ("distill ", bundle.distill),
        ("inference", bundle.inference),
    ]:
        print(f"  {label:<22}: {obj.name:<22} backend={getattr(obj, 'backend', '?')}")
    print("-" * 64)
    print(f"quantize-family methods : {sorted(QUANTIZE_METHODS)}")
    print(f"train-family methods    : {sorted(TRAIN_METHODS)}")

    if "--list-methods" in sys.argv:
        print("-" * 64)
        print("method support matrix")
        for method in sorted(bundle.runner.supported_methods):
            q = "y" if method in QUANTIZE_METHODS else "n"
            t = "y" if method in TRAIN_METHODS else "n"
            print(f"  {method:<26} quantize_family={q} train_family={t}")

    if backend == "gpu":
        print("-" * 64)
        print("NOTE: compression GPU runners are skeletons (tasks fail cleanly);")
        print("      inference.gpu IS implemented - run:")
        print("        PYTHONPATH=. ../.venv311gpu/bin/python scripts/test_gpu_inference.py")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
