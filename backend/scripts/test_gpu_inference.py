#!/usr/bin/env python
"""Local NVIDIA-GPU smoke test for InferenceGpuService.

Run from a GPU-ready interpreter (e.g. .venv311gpu):

    cd backend
    PYTHONPATH=. ../.venv311gpu/bin/python scripts/test_gpu_inference.py \
        /home/lanius/Model/Qwen2.5-0.5B-Instruct "用一句话介绍你自己" 24
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

for _k, _v in {
    "PROJECT_NAME": "test-gpu",
    "SECRET_KEY": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    "FIRST_SUPERUSER": "a@b.c",
    "FIRST_SUPERUSER_PASSWORD": "pw123",
    "DATABASE_URL": "postgresql://postgres:pw@localhost:5432/app",
    "REDIS_URL": "redis://localhost:6379/0",
}.items():
    os.environ.setdefault(_k, _v)


def main() -> int:
    from app.services import get_service_bundle

    repo = sys.argv[1] if len(sys.argv) > 1 else "/home/lanius/Model/Qwen2.5-0.5B-Instruct"
    prompt = sys.argv[2] if len(sys.argv) > 2 else "用一句话介绍你自己"
    max_new = int(sys.argv[3]) if len(sys.argv) > 3 else 24

    bundle = get_service_bundle("gpu")
    result = bundle.inference.chat_completion(
        repo, prompt, max_new_tokens=max_new, temperature=0.3, top_p=0.9, min_p=0.0
    )
    print("== GPU chat result ==")
    print("text  :", result["text"])
    for key in ("prompt_tokens", "completion_tokens", "total_tokens",
                "load_seconds", "latency_ms", "device", "backend"):
        print(f"{key:<16}: {result[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
