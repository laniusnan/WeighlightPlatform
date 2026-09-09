#!/usr/bin/env python
"""Local GPU real-model validation for llm_finetune (HF LoRA SFT).

Usage (GPU venv, project backend dir):
    PYTHONPATH=. \
    PROJECT_NAME=T SECRET_KEY=0123456789abcdef0123456789abcdef \
    FIRST_SUPERUSER=a@b.c FIRST_SUPERUSER_PASSWORD=pw123 \
    DATABASE_URL=postgresql://postgres:pw@localhost:5432/app \
    REDIS_URL=redis://localhost:6379/0 \
    ../.venv311gpu/bin/python ../scripts/local-gpu-qwen-validation.py

Prereq: Qwen/Qwen2.5-0.5B snapshot already in ~/.cache/huggingface
(下载: HF_ENDPOINT=https://hf-mirror.com huggingface_hub.snapshot_download("Qwen/Qwen2.5-0.5B"))
"""
from __future__ import annotations

import json
import shutil
import tempfile
import time
from pathlib import Path

from app.jobs.compression import llm_finetune


def _hf_snapshot(repo: str) -> Path:
    root = Path.home() / ".cache/huggingface/hub"
    matches = [d for d in root.glob(f"models--{repo.replace('/', '--')}/snapshots/*") if d.is_dir()]
    if not matches:
        raise SystemExit(f"no cached snapshot for {repo}; download it first")
    return sorted(matches)[0]


def main() -> None:
    base = _hf_snapshot("Qwen/Qwen2.5-0.5B")
    stub = type("M", (), {})()
    stub.framework = "huggingface"
    stub.file_path = str(base)
    stub.source_url = ""
    root = Path(tempfile.mkdtemp(prefix="qwen_val_"))
    out = root / "out"
    out.mkdir()
    data = "\n".join(
        [
            "hello => hi there",
            "what is your name? => my name is lanius",
            "今天天气怎么样 => 今天天气很好",
            "1+1等于多少 => 等于2",
            "你是什么模型 => 我是Qwen小型微调模型",
            "推荐一种水果 => 苹果",
        ]
    )
    t0 = time.time()
    _dir, result = llm_finetune(
        stub,
        {"data": data, "epochs": 2, "lr": 3e-4, "lora_rank": 8, "hf_max_len": 160},
        out,
        lambda *_: None,
    )
    print("QWEN_REAL", json.dumps(result, ensure_ascii=False))
    print("SECS", round(time.time() - t0, 1), "FILES", sorted(p.name for p in out.iterdir()))
    shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
