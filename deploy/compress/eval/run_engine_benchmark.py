#!/usr/bin/env python3
"""Run the platform's own LLM benchmark (engine.run_benchmark) on an artifact.

Why: the NNCF INT4 IR produced by `--algo=nncf_int4` uses non-standard output
naming (anonymous outputs, `stateful_runtime=false`), so stock optimum-intel
cannot load it — but the platform's own runner (`app.jobs.llm_runtime_nncf`)
can. This script measures original vs compressed text-generation throughput with
exactly the code path the platform uses.

    python run_engine_benchmark.py --original /models/Qwen2.5-0.5B \
        --artifact /work/out/20-qwen-int4/artifact --out /work/out/20-qwen-int4/bench_engine.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/backend")

# The engine imports the platform settings at import time (same placeholders the
# CLI uses); set them before the import so nothing touches a real database.
os.environ.setdefault("SECRET_KEY", "compress-eval")
os.environ.setdefault("PROJECT_NAME", "compress-eval")
os.environ.setdefault("DATABASE_URL", "postgresql://eval:eval@127.0.0.1:5432/eval")
os.environ.setdefault("FIRST_SUPERUSER", "eval@example.com")
os.environ.setdefault("FIRST_SUPERUSER_PASSWORD", "eval")
os.environ.setdefault("MODEL_STORAGE_PATH", "/tmp/compress-eval-storage")

from app.jobs import compression as engine  # noqa: E402


class ModelShim:
    """Minimal stand-in for the platform's Model row (engine reads these attrs)."""

    def __init__(self, file_path: str, name: str, framework: str, size_bytes: int) -> None:
        self.file_path = file_path
        self.name = name
        self.framework = framework
        self.size_bytes = size_bytes
        self.source_url = None
        self.version = 1
        self.id = "eval"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", required=True, help="HF model directory")
    parser.add_argument("--artifact", required=True, help="compressed artifact (OV IR dir)")
    parser.add_argument("--out", default=None)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=12)
    parser.add_argument("--prompt", default="请用一句话解释什么是焊接缺陷检测：")
    args = parser.parse_args()

    original = Path(args.original)
    size = sum(f.stat().st_size for f in original.rglob("*") if f.is_file())
    model = ModelShim(str(original), original.name, "huggingface", size)
    cfg = {
        "iterations": args.iterations,
        "warmup": args.warmup,
        "llm_tokens": args.tokens,
        "llm_prompt": args.prompt,
    }
    started = time.time()
    result = engine.run_benchmark(model, Path(args.artifact), cfg, lambda percent, msg="": None)
    result["elapsed_s"] = round(time.time() - started, 1)
    result["artifact_bytes"] = sum(f.stat().st_size for f in Path(args.artifact).rglob("*") if f.is_file())
    result["original_bytes"] = size
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
