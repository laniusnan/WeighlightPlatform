#!/usr/bin/env python3
"""
End-to-end check for llm_quantize (weight-only RTN INT4, OpenVINO).

Creates a tiny offline HuggingFace-format GPT-2 repo, then runs the exact
job function used by the platform. Requires a working optimum/optimum-intel
environment (see docs/portable-features.md notes on the NormalizedConfig bug).

Usage:
  .venv/bin/python scripts/e2e-llm-rtn.py
"""
import os, sys, tempfile
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("FASTAPI_ENV", "development")
os.environ.setdefault("PROJECT_NAME", "E2E")
for k, v in {
    "SECRET_KEY": "x" * 32,
    "POSTGRES_PASSWORD": "x" * 32,
    "FIRST_SUPERUSER": "a@b.c",
    "FIRST_SUPERUSER_PASSWORD": "x" * 8,
    "DATABASE_URL": "postgresql://postgres:x@localhost:5432/app",
    "REDIS_URL": "redis://localhost:6379/0",
}.items():
    os.environ.setdefault(k, v)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from transformers import GPT2Config, GPT2LMHeadModel  # noqa: E402
from app.jobs.compression import llm_quantize  # noqa: E402

print("building tiny local HF gpt2 ...", flush=True)
cfg = GPT2Config(vocab_size=128, n_positions=64, n_embd=32, n_layer=1, n_head=2, n_inner=64)
model_obj = GPT2LMHeadModel(cfg)
repo = Path(tempfile.mkdtemp()) / "tinygpt2"
model_obj.save_pretrained(str(repo))
model = SimpleNamespace(file_path=str(repo), source_url=None)
out = repo.parent / "ov_out"
out.mkdir()
metrics = llm_quantize(
    model,
    {"bits": 4, "group_size": 128, "sym": "true"},
    out,
    lambda *_: None,
)[1]
print("E2E_RTN_OK", metrics)
print("FILES", ",".join(sorted(p.name for p in out.iterdir())))
