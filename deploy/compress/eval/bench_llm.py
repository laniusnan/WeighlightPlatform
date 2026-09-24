#!/usr/bin/env python3
"""Measure LLM decode latency / throughput / perplexity / peak RSS (one side per run).

    python bench_llm.py --side baseline --model /models/Qwen2.5-0.5B \
        --corpus /work/datasets-derived/alpaca-eval.txt --out /work/out/20-qwen-int4/bench_baseline.json
    python bench_llm.py --side artifact --model /work/out/20-qwen-int4/artifact \
        --corpus /work/datasets-derived/alpaca-eval.txt --out /work/out/20-qwen-int4/bench_artifact.json

`--side baseline` uses transformers on CPU; `--side artifact` loads an OpenVINO IR
directory through optimum-intel (the same stack the platform's LLM inference uses).
"""
from __future__ import annotations

import argparse
import json
import resource
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT = "请用一句话解释什么是焊接缺陷检测："


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", choices=("baseline", "artifact"), required=True)
    parser.add_argument("--model", required=True, help="HF model dir (baseline) / OpenVINO IR dir (artifact)")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--ppl-window", type=int, default=256)
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def load(side: str, model_path: str):
    if side == "artifact":
        from optimum.intel import OVModelForCausalLM

        model_dir = Path(model_path)
        load_dir = model_dir
        if not list(model_dir.glob("openvino_model.xml")):
            # The platform stores the IR as model.xml/model.bin; optimum only
            # auto-detects openvino_model.*, so expose both names via symlinks.
            staging = Path("/tmp/ov-bench-model")
            staging.mkdir(parents=True, exist_ok=True)
            rename = {"model.xml": "openvino_model.xml", "model.bin": "openvino_model.bin"}
            for name in ("model.xml", "model.bin", "config.json", "generation_config.json",
                         "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                         "vocab.json", "merges.txt", "added_tokens.json"):
                source = model_dir / name
                if not source.is_file():
                    continue
                target = staging / rename.get(name, name)
                if target.exists() or target.is_symlink():
                    target.unlink()
                target.symlink_to(source)
            load_dir = staging
        model = OVModelForCausalLM.from_pretrained(str(load_dir), export=False)
        tokenizer = AutoTokenizer.from_pretrained(str(load_dir))
        return model, tokenizer, "openvino-ir-int4"
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer, "hf-fp32-cpu"


def measure_decode(model, tokenizer, side: str, max_new_tokens: int) -> dict:
    inputs = tokenizer(PROMPT, return_tensors="pt")
    input_ids = inputs["input_ids"]
    latencies: list[float] = []
    generated_ids = input_ids
    with torch.no_grad():
        for _ in range(3):  # warmup
            model(input_ids)
        for _ in range(max_new_tokens):
            started = time.perf_counter()
            output = model(generated_ids)
            logits = output.logits if hasattr(output, "logits") else output[0]
            next_token = logits[:, -1, :].argmax(-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, torch.as_tensor(next_token)], dim=-1)
            latencies.append((time.perf_counter() - started) * 1000.0)
    latencies.sort()

    def pct(p: float) -> float:
        k = min(len(latencies) - 1, int(round((p / 100.0) * (len(latencies) - 1))))
        return round(latencies[k], 2)

    text = tokenizer.batch_decode(generated_ids)[0]
    return {
        "generated_text": text[:200],
        "decode_tokens": max_new_tokens,
        "decode_latency_ms": {"p50": pct(50), "p99": pct(99), "mean": round(statistics.fmean(latencies), 2)},
        "decode_tokens_per_s": round(1000.0 / statistics.fmean(latencies), 2),
    }


def measure_perplexity(model, tokenizer, corpus: str, window: int) -> dict:
    ids = tokenizer(corpus, return_tensors="pt")["input_ids"]
    total_nll = 0.0
    total_tokens = 0
    with torch.no_grad():
        for start in range(0, max(ids.shape[1] - 1, 1), window):
            chunk = ids[:, start:start + window]
            if chunk.shape[1] < 2:
                continue
            logits = model(chunk).logits
            shift_logits = logits[:, :-1, :]
            shift_labels = chunk[:, 1:]
            loss = torch.nn.functional.cross_entropy(
                shift_logits.reshape(-1, shift_logits.shape[-1]),
                shift_labels.reshape(-1),
                reduction="sum",
            )
            total_nll += float(loss)
            total_tokens += int(shift_labels.numel())
    return {
        "corpus_tokens": int(ids.shape[1]),
        "ppl_tokens": total_tokens,
        "perplexity": round(float(torch.exp(torch.tensor(total_nll / max(total_tokens, 1)))), 4),
    }


def main() -> int:
    args = parse_args()
    started = time.time()
    model, tokenizer, kind = load(args.side, args.model)
    load_s = round(time.time() - started, 2)

    report = {
        "side": args.side,
        "kind": kind,
        "model": args.model,
        "model_bytes": sum(f.stat().st_size for f in Path(args.model).rglob("*") if f.is_file()),
        "load_s": load_s,
        "prompt": PROMPT,
    }
    report.update(measure_decode(model, tokenizer, args.side, args.max_new_tokens))
    report.update(measure_perplexity(model, tokenizer, Path(args.corpus).read_text(encoding="utf-8"),
                                     args.ppl_window))
    report["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)
    report["total_s"] = round(time.time() - started, 1)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
