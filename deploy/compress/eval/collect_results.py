#!/usr/bin/env python3
"""Aggregate every measured task into one summary JSON (report + method cards).

Reads the artefacts each task left behind:
  out/<task>/result.json   (CLI job report: artifact_bytes, duration_s, metrics, eval)
  out/<task>/bench.json    (bench.py: accuracy, p50/p99, throughput, peak RSS)

    python collect_results.py --root /home/lanius/compression-eval \
        --baseline-bytes 392548 --out /home/lanius/compression-eval/out/summary.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/home/lanius/compression-eval")
    parser.add_argument("--out", required=True)
    parser.add_argument("--baseline-bytes", type=int, default=392548,
                        help="baseline .pt bytes (CifarCNN10 checkpoint)")
    parser.add_argument("--baseline-accuracy", type=float, default=0.6879)
    parser.add_argument("--llm-model-dir", default="/home/lanius/models-local/Qwen2.5-0.5B",
                        help="LLM baseline directory (size is used for the LLM cards)")
    args = parser.parse_args()

    root = Path(args.root)
    tasks: dict[str, dict] = {}
    for task_dir in sorted((root / "out").iterdir()):
        if not task_dir.is_dir() or task_dir.name in {"baseline-pt"}:
            continue
        result = load(task_dir / "result.json")
        bench = load(task_dir / "bench.json")
        bench_baseline = load(task_dir / "bench_baseline.json")
        bench_artifact = load(task_dir / "bench_artifact.json")
        bench_engine = load(task_dir / "bench_engine.json")
        exit_file = root / "out" / f"{task_dir.name}.exit"
        record: dict = {
            "task": task_dir.name,
            "cli_exit": int(exit_file.read_text().strip()) if exit_file.is_file() else None,
        }
        if result:
            metrics = result.get("metrics") or {}
            record.update({
                "method": result.get("method"),
                "algo": result.get("algo"),
                "artifact_bytes": result.get("artifact_bytes"),
                "artifact_resolved": result.get("artifact_resolved"),
                "duration_s": result.get("duration_s"),
                "eval": metrics.get("eval"),
                "eval_error": metrics.get("eval_error"),
                "metrics": {k: v for k, v in metrics.items() if k not in {"eval"}},
            })
        if bench:
            record.update({
                "bench_kind": bench.get("kind"),
                "bench_bytes": bench.get("bytes"),
                "bench_accuracy": bench.get("accuracy"),
                "latency_ms": bench.get("latency_ms"),
                "throughput_samples_per_s": bench.get("throughput_samples_per_s"),
                "peak_rss_mb": bench.get("peak_rss_mb"),
                "model_load_s": bench.get("model_load_s"),
            })
        if bench_baseline or bench_artifact:
            record["llm_bench"] = {"baseline": bench_baseline, "artifact": bench_artifact}
        if bench_engine:
            record["engine_bench"] = bench_engine
        artifact_dir = task_dir / "artifact"
        if artifact_dir.is_dir():
            files = {p.name: p.stat().st_size for p in artifact_dir.glob("*") if p.is_file()}
            if files:
                record["artifact_files"] = files
        if record.get("artifact_bytes"):
            record["compression_pct_vs_baseline"] = round(
                (1 - record["artifact_bytes"] / args.baseline_bytes) * 100, 2)
        if record.get("bench_bytes"):
            record["compression_pct_vs_baseline"] = round(
                (1 - record["bench_bytes"] / args.baseline_bytes) * 100, 2)
        if record.get("bench_accuracy") is not None:
            record["accuracy_delta_vs_baseline"] = round(
                record["bench_accuracy"] - args.baseline_accuracy, 4)
        tasks[task_dir.name] = record

    baseline_bench = load(root / "out" / "baseline-pt" / "bench.json")
    llm_dir = Path(args.llm_model_dir)
    llm_bytes = (
        sum(f.stat().st_size for f in llm_dir.rglob("*") if f.is_file())
        if llm_dir.is_dir() else None
    )
    summary = {
        "baseline": {
            "model": "CifarCNN10 (trained on server, CIFAR10-train, 12 epochs)",
            "bytes": args.baseline_bytes,
            "accuracy": args.baseline_accuracy,
            "bench": baseline_bench,
        },
        "llm_baseline": {
            "model": "Qwen2.5-0.5B (server-local HF directory)",
            "bytes": llm_bytes,
        },
        "llm_baseline_bytes": llm_bytes,
        "tasks": tasks,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.out}: {len(tasks)} tasks")
    print(f"{'task':<22}{'bytes':>10}{'Δ%':>8}{'acc':>8}{'p50':>8}{'p99':>8}{'dur':>9}")
    for name, record in tasks.items():
        print(f"{name:<22}{record.get('bench_bytes') or record.get('artifact_bytes') or 0:>10}"
              f"{record.get('compression_pct_vs_baseline') or 0:>8}"
              f"{record.get('bench_accuracy') if record.get('bench_accuracy') is not None else '-':>8}"
              f"{record.get('latency_ms', {}).get('p50', '-'):>8}"
              f"{record.get('latency_ms', {}).get('p99', '-'):>8}"
              f"{record.get('duration_s', '-'):>9}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
