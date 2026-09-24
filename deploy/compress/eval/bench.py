#!/usr/bin/env python3
"""Measure accuracy / latency (p50,p99) / throughput / peak RSS of a CIFAR-10 model.

Supports the artifact kinds the platform produces: ONNX (dynamic or static batch),
OpenVINO IR directory (`model.xml`), and PyTorch checkpoints (`model.pt` + arch.py).

    python bench.py --artifact /work/out/dynamic-int8/artifact/model_quantized.onnx \
        --data /datasets/CIFAR10-test --image-size 32 --samples 2000 \
        --out /work/out/dynamic-int8/bench.json

All numbers are measured in-process; peak RSS comes from getrusage(RUSAGE_SELF),
so it covers model load + data + inference (documented in the report).
"""
from __future__ import annotations

import argparse
import json
import resource
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark a CIFAR-10 artifact")
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--data", default="/datasets/CIFAR10-test")
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--samples", type=int, default=2000, help="延迟采样样本数（batch=1）")
    parser.add_argument("--throughput-batch", type=int, default=32)
    parser.add_argument("--arch", default=None, help=".pt 需要的 arch.py")
    parser.add_argument("--out", default=None)
    parser.add_argument("--label", default=None)
    return parser.parse_args()


def resolve_artifact(path: Path) -> Path:
    """Directory artifacts (prune/qat/distill) → the concrete model file inside."""
    if path.is_file():
        return path
    for name in ("model_quantized.onnx", "model_qat.onnx", "model_pruned.onnx",
                 "model_student.onnx", "model_finetuned.onnx", "model.onnx"):
        candidate = path / name
        if candidate.is_file():
            return candidate
    for pattern in ("*.onnx", "*.pt", "*.pth"):
        found = sorted(path.glob(pattern))
        if found:
            return found[0]
    return path


def load_predictor(artifact: Path, image_size: int, arch: str | None):
    """Return (predict_fn, kind, bytes_on_disk). predict_fn takes NCHW float32."""
    if artifact.is_dir():  # OpenVINO IR
        import openvino as ov

        compiled = ov.Core().compile_model(str(artifact / "model.xml"), "CPU")
        meta = next(iter(compiled.inputs))
        name = meta.any_name
        partial = meta.partial_shape
        fixed = int(partial[0].get_length()) if len(partial) == 4 and partial[0].is_static else None
        size = sum(f.stat().st_size for f in artifact.rglob("*") if f.is_file())

        def predict(batch: np.ndarray) -> np.ndarray:
            n = batch.shape[0]
            if fixed and n > fixed:
                parts = [
                    np.asarray(compiled({name: batch[i:i + fixed]})[compiled.outputs[0]])
                    for i in range(0, n, fixed)
                ]
                return np.concatenate(parts, axis=0)
            return np.asarray(compiled({name: batch})[compiled.outputs[0]])

        return predict, f"openvino-ir(batch={'static' if fixed else 'dynamic'})", size

    suffix = artifact.suffix.lower()
    if suffix == ".onnx":
        import onnxruntime as ort

        session = ort.InferenceSession(str(artifact), providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        shape = session.get_inputs()[0].shape
        fixed = shape[0] if shape and isinstance(shape[0], int) else None

        def predict(batch: np.ndarray) -> np.ndarray:
            n = batch.shape[0]
            if fixed and n > fixed:
                parts = [session.run(None, {input_name: batch[i:i + fixed]})[0]
                         for i in range(0, n, fixed)]
                return np.concatenate(parts, axis=0)
            return session.run(None, {input_name: batch})[0]

        return predict, f"onnx(batch={'static' if fixed else 'dynamic'})", artifact.stat().st_size

    if suffix in {".pt", ".pth"}:
        import torch

        arch_path = Path(arch) if arch else artifact.parent / "arch.py"
        sys.path.insert(0, str(arch_path.parent))
        net = torch.load(str(artifact), map_location="cpu", weights_only=False)
        net.eval()

        def predict(batch: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                out = net(torch.from_numpy(batch))
            return out.numpy() if hasattr(out, "numpy") else np.asarray(out)

        return predict, "pytorch", artifact.stat().st_size

    raise SystemExit(f"unsupported artifact: {artifact}")


def main() -> int:
    args = parse_args()
    artifact = resolve_artifact(Path(args.artifact))
    if not artifact.exists():
        raise SystemExit(f"artifact not found: {artifact}")
    args.artifact = str(artifact)

    load_started = time.time()
    predict, kind, bytes_on_disk = load_predictor(artifact, args.image_size, args.arch)
    load_s = round(time.time() - load_started, 3)

    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    dataset = datasets.ImageFolder(args.data, transform=transform)
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=4)

    correct = total = 0
    for images, labels in loader:
        out = predict(images.numpy().astype(np.float32))
        correct += int((out.argmax(1) == labels.numpy()).sum())
        total += labels.numel()
    accuracy = round(correct / total, 4)

    warm = np.random.rand(1, 3, args.image_size, args.image_size).astype(np.float32)
    for _ in range(30):
        predict(warm)
    latencies: list[float] = []
    pool = np.random.default_rng(0).integers(0, total, size=min(args.samples, total))
    for index in pool:
        image, _label = dataset[int(index)]
        batch = image.unsqueeze(0).numpy().astype(np.float32)
        started = time.perf_counter()
        predict(batch)
        latencies.append((time.perf_counter() - started) * 1000.0)
    latencies.sort()

    def pct(p: float) -> float:
        k = min(len(latencies) - 1, int(round((p / 100.0) * (len(latencies) - 1))))
        return round(latencies[k], 3)

    throughput: dict[str, float] = {}
    for batch_size in sorted({1, args.throughput_batch}):
        count = 0
        started = time.perf_counter()
        while count < 512:
            idx = pool[count % len(pool):count % len(pool) + batch_size]
            if len(idx) < batch_size:
                idx = pool[:batch_size]
            images = np.stack([dataset[int(i)][0].numpy() for i in idx]).astype(np.float32)
            predict(images)
            count += batch_size
        elapsed = time.perf_counter() - started
        throughput[f"batch{batch_size}"] = round(count / elapsed, 1)

    peak_rss_mb = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)
    report = {
        "label": args.label or artifact.name,
        "artifact": str(artifact),
        "kind": kind,
        "bytes": bytes_on_disk,
        "accuracy": accuracy,
        "eval_samples": total,
        "latency_ms": {
            "p50": pct(50), "p90": pct(90), "p99": pct(99),
            "mean": round(statistics.fmean(latencies), 3),
            "samples": len(latencies),
        },
        "throughput_samples_per_s": throughput,
        "peak_rss_mb": peak_rss_mb,
        "model_load_s": load_s,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
