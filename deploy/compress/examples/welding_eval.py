#!/usr/bin/env python3
"""welding_eval.py — 表格分类模型（焊接缺陷判定）精度/性能评测脚本。

用途：在**与训练完全一致**的测试集切分上，对比多个模型产物（.pt / .onnx /
OpenVINO IR 目录）的准确率、loss、per-class F1 与单样本延迟，用于验证压缩后
「精度损失 < 3%」这类交付指标。

切分与训练脚本 scripts/train_transformer.py 完全一致：

    x_train, x_tmp = train_test_split(x, y, test_size=0.26, random_state=SEED, stratify=y)
    x_val, x_test = train_test_split(x_tmp, y_tmp, test_size=0.5, random_state=SEED, stratify=y_tmp)

归一化使用 model/welding_transformer_config.json 中的 mean/std（与训练一致）。

用法
----
    python welding_eval.py \\
      --csv  <pkg>/data/training/ds_41631c12a3_welding_training.csv \\
      --config <pkg>/model/welding_transformer_config.json \\
      --arch  <pkg>/scripts/welding_transformer.py \\
      --baseline <pkg>/model/welding_transformer.pt \\
      --candidate <output>/artifact/model_quantized.onnx \\
      --out eval.json

退出码：0 成功；2 参数错误；3 输入缺失；5 评测失败。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate tabular classification models (welding demo).")
    parser.add_argument("--csv", required=True, help="训练 CSV（含特征列与标签列）")
    parser.add_argument("--config", required=True, help="模型 config.json（features/labels/mean/std）")
    parser.add_argument("--arch", default=None, help="模型定义 .py（评测 .pt 检查点时用于重建类）")
    parser.add_argument("--baseline", required=True, help="基线模型：.pt / .onnx / IR 目录")
    parser.add_argument("--candidate", default=None, help="压缩后模型：.pt / .onnx / IR 目录")
    parser.add_argument("--label-column", default="缺陷类型", help="CSV 中的标签列名")
    parser.add_argument("--seed", type=int, default=20260622, help="切分随机种子（与训练一致）")
    parser.add_argument("--batch-size", type=int, default=512, help="推理批大小")
    parser.add_argument("--latency-samples", type=int, default=2000, help="延迟测量样本数")
    parser.add_argument("--out", default=None, help="结果 JSON 输出路径")
    parser.add_argument("--json", action="store_true", help="仅输出 JSON")
    return parser.parse_args()


def _read_csv(csv_path: Path, columns: list[str]) -> dict[str, list[str]]:
    """用标准库读取 CSV（镜像内不一定有 pandas）。"""
    import csv

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        missing = [col for col in columns if col not in header]
        if missing:
            raise SystemExit(f"missing columns in {csv_path}: {missing}")
        table: dict[str, list[str]] = {col: [] for col in columns}
        for row in reader:
            for col in columns:
                table[col].append(row[col])
    return table


def load_split(
    csv_path: Path, config: dict[str, Any], label_column: str, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """返回 (x_test_norm, y_test, x_val_norm, y_val)，切分与训练脚本完全一致。"""
    from sklearn.model_selection import train_test_split

    features = list(config["features"])
    labels = list(config["labels"])
    label_to_id = {label: idx for idx, label in enumerate(labels)}

    table = _read_csv(csv_path, features + [label_column])
    x = np.asarray(
        [[float(value) for value in row] for row in zip(*(table[col] for col in features))],
        dtype=np.float32,
    )
    raw_labels = table[label_column]
    unknown = sorted({value for value in raw_labels if value not in label_to_id})
    if unknown:
        raise SystemExit(f"unknown labels: {unknown}")
    y = np.asarray([label_to_id[value] for value in raw_labels], dtype=np.int64)

    x_train, x_tmp, _y_train, y_tmp = train_test_split(
        x, y, test_size=0.26, random_state=seed, stratify=y
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_tmp, y_tmp, test_size=0.5, random_state=seed, stratify=y_tmp
    )

    mean = np.asarray(config["preprocessing"]["mean"], dtype=np.float32)
    std = np.asarray(config["preprocessing"]["std"], dtype=np.float32)

    def norm(values: np.ndarray) -> np.ndarray:
        return ((values - mean) / std).astype("float32")

    return norm(x_test), y_test, norm(x_val), y_val


def _torch_predictor(
    path: Path, arch: Path | None, config: dict[str, Any]
) -> Callable[[np.ndarray], np.ndarray]:
    """重建 torch 模型（支持 {config, model_state_dict} + build_model(config)）。"""
    import importlib.util

    import torch

    module = None
    if arch is not None:
        spec = importlib.util.spec_from_file_location("_eval_arch", arch)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules["_eval_arch"] = module
            spec.loader.exec_module(module)

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict") or checkpoint.get("state_dict") or checkpoint
    model = None
    if module is not None and hasattr(module, "build_model"):
        model = module.build_model(config)
    if model is None and module is not None:
        for name in dir(module):
            obj = getattr(module, name)
            if isinstance(obj, type) and issubclass(obj, torch.nn.Module):
                try:
                    model = obj(len(config["features"]), len(config["labels"]))
                    break
                except Exception:
                    continue
    if model is None:
        raise SystemExit(f"cannot rebuild torch model from {path} (pass --arch)")
    model.load_state_dict(state)
    model.eval()

    def predict(batch: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return model(torch.from_numpy(batch)).numpy()

    return predict


def _onnx_predictor(path: Path) -> Callable[[np.ndarray], np.ndarray]:
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    meta = session.get_inputs()[0]
    input_name = meta.name
    batch_dim = meta.shape[0] if meta.shape else None

    def _run(batch: np.ndarray) -> np.ndarray:
        return session.run(None, {input_name: batch})[0]

    def predict(batch: np.ndarray) -> np.ndarray:
        # 有些导出的图声明的是动态 batch，但内部 Reshape 写死了单条形状
        # （batch>1 会报 input_shape_size == requested_shape_size）——发现后自动逐条推理
        if getattr(predict, "batch_limit", None) == 1 and batch.shape[0] > 1:
            return np.concatenate([_run(batch[i : i + 1]) for i in range(batch.shape[0])])
        try:
            return _run(batch)
        except Exception:
            if batch.shape[0] <= 1:
                raise
            predict.batch_limit = 1
            print("  [warn] 该模型只能 batch=1 推理（图内 Reshape 为固定形状），已自动逐条推理")
            return np.concatenate([_run(batch[i : i + 1]) for i in range(batch.shape[0])])

    # 静态 batch 的产物（如 batch=1）需要按固定批量喂入
    predict.batch_limit = batch_dim if isinstance(batch_dim, int) and batch_dim > 0 else None
    return predict


def _openvino_predictor(xml_path: Path) -> Callable[[np.ndarray], np.ndarray]:
    import openvino as ov

    compiled = ov.Core().compile_model(str(xml_path), "CPU")
    meta = next(iter(compiled.inputs))
    input_name = meta.any_name
    output_key = next(iter(compiled.outputs))
    dims = list(meta.partial_shape)
    batch_limit = None
    if dims and not dims[0].is_dynamic and dims[0].get_length() > 0:
        batch_limit = int(dims[0].get_length())

    def predict(batch: np.ndarray) -> np.ndarray:
        return compiled({input_name: batch})[output_key]

    predict.batch_limit = batch_limit
    return predict


def build_predictor(
    path: Path, arch: Path | None, config: dict[str, Any]
) -> Callable[[np.ndarray], np.ndarray]:
    if not path.exists():
        raise SystemExit(f"model not found: {path}")
    if path.is_dir():
        xml = path / "model.xml"
        if not xml.exists():
            raise SystemExit(f"no model.xml inside {path}")
        return _openvino_predictor(xml)
    suffix = path.suffix.lower()
    if suffix == ".onnx":
        return _onnx_predictor(path)
    if suffix in {".pt", ".pth"}:
        return _torch_predictor(path, arch, config)
    raise SystemExit(f"unsupported model file: {path}")


def file_size(path: Path) -> int:
    if path.is_dir():
        return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    return path.stat().st_size


def evaluate(
    name: str,
    model_path: Path,
    predictor: Callable[[np.ndarray], np.ndarray],
    x_test: np.ndarray,
    y_test: np.ndarray,
    labels: list[str],
    batch_size: int,
    latency_samples: int,
) -> dict[str, Any]:
    num_classes = len(labels)
    limit = getattr(predictor, "batch_limit", None)
    batch_size = int(limit) if limit else int(batch_size)
    preds = np.empty(y_test.shape[0], dtype=np.int64)
    logits_all = np.empty((y_test.shape[0], num_classes), dtype=np.float32)
    started = time.perf_counter()
    for start in range(0, y_test.shape[0], batch_size):
        chunk = x_test[start:start + batch_size]
        out = np.asarray(predictor(chunk), dtype=np.float32)
        if out.ndim == 1:
            out = out.reshape(chunk.shape[0], -1)
        logits_all[start:start + out.shape[0]] = out
        preds[start:start + out.shape[0]] = out.argmax(axis=1)
    total_seconds = time.perf_counter() - started

    accuracy = float((preds == y_test).mean())
    clipped = np.clip(logits_all, -50, 50)
    exp = np.exp(clipped - clipped.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)
    loss = float(-np.log(np.clip(probs[np.arange(y_test.shape[0]), y_test], 1e-12, 1.0)).mean())

    report: dict[str, Any] = {}
    for class_id, label in enumerate(labels):
        tp = int(((preds == class_id) & (y_test == class_id)).sum())
        fp = int(((preds == class_id) & (y_test != class_id)).sum())
        fn = int(((preds != class_id) & (y_test == class_id)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        report[label] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1-score": round(f1, 6),
            "support": int((y_test == class_id).sum()),
        }

    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for truth, pred in zip(y_test, preds):
        matrix[truth, pred] += 1

    latency_count = min(int(latency_samples), y_test.shape[0])
    single = x_test[:1]
    for _ in range(10):  # warm-up
        predictor(single)
    latency_started = time.perf_counter()
    for index in range(latency_count):
        predictor(x_test[index:index + 1])
    latency_ms = (time.perf_counter() - latency_started) / latency_count * 1000

    return {
        "name": name,
        "model_path": str(model_path),
        "size_bytes": file_size(model_path),
        "samples": int(y_test.shape[0]),
        "accuracy": round(accuracy, 6),
        "loss": round(loss, 6),
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "batch_seconds": round(total_seconds, 3),
        "latency_ms_per_sample": round(latency_ms, 4),
    }


def human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f}{unit}" if unit != "B" else f"{int(value)}B"
        value /= 1024
    return f"{value:.1f}GB"


def print_summary(result: dict[str, Any], baseline: dict[str, Any] | None = None) -> None:
    print("-" * 74)
    print(f"{'model':<28}{'size':>10}{'accuracy':>11}{'loss':>10}{'ms/sample':>13}")
    print("-" * 74)
    for item in [result] + ([baseline] if baseline else []):
        print(f"{Path(item['model_path']).name[:27]:<28}{human_size(item['size_bytes']):>10}"
              f"{item['accuracy']:>11.4f}{item['loss']:>10.4f}"
              f"{item['latency_ms_per_sample']:>13.3f}")
    print("-" * 74)
    if baseline:
        size_delta = 1 - result["size_bytes"] / baseline["size_bytes"]
        accuracy_delta = result["accuracy"] - baseline["accuracy"]
        print(f"压缩率（体积下降）: {size_delta * 100:.2f}%")
        print(f"精度变化          : {accuracy_delta * 100:+.2f} 个百分点（绝对损失 {abs(accuracy_delta) * 100:.2f}%）")
        print(f"loss 变化         : {result['loss'] - baseline['loss']:+.4f}")
        print(f"延迟变化          : {result['latency_ms_per_sample'] - baseline['latency_ms_per_sample']:+.3f} ms/样本")
        verdict = "通过" if (size_delta >= 0.60 and abs(accuracy_delta) < 0.03) else "未达标"
        print(f"交付判据（压缩率>60% 且 精度损失<3%）: {verdict}")
    print("-" * 74)


def main() -> int:
    args = parse_args()
    csv_path = Path(args.csv)
    config_path = Path(args.config)
    for required in (csv_path, config_path):
        if not required.exists():
            print(f"error: not found: {required}", file=sys.stderr)
            return 3
    config = json.loads(config_path.read_text(encoding="utf-8"))
    labels = list(config["labels"])
    arch = Path(args.arch) if args.arch else None

    x_test, y_test, _x_val, _y_val = load_split(csv_path, config, args.label_column, args.seed)

    baseline_path = Path(args.baseline)
    baseline = evaluate(
        "baseline", baseline_path, build_predictor(baseline_path, arch, config),
        x_test, y_test, labels, args.batch_size, args.latency_samples,
    )
    result: dict[str, Any] = {"baseline": baseline, "test_samples": int(y_test.shape[0])}

    candidate = None
    if args.candidate:
        candidate_path = Path(args.candidate)
        candidate = evaluate(
            "candidate", candidate_path, build_predictor(candidate_path, arch, config),
            x_test, y_test, labels, args.batch_size, args.latency_samples,
        )
        result["candidate"] = candidate
        result["size_reduction"] = round(1 - candidate["size_bytes"] / baseline["size_bytes"], 6)
        result["accuracy_delta"] = round(candidate["accuracy"] - baseline["accuracy"], 6)
        result["loss_delta"] = round(candidate["loss"] - baseline["loss"], 6)
        result["meets_delivery_criteria"] = bool(
            result["size_reduction"] >= 0.60 and abs(result["accuracy_delta"]) < 0.03
        )

    if args.out:
        Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"测试集: {y_test.shape[0]} 条（seed={args.seed}，分层切分，与训练一致）")
        print_summary(candidate or baseline, baseline if candidate else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

