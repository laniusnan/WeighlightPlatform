"""CPU-friendly model compression and evaluation jobs.

All ML libraries (torch, onnx, openvino, nncf, optimum-intel, ultralytics)
are imported lazily inside the functions so that the API process can start
without them and stays light.
"""
from __future__ import annotations

import shutil
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlmodel import Session

from app.core.db import engine
from app.jobs.storage import dir_size, task_dir
from app.models import CompressionMethod, CompressionTask, Model, TaskStatus

ProgressCallback = Callable[[float, str | None], None]


def _now() -> datetime:
    return datetime.now(UTC)


class _JobCancelled(Exception):
    """Raised internally when a task was marked as canceled while running."""


def run_compression_task(task_id: object) -> None:
    """Entry point called by the Arq worker (inside a thread)."""
    with Session(engine) as session:
        task = session.get(CompressionTask, task_id)
        if task is None:
            return
        if task.status == TaskStatus.canceled.value:
            return
        model = session.get(Model, task.model_id)
        if model is None:
            task.status = TaskStatus.failed.value
            task.error_message = "Referenced model not found"
            task.finished_at = _now()
            session.add(task)
            session.commit()
            return

        task.status = TaskStatus.running.value
        task.started_at = _now()
        session.add(task)
        session.commit()

        import time as _time

        _t0 = _time.time()
        progress_history: list[list[float]] = []
        # Sync Redis client for publishing live progress samples (optional).
        _redis: Any = None
        try:
            import redis

            from app.core.config import settings

            _redis = redis.Redis.from_url(
                str(settings.REDIS_URL), decode_responses=True
            )
        except Exception:
            _redis = None

        def progress(pct: float, _message: str | None = None) -> None:
            session.refresh(task)
            if task.status == TaskStatus.canceled.value:
                raise _JobCancelled()
            task.progress = round(min(max(float(pct), 0.0), 100.0), 1)
            session.add(task)
            session.commit()
            progress_history.append([round(_time.time() - _t0, 2), task.progress])
            if len(progress_history) > 300:
                progress_history.pop(0)
            if _redis is not None:
                try:
                    key = f"platform:progress:{task.id}"
                    pipe = _redis.pipeline()
                    pipe.rpush(key, task.progress)
                    pipe.ltrim(key, -300, -1)
                    pipe.expire(key, 60 * 60)
                    pipe.execute()
                except Exception:
                    pass

        try:
            output, method_metrics = _dispatch(model, task, progress)
            output_path = Path(output)
            out_bytes = dir_size(output_path)
            in_bytes = model.size_bytes
            if in_bytes in (0, None) and model.file_path:
                in_bytes = dir_size(Path(model.file_path))

            metrics: dict[str, Any] = dict(method_metrics)
            metrics["input_bytes"] = in_bytes
            metrics["output_bytes"] = out_bytes
            if out_bytes and in_bytes:
                metrics["size_reduction_pct"] = round(
                    (1 - out_bytes / in_bytes) * 100, 2
                )
                metrics["ratio"] = round(in_bytes / out_bytes, 4)

            eval_cfg = (task.params or {}).get("eval") or {}
            if eval_cfg:
                try:
                    metrics["eval"] = run_evaluation(
                        model, output_path, task_dir(task.id), eval_cfg, progress
                    )
                except Exception as exc:  # eval must never fail the job
                    metrics["eval"] = {"error": f"{type(exc).__name__}: {exc}"[:500]}

            bench_cfg = (task.params or {}).get("benchmark") or {}
            if bench_cfg:
                try:
                    metrics["benchmark"] = run_benchmark(
                        model, output_path, bench_cfg, progress
                    )
                except Exception as exc:  # benchmark must never fail the job
                    metrics["benchmark"] = {
                        "error": f"{type(exc).__name__}: {exc}"[:500]
                    }

            task.output_path = str(output_path)
            task.output_size_bytes = out_bytes
            task.input_size_bytes = in_bytes
            metrics["progress_history"] = progress_history
            task.metrics = metrics
            task.status = TaskStatus.succeeded.value
            task.progress = 100.0
            task.finished_at = _now()
            session.add(task)
            session.commit()
        except _JobCancelled:
            task.status = TaskStatus.canceled.value
            task.finished_at = _now()
            session.add(task)
            session.commit()
        except Exception as exc:
            task.status = TaskStatus.failed.value
            task.error_message = f"{type(exc).__name__}: {exc}"[:2048]
            task.finished_at = _now()
            session.add(task)
            session.commit()


def _dispatch(
    model: Model,
    task: CompressionTask,
    progress: ProgressCallback,
) -> tuple[Path, dict[str, Any]]:
    method = task.method
    params = task.params or {}
    out_dir = task_dir(task.id)
    if model.file_path and not Path(model.file_path).exists():
        raise FileNotFoundError(f"Model file not found: {model.file_path}")

    if method == CompressionMethod.export_onnx.value:
        return export_onnx(model, params, out_dir, progress)
    if method == CompressionMethod.quantize_onnx.value:
        return quantize_onnx(model, params, out_dir, progress)
    if method == CompressionMethod.quantize_onnx_static.value:
        return quantize_onnx_static(model, params, out_dir, progress)
    if method == CompressionMethod.openvino_int8.value:
        return openvino_int8(model, params, out_dir, progress)
    if method == CompressionMethod.sparsify.value:
        return sparsify(model, params, out_dir, progress)
    if method == CompressionMethod.qat.value:
        return qat(model, params, out_dir, progress)
    if method == CompressionMethod.export_gguf.value:
        return export_gguf(model, params, out_dir, progress)
    if method == CompressionMethod.prune.value:
        return prune(model, params, out_dir, progress)
    if method == CompressionMethod.llm_quantize.value:
        return llm_quantize(model, params, out_dir, progress)
    raise ValueError(f"Unknown compression method: {method}")


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------
def _load_torch_module(file_path: Path) -> Any:
    """Load a ``torch.nn.Module`` from a checkpoint file.

    Supports:
    * ``torch.save(model)`` full-module checkpoints (with an optional companion
      ``arch.py`` that defines the model class, required when the checkpoint
      references a class the worker cannot import).
    * dict checkpoints (``{"model": model}``, ``{"model_ema": model}``, ...).
    * ``state_dict`` checkpoints when a companion ``arch.py`` (uploaded next to
      the checkpoint) defines the model class.
    """

    import torch

    import __main__ as main_module

    arch_module = _load_arch_module(file_path.parent / "arch.py")
    arch_classes = _arch_classes(arch_module)

    # Register classes on ``__main__`` so pickled references like
    # ``__main__.TinyCNN`` (saved from a training script) resolve in the worker.
    for class_name, cls in arch_classes.items():
        if not hasattr(main_module, class_name):
            setattr(main_module, class_name, cls)

    def _try_load() -> Any:
        return torch.load(file_path, map_location="cpu", weights_only=False)

    try:
        checkpoint = _try_load()
    except (ModuleNotFoundError, AttributeError) as exc:
        # The checkpoint references a module/class the worker cannot import
        # (e.g. ``arch.TinyCNN`` from a training script). Register the uploaded
        # arch module under the missing name and retry.
        if arch_module is None or not _register_arch_for_pickle(
            exc, arch_module, arch_classes
        ):
            raise ValueError(
                f"Cannot load checkpoint: {exc}. "
                "Full-model checkpoints saved with `torch.save(model)` need the "
                "model class to be importable. Upload the architecture .py file "
                "(the model class definition) together with the checkpoint, or "
                "use an ONNX model."
            ) from exc
        try:
            checkpoint = _try_load()
        except (ModuleNotFoundError, AttributeError) as exc2:
            raise ValueError(
                f"Cannot load checkpoint: {exc2}. "
                "Full-model checkpoints saved with `torch.save(model)` need the "
                "model class to be importable. Upload the architecture .py file "
                "(the model class definition) together with the checkpoint, or "
                "use an ONNX model."
            ) from exc2

    if isinstance(checkpoint, torch.nn.Module):
        return checkpoint

    if isinstance(checkpoint, dict):
        for key in ("model", "net", "module", "model_ema"):
            candidate = checkpoint.get(key)
            if isinstance(candidate, torch.nn.Module):
                return candidate

        # state_dict support: rebuild the module from the uploaded architecture.
        state_dict = _find_state_dict(checkpoint)
        if state_dict is not None:
            net = _rebuild_from_state_dict(state_dict, arch_classes)
            if net is not None:
                return net
            raise ValueError(
                "The checkpoint contains a state_dict but no matching "
                "architecture was found. Upload the architecture .py file (the "
                "model class definition) together with the checkpoint."
            )

    raise ValueError(
        "Unsupported checkpoint format: expected a torch.nn.Module, a state_dict, "
        "or a dict containing one (model / net / module / model_ema). If this is "
        "a state_dict checkpoint, upload its architecture .py file as well."
    )


def _load_arch_module(arch_path: Path) -> Any | None:
    """Import a companion ``arch.py`` file (returns ``None`` if absent)."""
    import importlib.util

    if not arch_path.exists():
        return None
    spec = importlib.util.spec_from_file_location("_uploaded_model_arch", arch_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _arch_classes(arch_module: Any | None) -> dict[str, type]:
    """Return the ``nn.Module`` subclasses defined in an arch module."""
    import torch

    if arch_module is None:
        return {}
    return {
        name: obj
        for name, obj in vars(arch_module).items()
        if isinstance(obj, type)
        and issubclass(obj, torch.nn.Module)
        and obj is not torch.nn.Module
    }


def _register_arch_for_pickle(
    exc: Exception, arch_module: Any, arch_classes: dict[str, type]
) -> bool:
    """Best-effort: make pickled class references resolvable by the worker.

    Handles ``ModuleNotFoundError: No module named 'X'`` and
    ``AttributeError: module 'X' has no attribute 'Y'`` by registering the
    uploaded arch module (or its classes) under the missing names.
    """
    import re
    import sys

    match = re.search(r"No module named '([^']+)'", str(exc))
    if match:
        name = match.group(1)
        if name and name not in sys.modules:
            sys.modules[name] = arch_module
        return True

    match = re.search(r"module '([^']+)' has no attribute '([^']+)'", str(exc))
    if match:
        module_name, attr = match.group(1), match.group(2)
        module = sys.modules.get(module_name)
        if module is None:
            sys.modules[module_name] = arch_module
        elif not hasattr(module, attr) and attr in arch_classes:
            setattr(module, attr, arch_classes[attr])
        return True
    return False


def _find_state_dict(checkpoint: dict[str, Any]) -> dict[str, Any] | None:
    """Locate a plain state_dict inside a checkpoint dict, if any."""

    for key in ("model", "net", "module", "model_ema", "state_dict"):
        candidate = checkpoint.get(key)
        if isinstance(candidate, dict) and _looks_like_state_dict(candidate):
            return candidate
    if _looks_like_state_dict(checkpoint):
        return checkpoint
    return None


def _looks_like_state_dict(obj: dict[str, Any]) -> bool:
    import torch

    return any(isinstance(value, torch.Tensor) for value in obj.values())


def _rebuild_from_state_dict(
    state_dict: dict[str, Any], arch_classes: dict[str, type]
) -> Any | None:
    """Instantiate the model from the uploaded architecture and load the weights."""
    for cls in arch_classes.values():
        try:
            net = cls()
            net.load_state_dict(state_dict)
            return net
        except Exception:
            continue
    return None


def _parameter_count(module: Any) -> int:
    return sum(p.numel() for p in module.parameters())


def _parse_input_shape(value: Any, default: list[int]) -> list[int]:
    """Parse an ``input_shape`` task param into a list of ints.

    Accepts both a list of ints (``[1, 3, 224, 224]``) and the comma-separated
    string (``"1,3,224,224"``) that the frontend form sends.
    """
    if value is None:
        return default
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        if not parts:
            return default
        return [int(part) for part in parts]
    return [int(dim) for dim in value]


# ---------------------------------------------------------------------------
# export_onnx: PyTorch -> ONNX (+ onnxsim)
# ---------------------------------------------------------------------------
def export_onnx(
    model: Model,
    params: dict[str, Any],
    out_dir: Path,
    progress: ProgressCallback,
) -> tuple[Path, dict[str, Any]]:
    """PyTorch -> ONNX (+ onnxsim simplification). Supports YOLO checkpoints."""
    model_kind = params.get("model_kind", "cnn")
    if model.file_path is None:
        raise ValueError("Model has no local file")

    if model_kind == "yolo":
        from ultralytics import YOLO

        imgsz = int(params.get("imgsz", 640))
        opset = int(params.get("opset", 12))
        progress(10, "Exporting YOLO to ONNX (ultralytics)...")
        exported = YOLO(str(Path(model.file_path))).export(
            format="onnx", imgsz=imgsz, opset=opset, simplify=True
        )
        out = out_dir / "model.onnx"
        shutil.move(str(exported), out)
        progress(95, "Done.")
        return out, {"model_kind": "yolo"}

    import onnx
    import onnxsim
    import torch

    progress(5, "Loading PyTorch model...")
    net = _load_torch_module(Path(model.file_path))
    net.eval()
    input_shape = _parse_input_shape(params.get("input_shape"), [1, 3, 224, 224])
    opset = int(params.get("opset", 12))
    dynamic_axes = bool(params.get("dynamic_axes", False))
    dummy = torch.randn(*input_shape)

    out = out_dir / "model.onnx"
    progress(30, "Exporting to ONNX...")
    torch.onnx.export(
        net,
        dummy,
        str(out),
        opset_version=opset,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=(
            {"input": {0: "batch"}, "output": {0: "batch"}}
            if dynamic_axes
            else None
        ),
        # Use the legacy TorchScript exporter so weights stay inline. The new
        # torch.export-based exporter (default since 2.9) writes large tensors
        # to a sidecar "*.onnx.data" file, which would break single-file model
        # storage/download in this platform.
        dynamo=False,
    )
    progress(70, "Simplifying with onnxsim...")
    simplified, _ = onnxsim.simplify(str(out))
    onnx.save(simplified, str(out))
    progress(95, "Done.")
    return out, {"model_kind": model_kind, "params": _parameter_count(net)}


# ---------------------------------------------------------------------------
# quantize_onnx: ONNX dynamic INT8 (onnxruntime)
# ---------------------------------------------------------------------------
def quantize_onnx(
    model: Model,
    _params: dict[str, Any],
    out_dir: Path,
    progress: ProgressCallback,
) -> tuple[Path, dict[str, Any]]:
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from onnxruntime.quantization.shape_inference import quant_pre_process

    if model.file_path is None:
        raise ValueError("Model has no local file")
    src = Path(model.file_path)
    if src.suffix.lower() != ".onnx":
        raise ValueError("quantize_onnx requires an ONNX model, got: " + src.name)

    progress(10, "Running shape inference...")
    preprocessed = out_dir / "preprocessed.onnx"
    quant_pre_process(str(src), str(preprocessed))
    out = out_dir / "model_quantized.onnx"
    progress(50, "Quantizing to INT8 (dynamic)...")
    quantize_dynamic(
        str(preprocessed),
        str(out),
        weight_type=QuantType.QInt8,
        per_channel=True,
        # Note: `optimize_model` was removed in onnxruntime >= 1.22; model
        # optimization now happens automatically before quantization.
    )
    preprocessed.unlink(missing_ok=True)
    progress(95, "Done.")
    return out, {"quant_type": "dynamic_int8"}


# ---------------------------------------------------------------------------
# quantize_onnx_static: ONNX static INT8 (QDQ) with a calibration dataset
# ---------------------------------------------------------------------------
def quantize_onnx_static(
    model: Model,
    params: dict[str, Any],
    out_dir: Path,
    progress: ProgressCallback,
) -> tuple[Path, dict[str, Any]]:
    """Static INT8 quantization of an ONNX model using real calibration data.

    Calibration images come from ``calib_data_url`` (zip/tar archive) or
    ``calib_data_dir`` (local path). Falls back to synthetic random data when
    neither is provided. Produces a QDQ (QuantizeDequantize) INT8 model.
    """
    import numpy as np
    import onnxruntime
    from onnxruntime.quantization import (
        CalibrationDataReader,
        QuantFormat,
        QuantType,
        quantize_static,
    )
    from onnxruntime.quantization.shape_inference import quant_pre_process

    if model.file_path is None:
        raise ValueError("Model has no local file")
    src = Path(model.file_path)
    if src.suffix.lower() != ".onnx":
        raise ValueError(
            "quantize_onnx_static requires an ONNX model, got: " + src.name
        )

    image_size = int(params.get("image_size", 224))
    input_shape = _parse_input_shape(
        params.get("input_shape"), [1, 3, image_size, image_size]
    )
    calibration_samples = int(params.get("calibration_samples", 16))

    progress(10, "Preparing calibration data...")
    height = input_shape[2] if len(input_shape) == 4 else image_size
    width = input_shape[3] if len(input_shape) == 4 else image_size

    import onnx

    input_name = onnx.load(str(src)).graph.input[0].name

    images: list[np.ndarray] = []
    calib_url = params.get("calib_data_url")
    calib_dir = params.get("calib_data_dir")
    calib_dataset_id = params.get("calib_dataset_id")
    if calib_url or calib_dir or calib_dataset_id:
        eval_cfg: dict[str, Any] = {
            "type": "classification",
            "data_url": calib_url,
            "data_dir": calib_dir,
            "dataset_id": calib_dataset_id,
        }
        data_dir = _resolve_eval_data(eval_cfg, out_dir / "calib")
        candidates = sorted(
            p
            for p in data_dir.rglob("*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        )
        if not candidates:
            raise ValueError("No images found in calibration data")
        for img_path in candidates:
            if len(images) >= calibration_samples:
                break
            try:
                images.append(_preprocess_image(img_path, height, width))
            except Exception:
                continue
        if not images:
            raise ValueError("Could not preprocess any calibration images")
    else:
        progress(20, "No calibration data provided, using synthetic samples...")
        for _ in range(max(calibration_samples, 1)):
            images.append(np.random.randn(*input_shape).astype(np.float32))

    class _CalibReader(CalibrationDataReader):
        def __init__(self, data: list[np.ndarray], name: str) -> None:
            self._iterator = iter(data)
            self._name = name

        def get_next(self) -> dict[str, np.ndarray] | None:
            arr = next(self._iterator, None)
            return {self._name: arr} if arr is not None else None

    progress(30, "Running shape inference...")
    preprocessed = out_dir / "preprocessed.onnx"
    quant_pre_process(str(src), str(preprocessed))

    out = out_dir / "model_quantized.onnx"
    progress(50, "Running static INT8 quantization with calibration...")
    quantize_static(
        str(preprocessed),
        str(out),
        calibration_data_reader=_CalibReader(images, input_name),
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        weight_type=QuantType.QInt8,
        reduce_range=False,
    )

    # Validate the output loads with onnxruntime; if the QDQ graph is invalid
    # (can happen on small models with int32-quantized biases), fall back to the
    # more broadly supported QOperator format.
    try:
        onnxruntime.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    except Exception:
        progress(70, "QDQ model failed validation, retrying with QOperator...")
        fallback = out_dir / "model_quantized_qop.onnx"
        quantize_static(
            str(preprocessed),
            str(fallback),
            calibration_data_reader=_CalibReader(images, input_name),
            quant_format=QuantFormat.QOperator,
            per_channel=True,
            weight_type=QuantType.QInt8,
            reduce_range=False,
        )
        out.unlink(missing_ok=True)
        out = fallback

    preprocessed.unlink(missing_ok=True)
    progress(95, "Done.")
    return out, {
        "quant_type": "static_int8",
        "calibration_samples": len(images),
        "input_shape": input_shape,
    }


# ---------------------------------------------------------------------------
# openvino_int8: OpenVINO IR + NNCF INT8 post-training quantization
# ---------------------------------------------------------------------------
def openvino_int8(
    model: Model,
    params: dict[str, Any],
    out_dir: Path,
    progress: ProgressCallback,
) -> tuple[Path, dict[str, Any]]:
    import openvino as ov

    model_kind = params.get("model_kind", "cnn")
    if model.file_path is None:
        raise ValueError("Model has no local file")

    if model_kind == "yolo":
        from ultralytics import YOLO

        imgsz = int(params.get("imgsz", 640))
        progress(10, "Exporting YOLO to OpenVINO IR (ultralytics)...")
        exported = YOLO(str(Path(model.file_path))).export(
            format="openvino", imgsz=imgsz, half=False
        )
        ov_model = ov.convert_model(str(exported))
        out_dir.mkdir(parents=True, exist_ok=True)
        ov.save_model(ov_model, str(out_dir / "model.xml"))
        return out_dir, {"model_kind": "yolo", "precision": "fp32_ir"}

    if Path(model.file_path).suffix.lower() == ".onnx":
        progress(10, "Loading ONNX model...")
        ov_model = ov.convert_model(str(Path(model.file_path)))
    else:
        import torch

        progress(10, "Loading PyTorch model...")
        net = _load_torch_module(Path(model.file_path))
        net.eval()
        input_shape = _parse_input_shape(params.get("input_shape"), [1, 3, 224, 224])
        dummy = torch.randn(*input_shape)
        progress(40, "Converting to OpenVINO IR...")
        ov_model = ov.convert_model(net, example_input=dummy)

    precision = str(params.get("precision", "int8")).lower()
    if precision in ("fp16", "fp32"):
        progress(60, f"Saving OpenVINO IR ({precision})...")
        out_dir.mkdir(parents=True, exist_ok=True)
        ov.save_model(
            ov_model,
            str(out_dir / "model.xml"),
            compress_to_fp16=(precision == "fp16"),
        )
        progress(95, "Done.")
        return out_dir, {"model_kind": model_kind, "precision": precision}

    quantized = _nncf_quantize_ov(ov_model, params, progress)
    out_dir.mkdir(parents=True, exist_ok=True)
    ov.save_model(quantized, str(out_dir / "model.xml"))
    progress(95, "Done.")
    return out_dir, {"model_kind": model_kind, "precision": "int8"}


def _nncf_quantize_ov(
    ov_model: Any, params: dict[str, Any], progress: ProgressCallback
) -> Any:
    """Apply NNCF post-training INT8 quantization to an OpenVINO model."""
    import nncf
    import numpy as np

    calibration_samples = int(params.get("calibration_samples", 8))
    try:
        preset = nncf.QuantizationPreset.M1
    except AttributeError:  # pragma: no cover - API robustness
        preset = None

    # Build synthetic calibration data from the model input shapes when no
    # real calibration set is provided.
    calibration_data = []
    for _ in range(max(calibration_samples, 1)):
        sample = {}
        for tensor in ov_model.inputs:
            shape = [dim or 1 for dim in tensor.partial_shape.get_min_shape()]
            sample[tensor.any_name] = np.random.randn(*shape).astype(np.float32)
        calibration_data.append(sample)

    progress(60, "Quantizing to INT8 with NNCF...")
    dataset = nncf.Dataset(calibration_data)
    if preset is None:
        return nncf.quantize(ov_model, dataset)
    return nncf.quantize(ov_model, dataset, preset=preset)


# ---------------------------------------------------------------------------
# prune: L2-norm structured filter pruning (PyTorch CNNs)
# ---------------------------------------------------------------------------
def prune(
    model: Model,
    params: dict[str, Any],
    out_dir: Path,
    progress: ProgressCallback,
) -> tuple[Path, dict[str, Any]]:
    import torch

    if model.file_path is None:
        raise ValueError("Model has no local file")
    net = _load_torch_module(Path(model.file_path))
    net.eval()
    rate = float(params.get("rate", 0.3))
    if not 0.0 < rate < 1.0:
        raise ValueError("prune rate must be between 0 and 1")

    progress(20, "Computing filter norms...")
    conv_layers = [
        module for module in net.modules() if isinstance(module, torch.nn.Conv2d)
    ]
    if not conv_layers:
        raise ValueError("No Conv2d layers found, pruning is not applicable")

    total_filters = sum(conv.out_channels for conv in conv_layers)
    pruned_filters = 0
    keep_lists: list[list[int]] = []

    for i, conv in enumerate(conv_layers):
        norms = conv.weight.data.view(conv.out_channels, -1).norm(dim=1)
        num_keep = max(1, int(conv.out_channels * (1 - rate)))
        keep = torch.topk(norms, num_keep, largest=True).indices.sort()[0]
        keep_list = keep.tolist()
        keep_lists.append(keep_list)

        # Prune this conv's own output filters.
        conv.weight.data = conv.weight.data[keep]
        conv.out_channels = num_keep
        if conv.bias is not None:
            conv.bias.data = conv.bias.data[keep]
        pruned_filters += len(keep_list) - num_keep

        # Trim the input channels of the following conv (sequential structure).
        if i + 1 < len(conv_layers):
            next_conv = conv_layers[i + 1]
            next_conv.weight.data = next_conv.weight.data[:, keep]
            next_conv.in_channels = num_keep

        # Adjust the BatchNorm that directly follows this conv (if any).
        for _, module in net.named_modules():
            if (
                isinstance(module, torch.nn.BatchNorm2d)
                and module.num_features == conv.out_channels
            ):
                module.weight.data = module.weight.data[keep]
                module.bias.data = module.bias.data[keep]
                module.running_mean = module.running_mean[keep]
                module.running_var = module.running_var[keep]
                module.num_batches_tracked = module.num_batches_tracked
                module.num_features = num_keep
                break

    progress(60, "Saving pruned model...")
    out = out_dir / "model_pruned.pt"
    torch.save(net, out)

    # Also export to ONNX when the model is traceable.
    result_metrics: dict[str, Any] = {
        "total_filters": total_filters,
        "pruned_filters": pruned_filters,
        "params_after": _parameter_count(net),
    }
    if params.get("export_onnx", True):
        try:
            import onnx
            import onnxsim

            input_shape = _parse_input_shape(
                params.get("input_shape"), [1, 3, 224, 224]
            )
            dummy = torch.randn(*input_shape)
            onnx_path = out_dir / "model_pruned.onnx"
            torch.onnx.export(
                net,
                dummy,
                str(onnx_path),
                opset_version=int(params.get("opset", 12)),
                input_names=["input"],
                output_names=["output"],
                # Keep weights inline (single self-contained file).
                dynamo=False,
            )
            simplified, _ = onnxsim.simplify(str(onnx_path))
            onnx.save(simplified, str(onnx_path))
            result_metrics["onnx_exported"] = True
        except Exception as exc:  # ONNX export is best-effort
            result_metrics["onnx_exported"] = False
            result_metrics["onnx_error"] = str(exc)[:300]

    progress(95, "Done.")
    # Return the whole output directory so the zip download includes both the
    # pruned checkpoint (.pt) and the exported ONNX model.
    return out_dir, result_metrics


# ---------------------------------------------------------------------------
# export_gguf: HuggingFace LLM -> GGUF (via llama.cpp convert script)
# ---------------------------------------------------------------------------
# llama.cpp's convert_hf_to_gguf.py only ever *emits* these formats; the
# legacy/IK quants (q4_0, q4_K_M, iq4_xs, ...) listed in _GGUF_OUTTYPES are
# produced by the separate C++ ``llama-quantize`` tool, not by the converter.
_GGUF_CONVERTER_OUTTYPES = {"f32", "f16", "bf16", "q8_0", "tq1_0", "tq2_0"}

# Full GGUF format vocabulary (for validation/UI); anything not in
# _GGUF_CONVERTER_OUTTYPES needs the separate llama-quantize tool.
_GGUF_OUTTYPES = {
    "f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "q5_0", "q5_1",
    "q2_K", "q3_K", "q4_K", "q5_K", "q6_K",
    "q4_K_S", "q4_K_M", "q5_K_S", "q5_K_M", "iq1_s", "iq1_m", "iq2_xs", "iq2_s", "iq2_m", "iq3_xxs", "iq3_xs",
    "iq3_s", "iq4_xs", "iq4_nl", "iq5_xs", "iq5_xxs", "iq6_xs",
    "tq1_0", "tq2_0",
}


# llama.cpp conversion tooling mirrors: official GitHub first, then common
# CN-friendly proxies. ``git`` clones a shallow repo; ``tar`` downloads and
# extracts the master tarball (works when the git protocol is blocked).
_LLAMA_CPP_SOURCES: list[tuple[str, str]] = [
    ("git", "https://github.com/ggml-org/llama.cpp.git"),
    ("git", "https://ghfast.top/https://github.com/ggml-org/llama.cpp.git"),
    ("git", "https://gh-proxy.com/https://github.com/ggml-org/llama.cpp.git"),
    ("git", "https://gitclone.com/github.com/ggml-org/llama.cpp.git"),
    (
        "tar",
        "https://codeload.github.com/ggml-org/llama.cpp/tar.gz/refs/heads/master",
    ),
    (
        "tar",
        "https://ghfast.top/https://codeload.github.com/ggml-org/llama.cpp/"
        "tar.gz/refs/heads/master",
    ),
    (
        "tar",
        "https://gh-proxy.com/https://codeload.github.com/ggml-org/llama.cpp/"
        "tar.gz/refs/heads/master",
    ),
]


def _run_tool(args: list[str], timeout: int) -> Any:
    import subprocess

    return subprocess.run(
        args, capture_output=True, text=True, timeout=timeout
    )


# jsDelivr serves a read-only copy of every public GitHub repo; the CDN is
# reachable in networks where GitHub itself is blocked (mainland China etc.).
_JSDELIVR_HOSTS = (
    "https://cdn.jsdelivr.net",
    "https://fastly.jsdelivr.net",
    "https://gcore.jsdelivr.net",
)


def _fetch_llama_cpp_via_jsdelivr(cache: Path) -> bool:
    """Download llama.cpp conversion tooling through the jsDelivr CDN.

    The current llama.cpp converter is split across ``convert_hf_to_gguf.py``,
    a ``conversion/`` package and a vendored ``gguf-py`` package (~1.7 MB /
    118 files). The recursive file *list* comes from the GitHub API (small
    JSON, api.github.com is often reachable even when git/github.com are not);
    each required file is then downloaded from a jsDelivr host.
    """
    import httpx

    shutil.rmtree(cache, ignore_errors=True)
    try:
        tree = httpx.get(
            "https://api.github.com/repos/ggml-org/llama.cpp/git/trees/"
            "master?recursive=1",
            timeout=60,
            follow_redirects=True,
        ).json()
    except Exception:
        return False
    paths = sorted(
        t["path"]
        for t in tree.get("tree", [])
        if t["type"] == "blob"
        and (
            t["path"] == "convert_hf_to_gguf.py"
            or t["path"].startswith("conversion/")
            or t["path"].startswith("gguf-py/")
        )
    )
    if not paths:
        return False
    for p in paths:
        dest = cache / p
        dest.parent.mkdir(parents=True, exist_ok=True)
        ok = False
        for host in _JSDELIVR_HOSTS:
            try:
                resp = httpx.get(
                    f"{host}/gh/ggml-org/llama.cpp@master/{p}",
                    timeout=60,
                    follow_redirects=True,
                )
                if resp.status_code == 200:
                    dest.write_bytes(resp.content)
                    ok = True
                    break
            except Exception:
                continue
        if not ok:
            return False
    return (cache / "convert_hf_to_gguf.py").exists()


_MODELSCOPE_URL_PREFIXES = (
    "https://modelscope.cn/models/",
    "https://www.modelscope.cn/models/",
    "https://www.modelscope.com/models/",
    "modelscope.cn/models/",
    "www.modelscope.cn/models/",
)


def _is_modelscope_url(source: str) -> bool:
    return any(source.strip().startswith(p) for p in _MODELSCOPE_URL_PREFIXES)


def _modelscope_repo_id(source: str) -> str | None:
    """Extract a ModelScope repo id (``owner/name``) from a URL or bare id."""
    s = source.strip().rstrip("/")
    for prefix in _MODELSCOPE_URL_PREFIXES:
        if s.startswith(prefix):
            rest = s[len(prefix) :]
            parts = rest.split("/")
            return "/".join(parts[:2]) if len(parts) >= 2 else rest
    return None


def _download_modelscope_repo(
    repo_id: str, dest: Path, progress: ProgressCallback
) -> Path:
    """Download a full ModelScope model repo snapshot to ``dest``.

    Uses the public repo-files listing + per-file download endpoints, so no
    extra SDK is required. Large weight files are followed through the CDN
    redirects automatically.
    """
    import httpx

    base = f"https://modelscope.cn/api/v1/models/{repo_id}"
    dest.mkdir(parents=True, exist_ok=True)
    progress(20, f"Listing files of ModelScope repo {repo_id}...")
    try:
        listing = httpx.get(
            f"{base}/repo/files?Revision=master&Recursive=true",
            timeout=60,
            follow_redirects=True,
        )
        listing.raise_for_status()
        files = [
            f
            for f in listing.json().get("Data", {}).get("Files", [])
            if f.get("Type") == "blob" and f.get("Path")
        ]
    except Exception as exc:  # noqa: BLE001 - surface a friendly message
        raise ValueError(
            f"Could not reach ModelScope repo '{repo_id}': {exc}"
        ) from exc
    if not files:
        raise ValueError(
            f"ModelScope repo '{repo_id}' has no downloadable files"
        )
    total = sum(int(f.get("Size") or 0) for f in files)
    done = 0
    for idx, f in enumerate(files, start=1):
        rel = Path(str(f["Path"]))
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.stat().st_size == int(
            f.get("Size") or 0
        ):
            done += int(f.get("Size") or 0)
            continue
        progress(
            10 + int(80 * done / max(total, 1)),
            f"Downloading {rel.name} from ModelScope ({idx}/{len(files)})...",
        )
        with httpx.stream(
            "GET",
            f"{base}/repo?Revision=master&FilePath={f['Path']}",
            timeout=900,
            follow_redirects=True,
        ) as resp:
            resp.raise_for_status()
            with target.open("wb") as fh:
                for chunk in resp.iter_bytes(chunk_size=1024 * 1024):
                    fh.write(chunk)
        done += int(f.get("Size") or target.stat().st_size)
    progress(92, f"ModelScope download of {repo_id} finished.")
    return dest


def _fetch_llama_cpp_tools(progress: ProgressCallback) -> Path:
    """Locate (or download) llama.cpp conversion tooling.

    Priority: ``LLAMA_CPP_PATH`` env var -> shared on-disk cache (reused by all
    tasks on this machine) -> each mirror source with automatic retry.
    """
    import os
    import shutil
    import tarfile
    import time as _time

    local = os.environ.get("LLAMA_CPP_PATH")
    if local:
        path = Path(local)
        if (path / "convert_hf_to_gguf.py").exists():
            return path

    from app.jobs.download import download_file
    from app.jobs.storage import models_root

    tools_root = models_root() / "tools"
    tools_root.mkdir(parents=True, exist_ok=True)
    cache = tools_root / "llama.cpp"
    marker = cache / "convert_hf_to_gguf.py"
    if marker.exists():
        return cache

    # Simple lock so two jobs never download simultaneously.
    lock = tools_root / ".llama_cpp.lock"
    for _ in range(12):
        try:
            lock.touch(exist_ok=False)
            break
        except FileExistsError:
            _time.sleep(5)
    try:
        for kind, url in _LLAMA_CPP_SOURCES:
            # One attempt per source: mirrors fail over quickly instead of
            # hanging on a slow git clone (github TCP timeouts can be ~2 min).
            progress(15, f"Fetching llama.cpp conversion tooling via {url}...")
            try:
                if kind == "git":
                    shutil.rmtree(cache, ignore_errors=True)
                    proc = _run_tool(
                        [
                            "git",
                            "clone",
                            "--depth",
                            "1",
                            "--single-branch",
                            url,
                            str(cache),
                        ],
                        timeout=90,
                    )
                    if proc.returncode != 0:
                        raise RuntimeError(
                            (proc.stderr or proc.stdout or "")[-500:]
                        )
                else:
                    archive = tools_root / "llama.cpp.tar.gz"
                    archive.unlink(missing_ok=True)
                    download_file(
                        url,
                        archive,
                        max_bytes=400 * 1024 * 1024,
                    )
                    shutil.rmtree(cache, ignore_errors=True)
                    with tarfile.open(archive, "r:gz") as tf:
                        top_name = Path(tf.getnames()[0])
                        tf.extractall(tools_root, filter="data")
                    if not marker.exists():
                        top = tools_root / top_name
                        if top.exists():
                            shutil.move(str(top), str(cache))
                    archive.unlink(missing_ok=True)
                if marker.exists():
                    return cache
            except Exception:
                continue
        # git/tar mirrors exhausted -> jsDelivr CDN (works behind CN firewalls;
        # needs only api.github.com for the file list, file content comes from
        # cdn.jsdelivr.net / fastly.jsdelivr.net).
        progress(15, "Git mirrors unreachable; trying the jsDelivr CDN mirror...")
        if _fetch_llama_cpp_via_jsdelivr(cache) and marker.exists():
            return cache
        tried = "\n".join(
            f"  - [{kind}] {url}" for kind, url in _LLAMA_CPP_SOURCES
        )
        raise ValueError(
            "Could not fetch llama.cpp conversion tooling from any source "
            "(git mirrors, tarballs and the jsDelivr CDN all failed). Tried:\n"
            f"{tried}\n"
            "Tip: pre-fetch llama.cpp once and point the worker at it with "
            "the LLAMA_CPP_PATH environment variable."
        )
    finally:
        lock.unlink(missing_ok=True)


def export_gguf(
    model: Model,
    params: dict[str, Any],
    out_dir: Path,
    progress: ProgressCallback,
) -> tuple[Path, dict[str, Any]]:
    """Convert an LLM to GGUF for llama.cpp / llama-cpp-python.

    The model may be given as a HuggingFace repo id (``source_url``), a
    ModelScope model page URL or repo id (``source_url``), or a local HF-format
    directory (``file_path``). Model downloads fail over automatically:
    Hugging Face -> hf-mirror.com -> ModelScope, so CN networks work without
    extra configuration.

    llama.cpp's conversion tooling lives inside its own repository; it is
    fetched into a shared cache (git, tarball proxies, or the jsDelivr CDN),
    then ``convert_hf_to_gguf.py`` runs against the resolved model source.
    """
    import os
    import subprocess
    import sys

    source_url = model.source_url
    local_path = Path(model.file_path) if model.file_path else None
    if not source_url and not (local_path and local_path.is_dir()):
        raise ValueError(
            "GGUF export needs a HuggingFace/ModelScope repo id (source URL) "
            "or a local HF-format model directory"
        )

    outtype = str(params.get("outtype", "f16")).lower()
    if outtype not in _GGUF_OUTTYPES:
        raise ValueError(f"Unsupported GGUF outtype: {outtype}")
    if outtype not in _GGUF_CONVERTER_OUTTYPES:
        raise ValueError(
            f"GGUF outtype '{outtype}' cannot be produced by llama.cpp's "
            "convert_hf_to_gguf.py (it emits only f32 / f16 / bf16 / q8_0 / "
            "tq1_0 / tq2_0). Legacy & IK quantized formats (q4_0, q4_K_M, "
            "iq4_xs, ...) are created by the separate 'llama-quantize' C++ "
            "tool compiled from the llama.cpp sources."
        )

    # Resolve the model to a local directory whenever possible so that we (not
    # the converter) control which hub the weights come from.
    hf_id: str | None = None
    ms_id: str | None = None
    model_dir: Path | None = None
    if local_path and local_path.is_dir():
        model_dir = local_path
    elif source_url and _is_modelscope_url(source_url):
        ms_id = _modelscope_repo_id(source_url) or source_url
    else:
        hf_id = source_url or None
    if ms_id:
        model_dir = _download_modelscope_repo(ms_id, out_dir / "model_src", progress)

    repo = _fetch_llama_cpp_tools(progress)
    convert_script = repo / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        raise ValueError("llama.cpp checkout missing convert_hf_to_gguf.py")

    out_name = out_dir / "model.gguf"
    env = dict(os.environ)
    env["PYTHONPATH"] = (
        str(repo)
        + os.pathsep
        + str(repo / "gguf-py")
        + os.pathsep
        + env.get("PYTHONPATH", "")
    )
    timeout_s = max(int(params.get("timeout_seconds", 12 * 60 * 60)), 60)

    def _convert(current_env: dict[str, str], model_arg: str) -> Any:
        cmd = [
            sys.executable,
            str(convert_script),
            model_arg,
            "--outfile",
            str(out_name),
            "--outtype",
            outtype,
        ]
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s, env=current_env
        )

    def _network_failure(out: str) -> bool:
        low = (out or "").lower()
        return any(
            k in low
            for k in (
                "huggingface.co",
                "connection",
                "ssl",
                "resolve",
                "maxretry",
                "cannot connect",
            )
        )

    if model_dir is not None:
        model_arg = str(model_dir)
    elif hf_id:
        model_arg = hf_id
    elif ms_id:
        model_arg = ms_id
    else:  # pragma: no cover - guarded above
        raise ValueError("no GGUF model source")

    hub_label = (
        "ModelScope" if (model_dir and not hf_id) else ("local dir" if model_dir else "HuggingFace")
    )
    progress(
        40,
        f"Running GGUF conversion ({outtype}). Source: {hub_label}. This can "
        "take a long time...",
    )
    result = _convert(env, model_arg)
    tail = result.stderr or result.stdout or ""
    # Failover 1: huggingface.co unreachable -> retry via hf-mirror endpoint.
    if (
        result.returncode != 0
        and _network_failure(tail)
        and hf_id
        and model_dir is None
    ):
        progress(50, "huggingface.co unreachable; retrying via hf-mirror.com...")
        mirror_env = dict(env)
        mirror_env["HF_ENDPOINT"] = "https://hf-mirror.com"
        result = _convert(mirror_env, model_arg)
    # Failover 2: still no output -> mirror the same repo on ModelScope.
    if (
        (result.returncode != 0 or not out_name.exists())
        and hf_id
        and model_dir is None
    ):
        tail = result.stderr or result.stdout or ""
        if _network_failure(tail):
            progress(
                55,
                f"Hugging Face unreachable; trying the ModelScope mirror of "
                f"{hf_id}...",
            )
            try:
                model_dir = _download_modelscope_repo(
                    hf_id, out_dir / "model_src", progress
                )
                result = _convert(env, str(model_dir))
            except ValueError as exc:
                raise ValueError(
                    "GGUF conversion failed: huggingface.co / hf-mirror.com "
                    f"unreachable and ModelScope has no mirror of '{hf_id}' "
                    f"({exc}). Provide a ModelScope URL as the source URL "
                    "instead."
                ) from exc

    if result.returncode != 0 or not out_name.exists():
        raise ValueError(
            f"GGUF conversion failed: {(result.stderr or result.stdout or '')[-800:]}"
        )
    progress(95, "Done.")
    meta: dict[str, Any] = {"outtype": outtype, "tool": "llama.cpp"}
    if hf_id and model_dir is not None:
        meta["hub"] = "huggingface+modelscope-fallback"
    elif hf_id:
        meta["hub"] = "huggingface"
    elif ms_id:
        meta["hub"] = "modelscope"
    else:
        meta["hub"] = "local"
    return out_name, meta


# ---------------------------------------------------------------------------
# sparsify: unstructured magnitude pruning (weight sparsity, no extra deps)
# ---------------------------------------------------------------------------
def sparsify(
    model: Model,
    params: dict[str, Any],
    out_dir: Path,
    progress: ProgressCallback,
) -> tuple[Path, dict[str, Any]]:
    """Zero out the smallest weights (unstructured sparsity) of a PyTorch CNN
    and export the pruned model to ONNX.

    ``rate`` (0..1) controls the fraction of weights zeroed per layer.
    """
    import torch

    if model.file_path is None:
        raise ValueError("Model has no local file")

    rate = float(params.get("rate", 0.3))
    if not 0 <= rate < 1:
        raise ValueError("sparsify rate must be in [0, 1)")

    progress(10, "Loading PyTorch model...")
    net = _load_torch_module(Path(model.file_path))
    net.eval()

    total_params = 0
    zeroed = 0
    progress(40, "Pruning smallest weights...")
    for _name, param in net.named_parameters():
        if param.dim() < 2 or param.numel() == 0:
            continue
        total_params += param.numel()
        threshold = torch.quantile(param.abs().flatten(), rate)
        mask = param.abs() > threshold
        with torch.no_grad():
            param.mul_(mask)
        zeroed += int((~mask).sum().item())
    sparsity = (zeroed / total_params) if total_params else 0.0

    input_shape = _parse_input_shape(params.get("input_shape"), [1, 3, 224, 224])
    dummy = torch.randn(*input_shape)
    out = out_dir / "model_sparse.onnx"
    progress(70, "Exporting pruned model to ONNX...")
    torch.onnx.export(
        net,
        dummy,
        str(out),
        opset_version=int(params.get("opset", 12)),
        input_names=["input"],
        output_names=["output"],
        dynamo=False,
    )
    progress(95, "Done.")
    return out, {
        "sparsity_pct": round(sparsity * 100, 2),
        "rate": rate,
        "pruned_weights": zeroed,
    }


def _write_static_quantized_onnx(
    src: Path,
    images: list[Any],
    out_dir: Path,
    progress: ProgressCallback,
) -> Path:
    """Static INT8 (QDQ) quantize an ONNX file with the given calibration
    images. Falls back to QOperator format if the QDQ graph fails validation."""
    import numpy as np
    import onnxruntime
    from onnxruntime.quantization import (
        CalibrationDataReader,
        QuantFormat,
        QuantType,
        quantize_static,
    )
    from onnxruntime.quantization.shape_inference import quant_pre_process

    input_name = _onnx_input_name(src)

    class _Reader(CalibrationDataReader):
        def __init__(self, data: list[np.ndarray], name: str) -> None:
            self._it = iter(data)
            self._name = name

        def get_next(self) -> dict[str, np.ndarray] | None:
            arr = next(self._it, None)
            return {self._name: arr} if arr is not None else None

    preprocessed = out_dir / "preprocessed.onnx"
    quant_pre_process(str(src), str(preprocessed))
    out = out_dir / "model_quantized.onnx"

    def _quantize(fmt: QuantFormat) -> None:
        quantize_static(
            str(preprocessed),
            str(out),
            calibration_data_reader=_Reader(images, input_name),
            quant_format=fmt,
            per_channel=True,
            weight_type=QuantType.QInt8,
            reduce_range=False,
        )

    _quantize(QuantFormat.QDQ)
    try:
        onnxruntime.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    except Exception:
        progress(72, "QDQ validation failed, retrying with QOperator...")
        out.unlink(missing_ok=True)
        _quantize(QuantFormat.QOperator)
    preprocessed.unlink(missing_ok=True)
    return out


def _onnx_input_name(src: Path) -> str:
    import onnx

    return onnx.load(str(src)).graph.input[0].name


# ---------------------------------------------------------------------------
# qat: quantization-aware fine-tuning (PyTorch FX) -> quantized ONNX
# ---------------------------------------------------------------------------
def qat(
    model: Model,
    params: dict[str, Any],
    out_dir: Path,
    progress: ProgressCallback,
) -> tuple[Path, dict[str, Any]]:
    """Quantization-aware fine-tuning on user data, then export quantized ONNX.

    Uses torch's FX QAT flow (``prepare_qat_fx``/``convert_fx``), which inserts
    fake quantization during training so weights adapt to INT8, then exports a
    QDQ-style quantized ONNX that onnxruntime can execute.

    Requires a PyTorch CNN checkpoint with an importable architecture
    (``arch.py``) and a per-class image dataset (``dataset_id``/``data_url``).
    """
    import torch
    import torch.nn as nn

    if model.file_path is None:
        raise ValueError("Model has no local file")

    image_size = int(params.get("image_size", 224))
    input_shape = _parse_input_shape(
        params.get("input_shape"), [1, 3, image_size, image_size]
    )
    epochs = max(int(params.get("epochs", 3)), 1)
    lr = float(params.get("lr", 1e-3))
    batch_size = max(int(params.get("batch_size", 8)), 1)
    # Early stopping + LR scheduling controls.
    patience = max(int(params.get("patience", 2)), 0)
    lr_schedule = str(params.get("lr_schedule", "plateau")).lower()
    if lr_schedule not in ("plateau", "step", "none"):
        raise ValueError("lr_schedule must be one of: plateau, step, none")

    dataset_id = params.get("dataset_id")
    data_url = params.get("data_url")
    data_dir = params.get("data_dir")
    if not (dataset_id or data_url or data_dir):
        raise ValueError(
            "qat requires a training dataset (dataset_id or data_url)"
        )
    data_cfg: dict[str, Any] = {
        "type": "classification",
        "dataset_id": dataset_id,
        "data_url": data_url,
        "data_dir": data_dir,
    }

    progress(10, "Loading model and preparing data...")
    net = _load_torch_module(Path(model.file_path))
    net.train()

    height = input_shape[2] if len(input_shape) == 4 else image_size
    width = input_shape[3] if len(input_shape) == 4 else image_size
    data_path = _resolve_eval_data(data_cfg, out_dir / "qat_data")
    class_dirs = sorted(p for p in data_path.iterdir() if p.is_dir())
    if not class_dirs:
        raise ValueError("QAT data must contain per-class image folders")

    images: list[torch.Tensor] = []
    labels: list[int] = []
    for label, class_dir in enumerate(class_dirs):
        for img_path in sorted(class_dir.iterdir()):
            if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                arr = _preprocess_image(img_path, height, width)[0]
                images.append(torch.from_numpy(arr.copy()))
                labels.append(label)
    if not images:
        raise ValueError("No images found in QAT data")
    X = torch.stack(images)
    y = torch.tensor(labels, dtype=torch.long)

    def _accuracy(module: Any) -> float:
        module.eval()
        correct = 0
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                out = module(X[i : i + batch_size])
                if isinstance(out, (tuple, list)):
                    out = out[0]
                correct += int(
                    (out.argmax(dim=1) == y[i : i + batch_size]).sum().item()
                )
        return round(correct / len(y), 4)

    baseline_accuracy = _accuracy(net)
    loss_fn = nn.CrossEntropyLoss()
    train_info: dict[str, Any] = {
        "epochs_completed": 0,
        "best_val_loss": None,
        "final_lr": lr,
    }

    # Hold out a small validation split for early stopping / plateau signals.
    val_idx = None
    if len(y) >= 16:
        perm = torch.randperm(len(y))
        n_val = max(1, int(len(y) * 0.2))
        val_idx = perm[n_val:]

    def _train(module: Any, epochs_run: int) -> Any:
        module.train()
        optimizer = torch.optim.SGD(module.parameters(), lr=lr, momentum=0.9)
        if lr_schedule == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", patience=max(patience, 1), factor=0.5
            )
        elif lr_schedule == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=max(epochs_run // 2, 1), gamma=0.5
            )
        else:
            scheduler = None

        best_state: dict[str, Any] | None = None
        best_loss = float("inf")
        epochs_no_improve = 0
        for epoch in range(epochs_run):
            permutation = torch.randperm(len(X))
            running_loss = 0.0
            steps = 0
            for i in range(0, len(permutation), batch_size):
                idx = permutation[i : i + batch_size]
                optimizer.zero_grad()
                output = module(X[idx])
                if isinstance(output, (tuple, list)):
                    output = output[0]
                loss = loss_fn(output, y[idx])
                loss.backward()
                optimizer.step()
                running_loss += float(loss)
                steps += 1
            epoch_loss = running_loss / max(steps, 1)

            # Monitor loss on the val split (or train loss when too few samples).
            monitor_loss = epoch_loss
            if val_idx is not None:
                with torch.no_grad():
                    out = module(X[val_idx])
                    if isinstance(out, (tuple, list)):
                        out = out[0]
                    monitor_loss = float(loss_fn(out, y[val_idx]))

            if patience > 0 and monitor_loss < best_loss - 1e-4:
                best_loss = monitor_loss
                best_state = {
                    k: v.detach().clone()
                    for k, v in module.state_dict().items()
                }
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(monitor_loss)
                else:
                    scheduler.step()

            progress(
                45 + int(30 * (epoch + 1) / epochs_run),
                f"Epoch {epoch + 1}/{epochs_run} loss={epoch_loss:.4f} "
                f"monitor={monitor_loss:.4f} lr={optimizer.param_groups[0]['lr']:.2e}",
            )
            if patience > 0 and epochs_no_improve >= patience:
                if best_state is not None:
                    module.load_state_dict(best_state)
                break

        train_info["epochs_completed"] = epoch + 1
        train_info["best_val_loss"] = round(float(best_loss), 4)
        train_info["final_lr"] = optimizer.param_groups[0]["lr"]
        return module.eval()

    # ---- Primary path: FX QAT (fake-quantize aware training) ---------------
    qat_mode = "fx_qat"
    progress(40, "Preparing quantization-aware training graph...")
    try:
        from torch.ao.quantization import get_default_qat_qconfig_mapping
        from torch.ao.quantization.quantize_fx import convert_fx, prepare_qat_fx

        qconfig = get_default_qat_qconfig_mapping("fbgemm")
        prepared = prepare_qat_fx(net.train(), qconfig, example_inputs=(X[:1],))
        trained = _train(prepared, epochs)
        converted = convert_fx(trained.eval())
        converted.eval()
        out = out_dir / "model_qat.onnx"
        torch.onnx.export(
            converted,
            torch.randn(*input_shape),
            str(out),
            opset_version=int(params.get("opset", 13)),
            input_names=["input"],
            output_names=["output"],
            dynamo=False,
        )
        qat_accuracy = _accuracy(converted)
        progress(95, "Done.")
        return out, {
            "quant_type": "qat_int8",
            "qat_mode": qat_mode,
            "epochs": epochs,
            "epochs_completed": train_info["epochs_completed"],
            "best_val_loss": train_info["best_val_loss"],
            "final_lr": train_info["final_lr"],
            "lr_schedule": lr_schedule,
            "samples": len(y),
            "classes": len(class_dirs),
            "baseline_accuracy": baseline_accuracy,
            "qat_accuracy": qat_accuracy,
        }
    except Exception as fx_exc:
        # ---- Fallback: model not FX-traceable -> fine-tune + static INT8 -----
        qat_mode = "finetune_ptq"
        progress(
            40,
            f"FX QAT unavailable ({fx_exc}); falling back to fine-tune + PTQ...",
        )
        trained = _train(net, epochs)
        tuned_accuracy = _accuracy(trained)

        float_onnx = out_dir / "model_finetuned.onnx"
        progress(80, "Exporting fine-tuned model to ONNX...")
        torch.onnx.export(
            trained,
            torch.randn(*input_shape),
            str(float_onnx),
            opset_version=int(params.get("opset", 12)),
            input_names=["input"],
            output_names=["output"],
            dynamo=False,
        )
        progress(85, "Static INT8 quantization from fine-tuned model...")
        calib_arrays: list[Any] = []
        for img in images:
            calib_arrays.append(img.numpy())
        out = _write_static_quantized_onnx(
            float_onnx, calib_arrays, out_dir, progress
        )
        progress(95, "Done.")
        return out, {
            "quant_type": "qat_int8",
            "qat_mode": qat_mode,
            "epochs": epochs,
            "epochs_completed": train_info["epochs_completed"],
            "best_val_loss": train_info["best_val_loss"],
            "final_lr": train_info["final_lr"],
            "lr_schedule": lr_schedule,
            "samples": len(y),
            "classes": len(class_dirs),
            "baseline_accuracy": baseline_accuracy,
            "qat_accuracy": tuned_accuracy,
        }


# ---------------------------------------------------------------------------
# llm_quantize: LLM INT4/INT8 via optimum-intel (OpenVINO IR)
# ---------------------------------------------------------------------------
def llm_quantize(
    model: Model,
    params: dict[str, Any],
    out_dir: Path,
    progress: ProgressCallback,
) -> tuple[Path, dict[str, Any]]:
    bits = int(params.get("bits", 4))
    if bits not in (4, 8):
        raise ValueError("llm_quantize bits must be 4 or 8")

    model_id = model.source_url or model.file_path
    if not model_id:
        raise ValueError(
            "LLM quantization needs a HuggingFace repo id (set as source URL) "
            "or a local model directory"
        )
    if model.source_url and _is_modelscope_url(model.source_url):
        raise ValueError(
            "llm_quantize (OpenVINO) downloads weights through HuggingFace and "
            "needs an HF repo id as the source URL. ModelScope sources are "
            "supported by the GGUF export (export_gguf) method instead."
        )

    from optimum.intel import OVModelForCausalLM, OVWeightQuantizationConfig

    progress(10, f"Loading and exporting LLM as INT{bits} (can take a while)...")
    quantization_config = OVWeightQuantizationConfig(bits=bits)
    ov_model = OVModelForCausalLM.from_pretrained(
        model_id,
        export=True,
        quantization_config=quantization_config,
        device="cpu",
        compile=False,
        trust_remote_code=True,
    )
    progress(80, "Saving OpenVINO IR...")
    ov_model.save_pretrained(str(out_dir))
    progress(95, "Done.")
    return out_dir, {"bits": bits, "llm": True, "precision": f"int{bits}"}


# ---------------------------------------------------------------------------
# Benchmark (CPU latency / throughput of original vs compressed)
# ---------------------------------------------------------------------------
def _make_runner(path: Path) -> Any:
    """Return a function that runs a model on a numpy input array."""
    if path.is_dir() and (path / "model.xml").exists():
        import openvino as ov

        compiled = ov.Core().compile_model(str(path / "model.xml"), "CPU")
        input_name = next(iter(compiled.inputs)).any_name
        output_key = next(iter(compiled.outputs))
        return lambda arr: compiled({input_name: arr})[output_key]

    if path.suffix.lower() == ".onnx":
        import onnxruntime as ort

        session = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )
        input_name = session.get_inputs()[0].name
        return lambda arr: session.run(None, {input_name: arr})[0]

    if path.suffix.lower() in {".pt", ".pth"}:
        import torch

        net = _load_torch_module(path)
        net.eval()

        def _run(arr: Any) -> Any:
            with torch.no_grad():
                output = net(torch.from_numpy(arr))
                if isinstance(output, (tuple, list)):
                    output = output[0]
                return output.numpy()

        return _run

    raise ValueError(f"Cannot benchmark artifact: {path}")


def run_benchmark(
    model: Model,
    artifact: Path,
    cfg: dict[str, Any],
    progress: ProgressCallback,
) -> dict[str, Any]:
    """Measure CPU inference latency/throughput of the original and compressed
    models. Returns a dict keyed by ``original`` / ``compressed``."""
    import time

    import numpy as np

    progress(80, "Running performance benchmark (CPU)...")
    iterations = max(int(cfg.get("iterations", 20)), 1)
    warmup = max(int(cfg.get("warmup", 3)), 0)
    input_shape = _parse_input_shape(cfg.get("input_shape"), [1, 3, 224, 224])

    def _measure(runner: Any) -> dict[str, float]:
        import threading

        import psutil

        arr = np.random.randn(*input_shape).astype(np.float32)
        process = psutil.Process()
        baseline_mb = process.memory_info().rss / 1e6
        peak_mb: dict[str, float] = {"value": process.memory_info().rss / 1e6}
        stop = threading.Event()

        def _sample_memory() -> None:
            while not stop.is_set():
                try:
                    rss = process.memory_info().rss / 1e6
                    if rss > peak_mb["value"]:
                        peak_mb["value"] = rss
                except Exception:
                    pass
                time.sleep(0.002)

        for _ in range(warmup):
            runner(arr)
        latencies: list[float] = []
        sampler = threading.Thread(target=_sample_memory, daemon=True)
        sampler.start()
        for _ in range(iterations):
            start = time.perf_counter()
            runner(arr)
            latencies.append((time.perf_counter() - start) * 1000.0)
        stop.set()
        sampler.join()

        latencies.sort()
        mean = sum(latencies) / len(latencies)
        variance = sum((x - mean) ** 2 for x in latencies) / len(latencies)

        def _percentile(q: float) -> float:
            return latencies[max(int(len(latencies) * q) - 1, 0)]

        return {
            "latency_ms_mean": round(mean, 3),
            "latency_ms_p50": round(latencies[len(latencies) // 2], 3),
            "latency_ms_p95": round(_percentile(0.95), 3),
            "latency_ms_p99": round(_percentile(0.99), 3),
            "latency_ms_min": round(latencies[0], 3),
            "latency_ms_max": round(latencies[-1], 3),
            "latency_ms_std": round(variance**0.5, 3),
            "throughput_ips": round(1000.0 / mean, 2),
            "peak_memory_mb": round(peak_mb["value"], 2),
            "memory_delta_mb": round(peak_mb["value"] - baseline_mb, 2),
            "iterations": iterations,
        }

    result: dict[str, Any] = {}
    if model.file_path:
        try:
            result["original"] = _measure(_make_runner(Path(model.file_path)))
        except Exception as exc:
            result["original_error"] = f"{type(exc).__name__}: {exc}"[:300]
    try:
        result["compressed"] = _measure(_make_runner(artifact))
    except Exception as exc:
        result["compressed_error"] = f"{type(exc).__name__}: {exc}"[:300]
    if "original" in result and "compressed" in result:
        orig_mean = result["original"]["latency_ms_mean"]
        comp_mean = result["compressed"]["latency_ms_mean"]
        if comp_mean > 0:
            result["speedup_x"] = round(orig_mean / comp_mean, 3)
    return result


# ---------------------------------------------------------------------------
# Evaluation (optional accuracy comparison)
# ---------------------------------------------------------------------------
def run_evaluation(
    model: Model,
    artifact: Path,
    workspace: Path,
    eval_cfg: dict[str, Any],
    progress: ProgressCallback,
) -> dict[str, Any]:
    kind = eval_cfg.get("type", "classification")
    if kind == "classification":
        return _eval_classification(model, artifact, workspace, eval_cfg, progress)
    if kind == "yolo":
        return _eval_yolo(artifact, workspace, eval_cfg, progress)
    if kind == "perplexity":
        return _eval_perplexity(artifact, workspace, eval_cfg, progress)
    raise ValueError(f"Unknown eval type: {kind}")


def _dataset_dir_from_id(dataset_id: str) -> Path:
    """Resolve a platform-managed dataset id to its local extracted path."""
    from sqlmodel import Session

    from app.core.db import engine
    from app.models import Dataset

    with Session(engine) as session:
        dataset = session.get(Dataset, dataset_id)
        if not dataset or not dataset.file_path:
            raise ValueError(f"Dataset not found: {dataset_id}")
        path = Path(dataset.file_path)
    if not path.exists():
        raise ValueError(f"Dataset files missing: {dataset_id}")
    return path


def _resolve_eval_data(eval_cfg: dict[str, Any], workspace: Path) -> Path:
    """Return a local Path with the evaluation dataset.

    Sources, in priority order: ``dataset_id`` (platform-managed), ``data_url``
    (downloaded archive), ``data_dir`` (local path).
    """
    from app.jobs.download import download_and_extract_archive

    dataset_id = eval_cfg.get("dataset_id")
    data_url = eval_cfg.get("data_url")
    data_dir = eval_cfg.get("data_dir")
    if dataset_id:
        return _dataset_dir_from_id(str(dataset_id))
    if data_url:
        dest = workspace / "eval_data"
        marker = dest / ".ready"
        if not marker.exists():
            download_and_extract_archive(data_url, dest)
            marker.touch()
        return dest
    if data_dir:
        path = Path(data_dir)
        if not path.exists():
            raise ValueError(f"eval data_dir not found: {data_dir}")
        return path
    raise ValueError("eval requires data_dir, data_url, or dataset_id")


def _eval_classification(
    model: Model,
    artifact: Path,
    workspace: Path,
    eval_cfg: dict[str, Any],
    progress: ProgressCallback,
) -> dict[str, Any]:
    progress(70, "Running classification evaluation...")
    data_dir = _resolve_eval_data(eval_cfg, workspace)
    image_size = int(eval_cfg.get("image_size", 224))

    class_dirs = sorted(p for p in data_dir.iterdir() if p.is_dir())
    if not class_dirs:
        raise ValueError("Classification eval data must contain per-class folders")
    samples: list[tuple[Path, int]] = []
    for label, class_dir in enumerate(class_dirs):
        for img_path in sorted(class_dir.iterdir()):
            if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                samples.append((img_path, label))
    if not samples:
        raise ValueError("No images found in eval data")

    def accuracy(predict: Any) -> float:
        correct = sum(
            int(predict(img_path).argmax()) == label
            for img_path, label in samples
        )
        return round(correct / len(samples), 4)

    artifact_predictor = _make_classifier(artifact, image_size)
    result: dict[str, Any] = {"samples": len(samples)}
    result["accuracy"] = accuracy(artifact_predictor)

    if model.file_path:
        try:
            baseline = _make_classifier(Path(model.file_path), image_size)
            result["baseline_accuracy"] = accuracy(baseline)
            result["accuracy_delta"] = round(
                result["accuracy"] - result["baseline_accuracy"], 4
            )
        except Exception as exc:
            result["baseline_error"] = f"{type(exc).__name__}: {exc}"[:300]
    return result


def _preprocess_image(path: Path, height: int, width: int) -> Any:
    """Load + resize + ImageNet-normalize an image into an NCHW float32 batch."""
    import numpy as np
    from PIL import Image

    img = Image.open(path).convert("RGB").resize((width, height))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    return np.transpose(arr, (2, 0, 1))[None]


def _make_classifier(artifact: Path, image_size: int) -> Any:
    """Return a ``predict(image_path) -> np.ndarray`` function for an artifact."""
    import numpy as np

    def preprocess(path: Path) -> np.ndarray:
        return _preprocess_image(path, image_size, image_size)

    if artifact.is_dir() and (artifact / "model.xml").exists():
        import openvino as ov

        compiled = ov.Core().compile_model(str(artifact / "model.xml"), "CPU")
        input_name = next(iter(compiled.inputs)).any_name
        output_key = next(iter(compiled.outputs))

        def predict_ov(path: Path) -> np.ndarray:
            return compiled({input_name: preprocess(path)})[output_key]

        return predict_ov

    if artifact.suffix.lower() == ".onnx":
        import onnxruntime as ort

        session = ort.InferenceSession(
            str(artifact), providers=["CPUExecutionProvider"]
        )
        input_name = session.get_inputs()[0].name

        def predict_ort(path: Path) -> np.ndarray:
            return session.run(None, {input_name: preprocess(path)})[0]

        return predict_ort

    if artifact.suffix.lower() in {".pt", ".pth"}:
        import torch

        net = _load_torch_module(artifact)
        net.eval()

        def predict_torch(path: Path) -> np.ndarray:
            with torch.no_grad():
                output = net(torch.from_numpy(preprocess(path)))
                if isinstance(output, (tuple, list)):
                    output = output[0]
                return output.numpy()

        return predict_torch

    raise ValueError(f"Cannot build a classifier for artifact: {artifact}")


def _eval_yolo(
    artifact: Path,
    workspace: Path,
    eval_cfg: dict[str, Any],
    progress: ProgressCallback,
) -> dict[str, Any]:
    from ultralytics import YOLO

    progress(70, "Running YOLO evaluation...")
    data_dir = _resolve_eval_data(eval_cfg, workspace)
    yaml_path = next(
        (p for p in (data_dir / "data.yaml", data_dir / "data.yml") if p.exists()),
        None,
    )
    if yaml_path is None:
        raise ValueError("YOLO eval data must contain a data.yaml file")

    model_path = artifact
    if artifact.is_dir():
        candidates = [
            artifact / "best_openvino_model" / "best_openvino_model.xml",
            artifact / "model.xml",
        ]
        model_path = next((p for p in candidates if p.exists()), artifact)

    results = YOLO(str(model_path)).val(
        data=str(yaml_path), device="cpu", verbose=False
    )
    metrics = getattr(results, "results_dict", {}) or {}
    return {
        "map50": round(float(metrics.get("metrics/mAP50(B)", 0.0)), 4),
        "map50_95": round(float(metrics.get("metrics/mAP50-95(B)", 0.0)), 4),
        "precision": round(float(metrics.get("metrics/precision(B)", 0.0)), 4),
        "recall": round(float(metrics.get("metrics/recall(B)", 0.0)), 4),
    }


def _eval_perplexity(
    artifact: Path,
    workspace: Path,
    eval_cfg: dict[str, Any],
    progress: ProgressCallback,
) -> dict[str, Any]:
    import math

    import torch
    from optimum.intel import OVModelForCausalLM
    from transformers import AutoTokenizer

    progress(70, "Running perplexity evaluation...")
    text_url = eval_cfg.get("text_url")
    text_file = eval_cfg.get("text_file")
    if text_url:
        from app.jobs.download import download_file

        dest = workspace / "eval_text.txt"
        download_file(text_url, dest)
        text = dest.read_text(encoding="utf-8")
    elif text_file:
        text = Path(text_file).read_text(encoding="utf-8")
    else:
        raise ValueError("perplexity eval requires text_url or text_file")

    model_dir = artifact if artifact.is_dir() else Path(artifact)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    ov_model = OVModelForCausalLM.from_pretrained(
        str(model_dir), device="cpu", compile=False
    )
    ov_model.eval()

    max_length = int(eval_cfg.get("max_length", 512))
    encodings = tokenizer(text, return_tensors="pt")
    seq_len = encodings.input_ids.shape[1]
    nlls: list[float] = []
    stride = max_length
    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        input_ids = encodings.input_ids[:, begin:end]
        if input_ids.shape[1] < 2:
            continue
        with torch.no_grad():
            outputs = ov_model(input_ids=input_ids, labels=input_ids)
        nlls.append(float(outputs.loss.item()))
    if not nlls:
        raise ValueError("Text too short for perplexity evaluation")
    perplexity = math.exp(sum(nlls) / len(nlls))
    return {"perplexity": round(perplexity, 4), "chunks": len(nlls)}


