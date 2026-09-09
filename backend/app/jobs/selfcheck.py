"""Platform self-checks: one-click experiments that exercise the real job code
paths (distill/FitNets, etc.) without uploaded models, datasets or a DB row.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

_ARCH_SRC = '''import torch.nn as nn


class Net(nn.Module):
    def __init__(self, num_classes=2, width_mult=1.0):
        super().__init__()
        w = int(8 * width_mult)
        self.features = nn.Sequential(
            nn.Conv2d(3, w, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(w, w * 2, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(w * 2, num_classes)

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))
'''


class Net(nn.Module):
    """Identical to arch.Net so ``torch.save(model)`` pickling resolves."""

    def __init__(self, num_classes=2, width_mult=1.0):
        super().__init__()
        w = int(8 * width_mult)
        self.features = nn.Sequential(
            nn.Conv2d(3, w, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(w, w * 2, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(w * 2, num_classes)

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


def run_fitnets_smoke() -> dict[str, object]:
    """End-to-end FitNets distillation smoke on an auto-generated micro task."""
    from app.jobs.compression import distill

    root = Path(tempfile.mkdtemp(prefix="selfcheck_fitnets_"))
    try:
        (root / "arch.py").write_text(_ARCH_SRC, encoding="utf-8")
        torch.save(Net(num_classes=2, width_mult=1.0).eval(), root / "teacher.pt")

        data = root / "data"
        rng = np.random.default_rng(7)
        for label in ("cat", "dog"):
            (data / label).mkdir(parents=True)
            for i in range(6):
                arr = rng.integers(0, 256, size=(24, 24, 3), dtype=np.uint8)
                Image.fromarray(arr).save(data / label / f"{i}.png")

        params = {
            "method": "features",
            "width_mult": 0.5,
            "alpha": 0.7,
            "epochs": 1,
            "lr": 1e-3,
            "batch_size": 4,
            "image_size": 24,
            "input_shape": "1,3,24,24",
            "data_dir": str(data),
        }
        out = root / "out"

        def _progress(_pct: float, _msg: str | None = None) -> None:  # noqa: ARG001
            return None

        _, result = distill(
            type("M", (), {"file_path": str(root / "teacher.pt")})(),
            params,
            out,
            _progress,
        )
        return {
            "ok": True,
            "check": "fitnets_distill",
            "method": result.get("method"),
            "width_mult": result.get("width_mult"),
            "epochs_completed": result.get("epochs_completed"),
            "last_distill_loss": result.get("last_distill_loss"),
            "teacher_acc": result.get("teacher_acc"),
            "student_acc_before": result.get("student_acc_before"),
            "onnx_exported": result.get("onnx_exported"),
        }
    except Exception as exc:  # pragma: no cover - reported to the UI
        return {"ok": False, "check": "fitnets_distill", "error": f"{type(exc).__name__}: {exc}"}
    finally:
        shutil.rmtree(root, ignore_errors=True)



def _prep() -> tuple[Path, str, Path]:
    root = Path(tempfile.mkdtemp(prefix="selfcheck_"))
    (root / "arch.py").write_text(_ARCH_SRC, encoding="utf-8")
    teacher_pt = str(root / "teacher.pt")
    torch.save(Net(num_classes=2, width_mult=1.0).eval(), teacher_pt)
    data = root / "data"
    rng = np.random.default_rng(7)
    for label in ("cat", "dog"):
        (data / label).mkdir(parents=True)
        for i in range(6):
            arr = rng.integers(0, 256, size=(24, 24, 3), dtype=np.uint8)
            Image.fromarray(arr).save(data / label / f"{i}.png")
    return root, teacher_pt, data


def _model_stub(teacher_pt: str) -> object:
    return type("M", (), {"file_path": teacher_pt})()


def run_distill_logits_smoke() -> dict[str, object]:
    """Hinton logits KD smoke (teacher + auto dataset)."""
    from app.jobs.compression import distill

    root, teacher_pt, data = _prep()
    try:
        _, result = distill(
            _model_stub(teacher_pt),
            {
                "method": "logits",
                "width_mult": 0.5,
                "alpha": 0.7,
                "temperature": 4.0,
                "epochs": 1,
                "lr": 1e-3,
                "batch_size": 4,
                "image_size": 24,
                "input_shape": "1,3,24,24",
                "data_dir": str(data),
            },
            root / "out",
            lambda *_: None,
        )
        return {
            "ok": True,
            "check": "distill_logits",
            "method": result.get("method"),
            "epochs_completed": result.get("epochs_completed"),
            "last_distill_loss": result.get("last_distill_loss"),
            "onnx_exported": result.get("onnx_exported"),
        }
    except Exception as exc:
        return {"ok": False, "check": "distill_logits", "error": f"{type(exc).__name__}: {exc}"}
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _run_prune(algorithm: str, data_dir: Path | None) -> dict[str, object]:
    from app.jobs.compression import prune

    root, teacher_pt, auto_data = _prep()
    try:
        out_dir = root / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
        params: dict[str, object] = {
            "algorithm": algorithm,
            "rate": 0.25,
            "input_shape": "1,3,24,24",
        }
        if algorithm in {"group_fisher", "autoslim", "dmcp"}:
            source = data_dir or auto_data
            params.update({"data_dir": str(source), "fisher_image_size": 24,
                           "fisher_iters": 6, "fisher_batch_size": 4})
            if algorithm == "autoslim":
                params.update({"autoslim_epochs": 1, "autoslim_l1": 1e-2})
        _, result = prune(_model_stub(teacher_pt), params, out_dir, lambda *_: None)
        return {
            "ok": True,
            "check": f"prune_{algorithm}",
            "result": result,
            "files": sorted(p.name for p in out_dir.iterdir()),
        }
    except Exception as exc:
        return {"ok": False, "check": f"prune_{algorithm}", "error": f"{type(exc).__name__}: {exc}"}
    finally:
        shutil.rmtree(root, ignore_errors=True)


def run_prune_l1_smoke() -> dict[str, object]:
    """Structured L1 pruning smoke on an auto CNN teacher."""
    return _run_prune("l1", None)


def run_prune_group_fisher_smoke() -> dict[str, object]:
    """Group Fisher pruning smoke (Fisher estimated on the auto dataset)."""
    return _run_prune("group_fisher", None)


def run_prune_autoslim_smoke() -> dict[str, object]:
    """AutoSlim smoke (sparse-train BN gamma then prune on auto data)."""
    return _run_prune("autoslim", None)


def run_prune_dmcp_smoke() -> dict[str, object]:
    """DMCP-style (first-order Taylor) prune smoke on auto data."""
    return _run_prune("dmcp", None)


def run_llm_finetune_smoke() -> dict[str, object]:
    """CPU micro LoRA SFT smoke (demo data, built-in micro LM)."""
    from app.jobs.compression import llm_finetune

    root, teacher_pt, _data = _prep()
    try:
        out_dir = root / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
        _, result = llm_finetune(
            _model_stub(teacher_pt),
            {"epochs": 24, "lr": 2e-3, "lora_rank": 8, "max_new": 24},
            out_dir,
            lambda *_: None,
        )
        return {
            "ok": True,
            "check": "llm_finetune",
            "quant_type": result.get("quant_type"),
            "demo_data": result.get("demo_data"),
            "samples": result.get("samples"),
            "lora_rank": result.get("lora_rank"),
            "trainable_params": result.get("trainable_params"),
            "initial_loss": result.get("initial_loss"),
            "final_loss": result.get("final_loss"),
            "generated": result.get("generated"),
            "adapter_pt": (out_dir / "adapter_lora.pt").is_file(),
            "merged_pt": (out_dir / "model_lora_merged.pt").is_file(),
        }
    except Exception as exc:
        return {"ok": False, "check": "llm_finetune", "error": f"{type(exc).__name__}: {exc}"}
    finally:
        shutil.rmtree(root, ignore_errors=True)


def run_llm_finetune_excel_smoke() -> dict[str, object]:
    """Excel/CSV -> instruction set -> LoRA SFT smoke."""
    from app.jobs.compression import llm_finetune

    root, teacher_pt, _data = _prep()
    try:
        instruct = root / "instruct"
        instruct.mkdir()
        (instruct / "qa.csv").write_text(chr(10).join(["question,answer", "hello,hi there", "color,blue"]) + chr(10), encoding="utf-8")
        out_dir = root / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
        _, result = llm_finetune(
            _model_stub(teacher_pt),
            {"epochs": 8, "lr": 1e-3, "lora_rank": 4, "max_new": 8, "data_dir": str(instruct)},
            out_dir,
            lambda *_: None,
        )
        return {
            "ok": result.get("demo_data") is False and result.get("samples") == 2,
            "check": "llm_finetune_excel",
            "demo_data": result.get("demo_data"),
            "samples": result.get("samples"),
            "generated": result.get("generated"),
            "merged_pt": (out_dir / "model_lora_merged.pt").is_file(),
        }
    except Exception as exc:
        return {"ok": False, "check": "llm_finetune_excel", "error": f"{type(exc).__name__}: {exc}"}
    finally:
        shutil.rmtree(root, ignore_errors=True)


def run_llm_finetune_gpu_smoke() -> dict[str, object]:
    """GPU HF LoRA SFT smoke on the real Qwen2.5-0.5B snapshot.

    Skips cleanly on CPU-only hosts (no CUDA / no cached HF model) so the
    server selfcheck page keeps working.
    """
    import torch

    if not torch.cuda.is_available():
        return {"ok": True, "check": "llm_finetune_gpu", "skipped": "no CUDA on this host"}
    cache = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots"
    snaps = sorted(p for p in cache.glob("*") if p.is_dir()) if cache.exists() else []
    if not snaps:
        return {"ok": True, "check": "llm_finetune_gpu", "skipped": "Qwen2.5-0.5B snapshot not cached"}
    from app.jobs.compression import llm_finetune

    root, _teacher_pt, _data = _prep()
    try:
        out_dir = root / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
        stub = type("M", (), {})()
        stub.framework = "huggingface"
        stub.file_path = str(snaps[-1])
        stub.source_url = ""
        _, result = llm_finetune(
            stub,
            {"data": "hello => hi there", "epochs": 1, "lr": 3e-4,
             "lora_rank": 8, "hf_max_len": 128},
            out_dir,
            lambda *_: None,
        )
        return {
            "ok": result.get("device") == "cuda",
            "check": "llm_finetune_gpu",
            "device": result.get("device"),
            "hf_lora_layers": result.get("hf_lora_layers"),
            "initial_loss": result.get("initial_loss"),
            "final_loss": result.get("final_loss"),
            "generated": result.get("generated"),
            "adapter_pt": (out_dir / "adapter_lora.pt").is_file(),
        }
    except Exception as exc:
        return {"ok": False, "check": "llm_finetune_gpu", "error": f"{type(exc).__name__}: {exc}"}
    finally:
        shutil.rmtree(root, ignore_errors=True)


def run_qat_lsq_smoke() -> dict[str, object]:
    """QAT/LSQ smoke: learnable-step INT8 weight training on an auto CNN."""
    from app.jobs.compression import qat

    root, teacher_pt, data = _prep()
    try:
        out_dir = root / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
        _, result = qat(
            _model_stub(teacher_pt),
            {
                "scheme": "lsq",
                "lsq_bits": 8,
                "epochs": 1,
                "lr": 5e-3,
                "batch_size": 4,
                "image_size": 24,
                "input_shape": "1,3,24,24",
                "data_dir": str(data),
            },
            out_dir,
            lambda *_: None,
        )
        return {
            "ok": True,
            "check": "qat_lsq",
            "qat_mode": result.get("qat_mode"),
            "lsq_layers": result.get("lsq_layers"),
            "lsq_bits": result.get("lsq_bits"),
            "baseline_accuracy": result.get("baseline_accuracy"),
            "qat_accuracy": result.get("qat_accuracy"),
            "lsq_qdq": result.get("lsq_qdq"),
            "lsq_qdq_validated": result.get("lsq_qdq_validated"),
            "qdq_onnx": (out_dir / "model_qat.onnx").is_file(),
            "float_onnx": (out_dir / "model_qat_float.onnx").is_file(),
        }
    except Exception as exc:
        return {"ok": False, "check": "qat_lsq", "error": f"{type(exc).__name__}: {exc}"}
    finally:
        shutil.rmtree(root, ignore_errors=True)


_SELFCHECKS: dict[str, object] = {
    "fitnets_distill": run_fitnets_smoke,
    "distill_logits": run_distill_logits_smoke,
    "prune_l1": run_prune_l1_smoke,
    "prune_group_fisher": run_prune_group_fisher_smoke,
    "prune_autoslim": run_prune_autoslim_smoke,
    "prune_dmcp": run_prune_dmcp_smoke,
    "qat_lsq": run_qat_lsq_smoke,
    "llm_finetune": run_llm_finetune_smoke,
    "llm_finetune_excel": run_llm_finetune_excel_smoke,
    "llm_finetune_gpu": run_llm_finetune_gpu_smoke,
}


def run_selfcheck(check: str) -> dict[str, object]:
    fn = _SELFCHECKS.get(check)
    if fn is None:
        return {"ok": False, "check": check, "error": f"unknown selfcheck: {check}"}
    return fn()
