"""GPU inference service (first GPU implementation, NVIDIA CUDA).

Supported artifact flavours:
* HuggingFace local dir (config + safetensors/bin) -> fp16 CUDA decoder,
  naive per-token generation (no KV cache yet) - mirrors the CPU chat flow.
* ``.onnx`` file -> onnxruntime with ``CUDAExecutionProvider`` (falls back to
  CPU EP when the CUDA EP is not installed).
* OpenVINO ``model.xml`` -> attempts the OV ``GPU`` plugin (Intel GPUs); on
  NVIDIA hosts this is not available, so it logs a warning and runs CPU via the
  existing runner (keeps behaviour predictable while migration is partial).

Enabling: set ``COMPUTE_BACKEND=gpu`` AND make sure torch.cuda.is_available().
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from app.services.base import InferenceService, NotImplementedBackendError

logger = logging.getLogger("app.services.inference_gpu")


def _require_cuda() -> Any:
    """Return the torch module or raise a clear error."""
    import torch

    if not torch.cuda.is_available():
        raise NotImplementedBackendError(
            "COMPUTE_BACKEND=gpu requires torch.cuda.is_available()==True "
            "(install the cu126/cu128 wheels, see backend/requirements-gpu.txt)"
        )
    return torch


class InferenceGpuService(InferenceService):
    name = "inference.gpu"
    backend = "gpu"

    # ------------------------------------------------------------------ #
    # generic predict
    # ------------------------------------------------------------------ #
    def run_predict(self, artifact: Any, inputs: Any, **kwargs: Any) -> Any:
        path = Path(artifact)

        # ONNX -> onnxruntime CUDA EP when available.
        if path.suffix.lower() == ".onnx" or (path.is_dir() and not (path / "model.xml").exists()):
            return self._run_onnx(path, np.asarray(inputs, dtype=np.float32), kwargs)

        # OpenVINO IR -> GPU plugin (Intel), CPU fallback on NVIDIA hosts.
        if path.is_dir() and (path / "model.xml").exists():
            return self._run_openvino(path, np.asarray(inputs, dtype=np.float32))

        raise NotImplementedBackendError(
            "run_predict GPU: unsupported artifact. Use an .onnx file or an "
            "OpenVINO model.xml directory (chat HF dirs go through "
            "chat_completion)."
        )

    def _run_onnx(self, path: Path, arr: np.ndarray, kwargs: dict[str, Any]) -> np.ndarray:
        import onnxruntime as ort

        shape = kwargs.get("input_shape")
        if shape:
            arr = arr.reshape([int(x) for x in shape])
        available = ort.get_available_providers()
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in available
            else ["CPUExecutionProvider"]
        )
        if "CUDAExecutionProvider" not in providers:
            logger.warning(
                "onnxruntime CUDA EP unavailable (providers=%s); using CPU EP",
                available,
            )
        session = ort.InferenceSession(str(path), providers=providers)
        name = session.get_inputs()[0].name
        return np.asarray(session.run(None, {name: arr}))

    def _run_openvino(self, path: Path, arr: np.ndarray) -> np.ndarray:
        import openvino as ov

        core = ov.Core()
        try:
            compiled = core.compile_model(str(path / "model.xml"), "GPU")
        except Exception as exc:
            logger.warning(
                "OpenVINO GPU plugin unavailable (%s); falling back to CPU. "
                "(OV GPU targets Intel iGPU/dGPU; NVIDIA hosts use ONNX/CUDA "
                "or torch CUDA paths instead.)",
                exc,
            )
            compiled = core.compile_model(str(path / "model.xml"), "CPU")
        input_arr = arr.reshape(1, *arr.shape) if arr.ndim == 3 else arr
        return np.asarray(compiled([input_arr])[0])

    # ------------------------------------------------------------------ #
    # LLM chat
    # ------------------------------------------------------------------ #
    @staticmethod
    def _sample(
        logits: Any, temperature: float, top_p: float, min_p: float, do_sample: bool
    ) -> int:
        import torch

        if not do_sample or temperature <= 0:
            return int(torch.argmax(logits, dim=-1).item())
        scaled = logits / max(float(temperature), 1e-6)
        probs = torch.softmax(scaled.float(), dim=-1)
        keep = torch.ones_like(probs, dtype=torch.bool)
        if top_p < 1.0:
            sorted_probs, _ = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            cutoff = int(torch.searchsorted(cumsum, float(top_p))) + 1
            top_ids = torch.topk(probs, k=max(cutoff, 1)).indices
            keep = torch.zeros_like(probs, dtype=torch.bool)
            keep[top_ids] = True
        if min_p > 0.0:
            keep &= probs >= (float(probs.max()) * float(min_p))
        if not bool(keep.any()):
            return int(torch.argmax(probs).item())
        masked = torch.where(keep, probs, torch.zeros_like(probs))
        return int(torch.multinomial(masked / masked.sum(), 1).item())

    def chat_completion(
        self,
        artifact: Any,
        prompt: str,
        *,
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        top_p: float = 1.0,
        min_p: float = 0.0,
        already_formatted: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        import time

        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        torch = _require_cuda()
        device = "cuda"
        start = time.perf_counter()
        repo = Path(artifact)
        if not (repo / "config.json").is_file():
            raise NotImplementedBackendError(
                f"chat_completion GPU expects a local HuggingFace dir with "
                f"config.json, got {repo}"
            )
        logger.info("loading HF model %s in fp16 on cuda...", repo)
        tokenizer = AutoTokenizer.from_pretrained(str(repo))
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id or 0
        # Instruct models need their chat template for coherent completions.
        if not already_formatted and getattr(tokenizer, "chat_template", None):
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        model = AutoModelForCausalLM.from_pretrained(
            str(repo), torch_dtype=torch.float16
        ).to(device).eval()
        try:
            config = AutoConfig.from_pretrained(str(repo))
        except Exception:
            config = None
        max_len = None
        if config is not None:
            for key in ("n_positions", "max_position_embeddings"):
                value = getattr(config, key, None)
                if value:
                    max_len = int(value)
                    break
        load_s = time.perf_counter() - start

        ids = torch.as_tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
        eos = tokenizer.eos_token_id
        n = max(int(max_new_tokens), 1)
        for _ in range(n):
            if max_len is not None and ids.shape[-1] >= max_len:
                break
            with torch.inference_mode():
                out = model(input_ids=ids)
            logits = out.logits[:, -1, :]  # [1, vocab]
            nxt = self._sample(logits[0], temperature, top_p, min_p, temperature > 0)
            if eos is not None and nxt == eos:
                break
            ids = torch.cat([ids, torch.tensor([[nxt]], device=device)], dim=-1)
        gen = ids[0][len(tokenizer.encode(prompt)) :].tolist()
        text = tokenizer.decode(gen, skip_special_tokens=True)
        return {
            "text": text,
            "prompt_tokens": int(len(tokenizer.encode(prompt))),
            "completion_tokens": int(len(gen)),
            "total_tokens": int(len(tokenizer.encode(prompt)) + len(gen)),
            "load_seconds": round(load_s, 3),
            "latency_ms": round((time.perf_counter() - start) * 1000, 3),
            "device": device,
            "backend": self.backend,
        }
