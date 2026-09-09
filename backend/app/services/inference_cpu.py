"""CPU inference service.

Delegates to the exact runtime paths the API already uses:
* ``run_predict``   -> ``app.jobs.compression._make_runner`` (ONNX/OV/torch CPU);
* ``chat_completion`` -> ``app.api.routes.inference._load_chat_model``
  (optimum OVModelForCausalLM OR engine=nncf NncfChatModel naive runtime).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from app.services.base import InferenceService

logger = logging.getLogger("app.services.inference_cpu")


class InferenceCpuService(InferenceService):
    name = "inference.cpu"
    backend = "cpu"

    def run_predict(self, artifact: Any, inputs: Any, **kwargs: Any) -> Any:
        import numpy as np

        from app.jobs.compression import _make_runner

        runner = _make_runner(Path(artifact))
        arr = np.asarray(inputs, dtype=np.float32)
        if kwargs.get("input_shape"):
            arr = arr.reshape([int(x) for x in kwargs["input_shape"]])
        return np.asarray(runner(arr))

    def chat_completion(
        self,
        artifact: Any,
        prompt: str,
        *,
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        top_p: float = 1.0,
        min_p: float = 0.0,
        **kwargs: Any,
    ) -> dict[str, Any]:
        from app.api.routes.inference import _load_chat_model

        logger.info("cpu chat completion artifact=%s", artifact)
        model, tokenizer = _load_chat_model(Path(artifact))
        inputs = tokenizer(prompt, return_tensors="pt")
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            max_new_tokens=max(int(max_new_tokens), 1),
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=top_p,
            min_p=min_p,
            **kwargs,
        )
        generated = output_ids[0][inputs["input_ids"].shape[1] :]
        text = tokenizer.decode(generated, skip_special_tokens=True)
        return {
            "text": text,
            "prompt_tokens": int(inputs["input_ids"].shape[1]),
            "completion_tokens": int(generated.shape[0]),
            "total_tokens": int(inputs["input_ids"].shape[1] + generated.shape[0]),
            "backend": self.backend,
        }


def get_inference_cpu_service() -> InferenceCpuService:
    return InferenceCpuService()
