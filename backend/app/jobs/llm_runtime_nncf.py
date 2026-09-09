"""Minimal CPU runtime for ``engine=nncf`` single-forward OpenVINO IR decoders.

The NNCF path exports one *stateless* forward pass (``input_ids +
attention_mask -> logits``) with no KV cache / stateful runtime.  This module
wraps such an IR in a naive generation loop (one full forward per token) and
exposes the small subset of the transformers ``.generate()`` interface that
``app/api/routes/inference.py`` relies on, so ``/chat/completions`` (plain and
SSE) can serve these artifacts until a stateful runtime exists.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import numpy as np


class NncfChatModel:
    """Wrap a pure NNCF IR decoder (``model.xml`` + HF tokenizer/config)."""

    #: lets the API routes pick the naive token-stream generator below.
    naive = True

    def __init__(self, artifact_dir: str | Path, device: str = "CPU") -> None:
        import openvino as ov
        from transformers import AutoConfig, AutoTokenizer

        self._dir = Path(artifact_dir)
        ir_path = self._dir / "model.xml"
        if not ir_path.exists():
            raise FileNotFoundError(
                f"{ir_path} not found; not a pure-NNCF artifact"
            )
        self.tokenizer = AutoTokenizer.from_pretrained(str(self._dir))
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id or 0
        try:
            self.config = AutoConfig.from_pretrained(str(self._dir))
        except Exception:
            self.config = None
        self.max_len: int | None = None
        if self.config is not None:
            for key in ("n_positions", "max_position_embeddings"):
                value = getattr(self.config, key, None)
                if value:
                    self.max_len = int(value)
                    break

        core = ov.Core()
        model = core.read_model(str(ir_path))
        self._input_names = [port.get_any_name() for port in model.inputs]
        if len(self._input_names) < 2:
            raise RuntimeError(
                "expected (input_ids, attention_mask) inputs in the NNCF IR"
            )
        # Keep the batch dimension static but make the sequence dimension
        # dynamic so the naive loop can grow the prompt token by token.
        model.reshape(
            {name: ov.PartialShape([1, -1]) for name in self._input_names}
        )
        self._compiled = core.compile_model(model, device)

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    def _logits_last(self, ids: np.ndarray) -> np.ndarray:
        mask = np.ones_like(ids)
        outputs = self._compiled([ids, mask])
        logits = np.asarray(outputs[0])  # [1, T, vocab]
        return logits[0, -1, :]

    @staticmethod
    def _choose(
        logits: np.ndarray,
        do_sample: bool,
        temperature: float,
        rng: np.random.Generator,
        top_p: float = 1.0,
        min_p: float = 0.0,
    ) -> int:
        if not do_sample or temperature <= 0:
            return int(np.argmax(logits))
        scaled = logits / max(float(temperature), 1e-6)
        scaled = scaled - float(np.max(scaled))
        probs = np.exp(scaled)
        probs = probs / probs.sum()
        # top_p (nucleus) + min_p filtering, combined mask.
        keep = np.ones_like(probs, dtype=bool)
        if top_p < 1.0:
            order = np.argsort(probs)[::-1]
            sorted_probs = probs[order]
            cumulative = np.cumsum(sorted_probs)
            cutoff = int(np.searchsorted(cumulative, float(top_p))) + 1
            keep_idx = order[: max(cutoff, 1)]
            keep = np.zeros_like(probs, dtype=bool)
            keep[keep_idx] = True
        if min_p > 0.0:
            threshold = float(np.max(probs)) * float(min_p)
            keep &= probs >= threshold
        if not keep.any():
            return int(np.argmax(probs))
        masked = np.where(keep, probs, 0.0)
        masked = masked / masked.sum()
        return int(rng.choice(masked.shape[0], p=masked))

    def _to_sequence(self, input_ids: Any) -> np.ndarray:
        import torch

        if torch.is_tensor(input_ids):
            values = input_ids[0].tolist()
        else:
            values = np.asarray(input_ids)[0].tolist()
        sequence = [int(x) for x in values]
        if self.max_len:
            sequence = sequence[: self.max_len]
        return np.asarray([sequence], dtype=np.int64)

    def _eos_token_id(self, eos_token_id: int | None) -> int | None:
        if eos_token_id is not None:
            return int(eos_token_id)
        return self.tokenizer.eos_token_id

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def generate(
        self,
        input_ids: Any = None,
        attention_mask: Any = None,  # noqa: ARG002 - accepted for HF parity
        max_new_tokens: int = 64,
        do_sample: bool = False,
        temperature: float = 0.7,
        top_p: float = 1.0,
        min_p: float = 0.0,
        eos_token_id: int | None = None,
        **_: Any,
    ) -> Any:
        """Generate and return the full sequence like HF's ``generate``.

        Returns an int64 tensor shaped ``[1, total]`` so the API route can
        keep slicing ``output_ids[0][prompt_len:]`` unchanged.
        """
        import torch

        ids = self._to_sequence(input_ids)
        eos = self._eos_token_id(eos_token_id)
        rng = np.random.default_rng()
        for _ in range(max(1, int(max_new_tokens))):
            if self.max_len is not None and ids.shape[1] >= self.max_len:
                break
            nxt = self._choose(
                self._logits_last(ids),
                bool(do_sample),
                float(temperature),
                rng,
                float(top_p),
                float(min_p),
            )
            if eos is not None and nxt == eos:
                break
            ids = np.concatenate(
                [ids, np.asarray([[nxt]], dtype=np.int64)], axis=1
            )
        return torch.tensor(ids, dtype=torch.int64)

    def generate_text_stream(
        self,
        input_ids: Any = None,
        attention_mask: Any = None,  # noqa: ARG002 - accepted for HF parity
        max_new_tokens: int = 64,
        do_sample: bool = False,
        temperature: float = 0.7,
        top_p: float = 1.0,
        min_p: float = 0.0,
        stop_event: Any = None,
        **_: Any,
    ) -> Iterator[str]:
        """Yield one decoded text piece per generated token (SSE chunks)."""
        ids = self._to_sequence(input_ids)
        eos = self._eos_token_id(None)
        rng = np.random.default_rng()
        for _ in range(max(1, int(max_new_tokens))):
            if stop_event is not None and stop_event.is_set():
                break
            if self.max_len is not None and ids.shape[1] >= self.max_len:
                break
            nxt = self._choose(
                self._logits_last(ids),
                bool(do_sample),
                float(temperature),
                rng,
                float(top_p),
                float(min_p),
            )
            if eos is not None and nxt == eos:
                break
            ids = np.concatenate(
                [ids, np.asarray([[nxt]], dtype=np.int64)], axis=1
            )
            piece = self.tokenizer.decode([nxt], skip_special_tokens=True)
            if piece:
                yield piece
