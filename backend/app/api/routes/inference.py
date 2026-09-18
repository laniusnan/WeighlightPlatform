"""On-demand model inference (serving) for models and compression results.

Loads artifacts lazily (CPU) and exposes a predict endpoint plus an
OpenAI-compatible chat completions endpoint for LLM artifacts.
"""
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from app.api.deps import CurrentUser, SessionDep
from app.jobs.compression import _make_runner
from app.models import CompressionTask, Model, TaskStatus

router = APIRouter(prefix="/inference", tags=["inference"])


class PredictRequest(BaseModel):
    task_id: str | None = None
    model_id: str | None = None
    # Nested list shaped like the model input (e.g. [[[[...]]]] for 1x3x224x224).
    inputs: Any
    input_shape: list[int] | None = None


class PredictResponse(BaseModel):
    output_shape: list[int]
    output: Any
    latency_ms: float


def _resolve_artifact(
    session: SessionDep,
    current_user: CurrentUser,
    task_id: str | None,
    model_id: str | None,
) -> Path:
    if task_id:
        task = session.get(CompressionTask, task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        if not current_user.is_superuser and task.owner_id != current_user.id:
            raise HTTPException(status_code=403, detail="Not enough permissions")
        if task.status != TaskStatus.succeeded.value or not task.output_path:
            raise HTTPException(
                status_code=422, detail="Task has no usable output"
            )
        return Path(task.output_path)
    if model_id:
        model = session.get(Model, model_id)
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        if not current_user.is_superuser and model.owner_id != current_user.id:
            raise HTTPException(status_code=403, detail="Not enough permissions")
        if not model.file_path:
            raise HTTPException(status_code=422, detail="Model has no local file")
        return Path(model.file_path)
    raise HTTPException(status_code=422, detail="Provide task_id or model_id")


def _infer_input_shape(path: Path) -> list[int] | None:
    """Best-effort: read the model's declared input shape."""
    if path.is_dir() and (path / "model.xml").exists():
        import openvino as ov

        model = ov.Core().read_model(str(path / "model.xml"))
        shape = next(iter(model.inputs)).partial_shape
        dims = []
        for dim in shape.get_min_shape():
            value = int(dim) if dim > 0 else 1
            dims.append(value)
        return dims if dims else None
    if path.suffix.lower() == ".onnx":
        import onnxruntime as ort

        session = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )
        shape = session.get_inputs()[0].shape
        if all(isinstance(dim, int) and dim > 0 for dim in shape):
            return [int(dim) for dim in shape]
    return None


@router.post("/predict", response_model=PredictResponse)
def predict(
    body: PredictRequest,
    session: SessionDep,
    current_user: CurrentUser,
) -> Any:
    """Run a single prediction using a compression result or a model file."""
    import time

    path = _resolve_artifact(session, current_user, body.task_id, body.model_id)
    from app.jobs.compression import resolve_artifact_file

    path = resolve_artifact_file(path)
    from app.core.config import settings

    if str(settings.COMPUTE_BACKEND).strip().lower() == "gpu":
        try:
            from app.services import get_service_bundle

            out = get_service_bundle("gpu").inference.run_predict(
                path, body.inputs, input_shape=body.input_shape
            )
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"GPU inference failed: {exc}") from exc
        out_arr = np.asarray(out)
        return PredictResponse(
            output_shape=list(out_arr.shape),
            output=out_arr.tolist(),
            latency_ms=round(0.0, 3),
        )
    try:
        runner = _make_runner(path)
    except Exception as exc:
        raise HTTPException(
            status_code=422, detail=f"Cannot load model for inference: {exc}"
        )

    input_shape = body.input_shape
    if input_shape is None:
        inferred = _infer_input_shape(path)
        if inferred:
            input_shape = inferred

    try:
        arr = np.asarray(body.inputs, dtype=np.float32)
        if input_shape:
            arr = arr.reshape([int(x) for x in input_shape])
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid inputs: {exc}. Provide input_shape or a model with a known input shape.",
        )

    try:
        start = time.perf_counter()
        output = runner(arr)
        latency_ms = (time.perf_counter() - start) * 1000
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Inference failed: {exc}")
    out_arr = np.asarray(output)
    return PredictResponse(
        output_shape=list(out_arr.shape),
        output=out_arr.tolist(),
        latency_ms=round(float(latency_ms), 3),
    )


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    task_id: str
    messages: list[ChatMessage]
    max_new_tokens: int = 64
    temperature: float = 0.7
    top_p: float = 1.0
    min_p: float = 0.0
    stream: bool = False


def _load_chat_model(artifact: Path) -> tuple[Any, Any]:
    """Load the OpenVINO IR LLM + tokenizer for a chat session.

    Two artifact flavours are supported:
    * ``engine=nncf`` single-forward IR (no KV cache) -> ``NncfChatModel``
      naive generator (``model.xml`` + ``engine_nncf.txt`` marker);
    * optimum-intel exported IR -> ``OVModelForCausalLM``.
    """
    pure_nncf = (
        (artifact / "engine_nncf.txt").exists()
        or (
            (artifact / "model.xml").exists()
            and not (artifact / "openvino_model.xml").exists()
            and not (artifact / "openvino_config.json").exists()
        )
    )
    if pure_nncf:
        from app.jobs.llm_runtime_nncf import NncfChatModel

        model = NncfChatModel(str(artifact), device="CPU")
        return model, model.tokenizer
    from optimum.intel import OVModelForCausalLM
    from transformers import AutoTokenizer

    model = OVModelForCausalLM.from_pretrained(
        str(artifact), device="CPU", compile=False
    )
    tokenizer = AutoTokenizer.from_pretrained(str(artifact))
    return model, tokenizer


@router.post("/chat/completions")
async def chat_completions(
    body: ChatRequest,
    request: Request,
    session: SessionDep,
    current_user: CurrentUser,
) -> Any:
    """OpenAI-compatible chat completions for LLM compression results.

    Supports one-shot responses and SSE token streaming (``stream: true``).
    Streaming stops promptly when the client disconnects. Expects an OpenVINO
    IR LLM artifact (from ``llm_quantize``); runs on CPU via optimum-intel.
    """
    import asyncio
    import json
    import queue as _queue
    import threading
    import time

    task = session.get(CompressionTask, body.task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    if not current_user.is_superuser and task.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    if task.status != TaskStatus.succeeded.value or not task.output_path:
        raise HTTPException(status_code=422, detail="Task has no usable output")
    artifact = Path(task.output_path)

    from app.core.config import settings

    if str(settings.COMPUTE_BACKEND).strip().lower() == "gpu":
        if body.stream:
            raise HTTPException(
                status_code=501,
                detail="GPU backend: SSE streaming not implemented yet - set stream=false",
            )
        try:
            from transformers import AutoTokenizer

            from app.services import get_service_bundle

            gpu_tok = AutoTokenizer.from_pretrained(str(artifact))
            rendered = gpu_tok.apply_chat_template(
                [m.model_dump() for m in body.messages],
                tokenize=False,
                add_generation_prompt=True,
            )
            result = get_service_bundle("gpu").inference.chat_completion(
                artifact,
                rendered,
                already_formatted=True,
                max_new_tokens=max(body.max_new_tokens, 1),
                temperature=body.temperature,
                top_p=body.top_p,
                min_p=body.min_p,
            )
            return {
                "id": f"chatcmpl-{task.id}",
                "object": "chat.completion",
                "created": int(__import__("time").time()),
                "model": task.model.name if task.model else str(task.model_id),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": result["text"]},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": int(result.get("prompt_tokens", 0)),
                    "completion_tokens": int(result.get("completion_tokens", 0)),
                    "total_tokens": int(result.get("total_tokens", 0)),
                },
                "latency_ms": round(float(result.get("latency_ms", 0.0)), 3),
            }
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=422, detail=f"GPU chat inference failed: {exc}"
            ) from exc

    try:
        model, tokenizer = _load_chat_model(artifact)
        prompt = tokenizer.apply_chat_template(
            [m.model_dump() for m in body.messages],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt")
    except Exception as exc:
        raise HTTPException(
            status_code=422, detail=f"Chat inference failed: {exc}"
        )

    gen_kwargs = {
        "max_new_tokens": max(body.max_new_tokens, 1),
        "do_sample": body.temperature > 0,
        "temperature": body.temperature,
        "top_p": max(min(body.top_p, 1.0), 0.0),
        "min_p": max(min(body.min_p, 1.0), 0.0),
    }

    if body.stream:
        from fastapi.responses import StreamingResponse

        stop_event = threading.Event()
        if getattr(model, "naive", False):
            # engine=nncf single-forward IR (no KV cache): the naive generator
            # streams one decoded text piece per token; no TextIteratorStreamer.
            text_queue: _queue.Queue[str | None] = _queue.Queue()

            def _naive_run() -> None:
                try:
                    for piece in model.generate_text_stream(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs.get("attention_mask"),
                        stop_event=stop_event,
                        **gen_kwargs,
                    ):
                        text_queue.put(piece)
                finally:
                    text_queue.put(None)

            thread = threading.Thread(target=_naive_run, daemon=True)
        else:
            from transformers import StoppingCriteria, TextIteratorStreamer

            class _StopOnDisconnect(StoppingCriteria):
                def __call__(
                    self,
                    input_ids: Any,
                    scores: Any,
                    **kwargs: Any,
                ) -> bool:
                    return stop_event.is_set()

            streamer = TextIteratorStreamer(
                tokenizer, skip_prompt=True, skip_special_tokens=True
            )
            text_queue = streamer.text_queue
            thread = threading.Thread(
                target=lambda: model.generate(
                    **inputs,
                    streamer=streamer,
                    stopping_criteria=[_StopOnDisconnect()],
                    **gen_kwargs,
                ),
                daemon=True,
            )
        thread.start()
        created = int(time.time())
        completion_id = f"chatcmpl-{task.id}"

        def _chunk(content: str | None, finish: str | None) -> str:
            return json.dumps(
                {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": task.model.name if task.model else str(task.model_id),
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": content} if content else {},
                            "finish_reason": finish,
                        }
                    ],
                },
                ensure_ascii=False,
            )

        async def event_gen():
            try:
                while True:
                    if await request.is_disconnected():
                        stop_event.set()
                        break
                    try:
                        text = text_queue.get(timeout=0.5)
                    except _queue.Empty:
                        if not thread.is_alive():
                            break
                        continue
                    if text is None:
                        break
                    if text:
                        yield f"data: {_chunk(text, None)}\n\n"
            finally:
                stop_event.set()
            yield f"data: {_chunk(None, 'stop')}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            event_gen(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    start = time.perf_counter()
    output_ids = await asyncio.to_thread(model.generate, **inputs, **gen_kwargs)
    latency_ms = (time.perf_counter() - start) * 1000
    generated = output_ids[0][inputs.input_ids.shape[1] :]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return {
        "id": f"chatcmpl-{task.id}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": task.model.name if task.model else str(task.model_id),
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": int(inputs.input_ids.shape[1]),
            "completion_tokens": int(generated.shape[0]),
            "total_tokens": int(inputs.input_ids.shape[1] + generated.shape[0]),
        },
        "latency_ms": round(float(latency_ms), 3),
    }

