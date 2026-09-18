"""Drop-in FastAPI router: 把「模型压缩」作为推理模块的子功能接入。

设计要点
--------
* **单文件、零耦合**：只依赖 `fastapi` + `httpx` + `pydantic`，不 import 压缩平台代码；
  平台通过 HTTP 调用（`COMPRESSION_API_BASE`），两个站点可独立部署与升级。
* **异步任务**：`POST /tasks` 立即返回 `task_id`；前端用 `GET /tasks/{id}` 轮询 `progress`。
* **产物不下沉**：推理时只传 `task_id`，平台按 id 定位产物（`/inference/predict`、
  `/inference/chat/completions`）；需要离线包时再 `GET /tasks/{id}/deploy` 流式下载。
* **方法目录由平台侧驱动**：`GET /methods` 返回 UI 方法 → 平台 method + 默认参数 + 表单字段，
  前端据此渲染下拉框与高级参数表单（平台新增方法时前端无需改代码）。
* **环境变量可覆盖**：`COMPRESSION_API_BASE` / `COMPRESSION_SERVICE_TOKEN`（或
  `COMPRESSION_EMAIL`+`COMPRESSION_PASSWORD`）/ `COMPRESSION_TIMEOUT` / `COMPRESSION_VERIFY_SSL`。

挂载（你网站的推理模块，例如 `app/main.py`）::

    from api.compression_proxy import router as compression_router
    app.include_router(compression_router)      # 自带 /api/compression 前缀

使用流程与前端对接见同目录 README.md；本地验证脚本见同目录 test_harness.py。
"""
from __future__ import annotations

import os
from typing import Any, Literal

import httpx
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- #
# 配置（环境变量）
# --------------------------------------------------------------------------- #
API_BASE = os.getenv("COMPRESSION_API_BASE", "http://127.0.0.1:8001/api/v1").rstrip("/")
SERVICE_TOKEN = os.getenv("COMPRESSION_SERVICE_TOKEN", "")
SERVICE_EMAIL = os.getenv("COMPRESSION_EMAIL", "")
SERVICE_PASSWORD = os.getenv("COMPRESSION_PASSWORD", "")
TIMEOUT = float(os.getenv("COMPRESSION_TIMEOUT", "60"))
VERIFY_SSL = os.getenv("COMPRESSION_VERIFY_SSL", "true").lower() not in {"0", "false", "no"}

router = APIRouter(prefix="/api/compression", tags=["compression"])

_token_cache: dict[str, str] = {}


async def _token(client: httpx.AsyncClient, force: bool = False) -> str | None:
    """服务账号 token（缓存；401 时强制刷新）。"""
    if SERVICE_TOKEN:
        return SERVICE_TOKEN
    if _token_cache.get("token") and not force:
        return _token_cache["token"]
    if not (SERVICE_EMAIL and SERVICE_PASSWORD):
        return None
    response = await client.post(
        f"{API_BASE}/login/access-token",
        data={"username": SERVICE_EMAIL, "password": SERVICE_PASSWORD},
    )
    if response.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"compression login failed: {response.text[:200]}")
    _token_cache["token"] = response.json()["access_token"]
    return _token_cache["token"]


def _detail(response: httpx.Response) -> Any:
    try:
        return response.json().get("detail", response.text[:300])
    except Exception:  # noqa: BLE001 - non-JSON error body
        return response.text[:300]


async def _request(
    method: str,
    path: str,
    *,
    json: Any = None,
    params: dict[str, Any] | None = None,
    retry_auth: bool = True,
) -> Any:
    """调用平台 API，并把错误翻译成等价状态码（404/422 原样，5xx → 502）。"""
    headers: dict[str, str] = {}
    async with httpx.AsyncClient(base_url=API_BASE, timeout=TIMEOUT, verify=VERIFY_SSL) as client:
        token = await _token(client)
        if token:
            headers["Authorization"] = f"Bearer {token}"
        response = await client.request(method, path, json=json, params=params, headers=headers)
        if response.status_code == 401 and retry_auth and not SERVICE_TOKEN:
            token = await _token(client, force=True)
            headers["Authorization"] = f"Bearer {token}"
            response = await client.request(method, path, json=json, params=params, headers=headers)
    if response.status_code == 404:
        raise HTTPException(status_code=404, detail=response.text[:300])
    if response.status_code == 422:
        raise HTTPException(status_code=422, detail=_detail(response))
    if response.status_code >= 500:
        raise HTTPException(status_code=502,
                            detail=f"compression platform error: {response.text[:300]}")
    if response.status_code >= 400:
        raise HTTPException(status_code=response.status_code, detail=_detail(response))
    return response.json() if response.content else {}


# --------------------------------------------------------------------------- #
# 方法目录：UI 方法 → 平台 method + 默认参数 + 表单字段
# --------------------------------------------------------------------------- #
METHOD_CATALOG: dict[str, dict[str, Any]] = {
    "int4": {
        "label": "INT4 量化（LLM / OpenVINO IR）",
        "platform_method": "llm_quantize",
        "params": {"engine": "nncf", "bits": 4, "group_size": 128, "sym": "true"},
        "fields": [
            {"name": "bits", "type": "int", "default": 4, "choices": [4, 8]},
            {"name": "group_size", "type": "int", "default": 128, "choices": [32, 64, 128, 256]},
            {"name": "sym", "type": "bool", "default": True},
        ],
    },
    "int8_openvino": {
        "label": "INT8 量化（OpenVINO + NNCF，CPU 推荐）",
        "platform_method": "openvino_int8",
        "params": {"precision": "int8", "model_kind": "cnn", "calibration_samples": 8},
        "fields": [
            {"name": "precision", "type": "str", "default": "int8", "choices": ["int8", "fp16"]},
            {"name": "calibration_samples", "type": "int", "default": 8},
            {"name": "calib_dataset_id", "type": "dataset", "default": None},
        ],
    },
    "onnx_int8": {
        "label": "ONNX 动态 INT8",
        "platform_method": "quantize_onnx",
        "params": {"opset": 12},
        "fields": [{"name": "opset", "type": "int", "default": 12}],
    },
    "onnx_static_int8": {
        "label": "ONNX 静态 INT8（QDQ，需校准集）",
        "platform_method": "quantize_onnx_static",
        "params": {"image_size": 224, "calibration_samples": 16},
        "fields": [
            {"name": "image_size", "type": "int", "default": 224},
            {"name": "calibration_samples", "type": "int", "default": 16},
            {"name": "calib_dataset_id", "type": "dataset", "default": None},
        ],
    },
    "gguf_q4km": {
        "label": "GGUF q4_K_M（llama.cpp / ollama）",
        "platform_method": "export_gguf",
        "params": {"outtype": "q4_K_M"},
        "fields": [
            {"name": "outtype", "type": "str", "default": "q4_K_M",
             "choices": ["f16", "bf16", "q8_0", "q4_0", "q4_K", "q4_K_S", "q4_K_M", "q6_K"]},
        ],
    },
    "prune_l2": {
        "label": "结构化剪枝（L2 范数）",
        "platform_method": "prune",
        "params": {"algorithm": "l2", "rate": 0.3, "export_onnx": True},
        "fields": [
            {"name": "rate", "type": "float", "default": 0.3, "min": 0.05, "max": 0.95},
            {"name": "input_shape", "type": "shape", "default": [1, 3, 224, 224]},
        ],
    },
    "prune_fisher": {
        "label": "结构化剪枝（Group Fisher，需数据集）",
        "platform_method": "prune",
        "params": {"algorithm": "group_fisher", "rate": 0.3, "export_onnx": True},
        "requires_dataset": True,
        "fields": [
            {"name": "rate", "type": "float", "default": 0.3},
            {"name": "dataset_id", "type": "dataset", "required": True},
            {"name": "input_shape", "type": "shape", "default": [1, 3, 224, 224]},
        ],
    },
    "distill_logits": {
        "label": "知识蒸馏（Hinton logits KD）",
        "platform_method": "distill",
        "params": {"method": "logits", "temperature": 4.0, "alpha": 0.7, "width_mult": 0.5},
        "requires_dataset": True,
        "fields": [
            {"name": "temperature", "type": "float", "default": 4.0},
            {"name": "alpha", "type": "float", "default": 0.7},
            {"name": "width_mult", "type": "float", "default": 0.5},
            {"name": "dataset_id", "type": "dataset", "required": True},
        ],
    },
    "distill_features": {
        "label": "知识蒸馏（FitNets 特征对齐）",
        "platform_method": "distill",
        "params": {"method": "features", "temperature": 4.0, "alpha": 0.7, "width_mult": 0.5},
        "requires_dataset": True,
        "fields": [
            {"name": "temperature", "type": "float", "default": 4.0},
            {"name": "alpha", "type": "float", "default": 0.7},
            {"name": "width_mult", "type": "float", "default": 0.5},
            {"name": "dataset_id", "type": "dataset", "required": True},
        ],
    },
    "lora": {
        "label": "LoRA SFT 微调",
        "platform_method": "llm_finetune",
        "params": {"stage": "sft", "epochs": 1, "lora_rank": 8, "hf_max_len": 128},
        "requires_dataset": True,
        "fields": [
            {"name": "epochs", "type": "int", "default": 1},
            {"name": "lr", "type": "float", "default": 3e-3},
            {"name": "lora_rank", "type": "int", "default": 8},
            {"name": "dataset_id", "type": "dataset", "required": True},
        ],
    },
    "dpo": {
        "label": "DPO 偏好优化",
        "platform_method": "llm_finetune",
        "params": {"stage": "dpo", "epochs": 1, "dpo_beta": 0.1, "lora_rank": 8},
        "requires_dataset": True,
        "fields": [
            {"name": "dpo_beta", "type": "float", "default": 0.1},
            {"name": "epochs", "type": "int", "default": 1},
            {"name": "dataset_id", "type": "dataset", "required": True},
        ],
    },
}


@router.get("/methods", summary="可用的压缩方法（驱动前端表单）")
async def list_methods() -> dict[str, Any]:
    """返回 UI 方法目录：前端下拉框 + 高级参数表单都由它渲染。"""
    return {
        "methods": [
            {
                "key": key,
                "label": spec["label"],
                "platform_method": spec["platform_method"],
                "defaults": spec["params"],
                "requires_dataset": spec.get("requires_dataset", False),
                "fields": spec["fields"],
            }
            for key, spec in METHOD_CATALOG.items()
        ]
    }


# --------------------------------------------------------------------------- #
# 请求模型
# --------------------------------------------------------------------------- #
class SubmitRequest(BaseModel):
    """提交压缩任务：UI 方法 + 参数覆盖（可选评估/基准配置）。"""

    model_id: str = Field(..., description="平台模型 UUID 或名称（唯一前缀也可）")
    method: str = Field(..., description="METHOD_CATALOG 的 key，如 int4 / prune_fisher / gguf_q4km")
    params: dict[str, Any] = Field(default_factory=dict, description="覆盖默认参数（高级表单）")
    eval: dict[str, Any] | None = Field(
        default=None,
        description='可选评估配置，如 {"type":"classification","dataset_id":"…","image_size":32}',
    )
    benchmark: dict[str, Any] | None = None


class ReevaluateRequest(BaseModel):
    kind: Literal["eval", "benchmark"] = "eval"


class InferRequest(BaseModel):
    """用压缩产物推理：chat=True（或带 messages）走 chat/completions，否则走 predict。"""

    task_id: str
    chat: bool = False
    inputs: list[Any] | None = None
    input_shape: list[int] | None = None
    messages: list[dict[str, str]] | None = None
    max_tokens: int = 128
    temperature: float = 0.7


def _normalise_task(task: dict[str, Any]) -> dict[str, Any]:
    """平台任务对象 → 前端友好结构（含 ready / 下载链接）。"""
    status = task.get("status")
    ready = status == "succeeded" and bool(task.get("output_size_bytes"))
    task_id = task.get("id")
    payload: dict[str, Any] = {
        "task_id": task_id,
        "model_id": task.get("model_id"),
        "model_name": task.get("model_name"),
        "method": task.get("method"),
        "status": status,
        "progress": task.get("progress"),
        "error": task.get("error_message"),
        "input_size_bytes": task.get("input_size_bytes"),
        "output_size_bytes": task.get("output_size_bytes"),
        "metrics": task.get("metrics") or {},
        "created_at": task.get("created_at"),
        "finished_at": task.get("finished_at"),
        "ready": ready,
    }
    if ready:
        payload["artifact_url"] = f"/api/compression/tasks/{task_id}/artifact"
        payload["deploy_url"] = f"/api/compression/tasks/{task_id}/deploy"
    return payload


# --------------------------------------------------------------------------- #
# 资产盘点（供前端下拉框）
# --------------------------------------------------------------------------- #
@router.get("/platform/models", summary="平台已注册模型")
async def list_platform_models(limit: int = Query(100, le=500)) -> dict[str, Any]:
    page = await _request("GET", "/models/", params={"limit": limit})
    rows = page.get("data", page) if isinstance(page, dict) else page
    return {
        "count": len(rows),
        "models": [
            {"model_id": m.get("id"), "name": m.get("name"), "framework": m.get("framework"),
             "status": m.get("status"), "size_bytes": m.get("size_bytes")}
            for m in rows
        ],
    }


@router.get("/platform/datasets", summary="平台数据集（含服务器本地开放数据集）")
async def list_platform_datasets(limit: int = Query(100, le=500)) -> dict[str, Any]:
    page = await _request("GET", "/datasets/", params={"limit": limit})
    rows = page.get("data", page) if isinstance(page, dict) else page
    return {
        "count": len(rows),
        "datasets": [
            {"dataset_id": d.get("id"), "name": d.get("name"),
             "source_type": d.get("source_type"), "size_bytes": d.get("size_bytes")}
            for d in rows
        ],
    }


@router.get("/platform/local-datasets", summary="扫描只读根 /datasets-local 下的本地数据集")
async def list_local_datasets() -> dict[str, Any]:
    return await _request("GET", "/datasets/local-dirs")


async def _resolve_model_id(reference: str) -> str:
    """接受模型 UUID 或名称（唯一前缀匹配），与 CLI 行为一致。"""
    page = await _request("GET", "/models/", params={"limit": 200})
    rows = page.get("data", page) if isinstance(page, dict) else page
    for model in rows:
        if str(model.get("id")) == reference:
            return reference
    exact = [m for m in rows if m.get("name") == reference]
    prefix = [m for m in rows if str(m.get("name", "")).startswith(reference)]
    if len(exact) == 1:
        return str(exact[0]["id"])
    if len(prefix) == 1:
        return str(prefix[0]["id"])
    if not exact and not prefix:
        raise HTTPException(status_code=404, detail=f"no model matches {reference!r}")
    candidates = ", ".join(sorted(m["name"] for m in (exact or prefix)))
    raise HTTPException(status_code=422, detail=f"{reference!r} is ambiguous: {candidates}")


# --------------------------------------------------------------------------- #
# 任务端点
# --------------------------------------------------------------------------- #
@router.post("/tasks", status_code=202, summary="提交压缩任务（异步，立即返回 task_id）")
async def submit_task(body: SubmitRequest) -> dict[str, Any]:
    spec = METHOD_CATALOG.get(body.method)
    if spec is None:
        raise HTTPException(status_code=422, detail=f"unknown method {body.method!r}; "
                            f"available: {', '.join(sorted(METHOD_CATALOG))}")
    params: dict[str, Any] = {
        **spec["params"],
        **{k: v for k, v in body.params.items() if v is not None},
    }
    if spec.get("requires_dataset") and not params.get("dataset_id"):
        raise HTTPException(status_code=422, detail=f"method {body.method!r} requires params.dataset_id")
    if body.eval:
        params["eval"] = body.eval
    if body.benchmark:
        params["benchmark"] = body.benchmark

    model_id = await _resolve_model_id(body.model_id)
    task = await _request("POST", "/compressions/", json={
        "model_id": model_id, "method": spec["platform_method"], "params": params,
    })
    if not isinstance(task, dict) or "id" not in task:
        raise HTTPException(status_code=502, detail=f"unexpected platform response: {task}")
    return {
        "task_id": task["id"],
        "status": task.get("status"),
        "platform_method": spec["platform_method"],
        "params": params,
    }


@router.get("/tasks", summary="任务列表（最新在前）")
async def list_tasks(
    status: str | None = Query(None, description="pending/running/succeeded/failed/canceled"),
    limit: int = Query(20, le=200),
) -> dict[str, Any]:
    page = await _request("GET", "/compressions/", params={"limit": limit})
    rows = page.get("data", page) if isinstance(page, dict) else page
    tasks = [_normalise_task(t) for t in rows]
    if status:
        tasks = [t for t in tasks if t["status"] == status]
    return {"count": len(tasks), "tasks": tasks}


@router.get("/tasks/{task_id}", summary="任务状态（前端轮询；progress 0-100）")
async def get_task(task_id: str) -> dict[str, Any]:
    return _normalise_task(await _request("GET", f"/compressions/{task_id}"))


@router.post("/tasks/{task_id}/retry", status_code=202, summary="重试失败的压缩任务")
async def retry_task(task_id: str) -> dict[str, Any]:
    created = await _request("POST", f"/compressions/{task_id}/retry")
    return {"task_id": created.get("id"), "status": created.get("status")}


@router.post("/tasks/{task_id}/reevaluate", summary="重跑已存配置的评估 / 基准")
async def reevaluate_task(task_id: str, body: ReevaluateRequest) -> dict[str, Any]:
    return await _request("POST", f"/compressions/{task_id}/reevaluate", json={"kind": body.kind})


@router.delete("/tasks/{task_id}", summary="删除任务（运行中会先取消）")
async def delete_task(task_id: str) -> dict[str, Any]:
    return await _request("DELETE", f"/compressions/{task_id}")


async def _stream(path: str) -> StreamingResponse:
    """把平台侧二进制端点（zip）流式转发给浏览器，避免大文件进内存。

    注意：``httpx.AsyncClient`` 必须在**流被消费完**之后再关闭，因此这里不使用
    ``async with``（提前退出会关闭连接，客户端会看到 IncompleteRead / 0 字节）。
    """
    client = httpx.AsyncClient(base_url=API_BASE, timeout=max(TIMEOUT, 300), verify=VERIFY_SSL)
    token = await _token(client)
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    request = client.build_request("GET", path, headers=headers)
    response = await client.send(request, stream=True)
    if response.status_code >= 400:
        await response.aclose()
        await client.aclose()
        raise HTTPException(status_code=response.status_code,
                            detail=f"artifact unavailable ({response.status_code})")
    disposition = response.headers.get("content-disposition")

    async def _chunks() -> Any:
        try:
            async for chunk in response.aiter_bytes(1 << 20):
                yield chunk
        finally:
            await response.aclose()
            await client.aclose()

    return StreamingResponse(
        _chunks(),
        media_type=response.headers.get("content-type", "application/zip"),
        headers={"Content-Disposition": disposition} if disposition else None,
    )


@router.get("/tasks/{task_id}/artifact", summary="下载压缩产物 zip（流式）")
async def download_artifact(task_id: str) -> StreamingResponse:
    return await _stream(f"/compressions/{task_id}/download")


@router.get("/tasks/{task_id}/deploy", summary="下载部署包（Dockerfile + serve.py + model_card.json）")
async def download_deploy_bundle(task_id: str) -> StreamingResponse:
    return await _stream(f"/compressions/{task_id}/deploy-export")


# --------------------------------------------------------------------------- #
# 推理端点（压缩完即可推理，前端只需 task_id）
# --------------------------------------------------------------------------- #
@router.post("/infer", summary="用压缩产物推理（CNN/ONNX/OV predict 或 LLM chat）")
async def infer(body: InferRequest) -> Any:
    task = _normalise_task(await _request("GET", f"/compressions/{body.task_id}"))
    if not task["ready"]:
        raise HTTPException(
            status_code=409,
            detail=f"task not ready (status={task['status']}, progress={task['progress']})",
        )
    if body.chat or body.messages:
        return await _request("POST", "/inference/chat/completions", json={
            "task_id": body.task_id,
            "messages": body.messages or [{"role": "user", "content": "Hello"}],
            "max_tokens": body.max_tokens,
            "temperature": body.temperature,
        })
    if body.inputs is None:
        raise HTTPException(status_code=422, detail="provide inputs (CNN/ONNX/OV) or messages (LLM)")
    return await _request("POST", "/inference/predict", json={
        "task_id": body.task_id,
        "inputs": body.inputs,
        "input_shape": body.input_shape,
    })




