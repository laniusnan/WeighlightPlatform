import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from sqlmodel import col, func, select

from app.api.deps import CurrentUser, SessionDep
from app.core.redis import enqueue_job
from app.jobs.storage import task_dir
from app.models import (
    CompressionMethod,
    CompressionTask,
    CompressionTaskCreate,
    CompressionTaskPublic,
    CompressionTasksPublic,
    Message,
    Model,
    TaskStatus,
)

router = APIRouter(prefix="/compressions", tags=["compressions"])


def _now() -> datetime:
    return datetime.now(UTC)


def _get_user_task(
    session: SessionDep, current_user: CurrentUser, task_id: uuid.UUID
) -> CompressionTask:
    task = session.get(CompressionTask, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Compression task not found")
    if not current_user.is_superuser and (task.owner_id != current_user.id):
        raise HTTPException(status_code=403, detail="Not enough permissions")
    return task


def _to_public(task: CompressionTask) -> CompressionTaskPublic:
    public = CompressionTaskPublic.model_validate(task)
    if task.model is not None:
        public.model_name = task.model.name
    return public


@router.get("/", response_model=CompressionTasksPublic)
def read_compression_tasks(
    session: SessionDep,
    current_user: CurrentUser,
    skip: int = 0,
    limit: int = 100,
    status: TaskStatus | None = None,
) -> Any:
    """
    Retrieve compression tasks (admins see all, regular users see their own).
    """
    if current_user.is_superuser:
        count_statement = select(func.count()).select_from(CompressionTask)
        statement = select(CompressionTask)
    else:
        count_statement = (
            select(func.count())
            .select_from(CompressionTask)
            .where(CompressionTask.owner_id == current_user.id)
        )
        statement = select(CompressionTask).where(
            CompressionTask.owner_id == current_user.id
        )
    if status is not None:
        count_statement = count_statement.where(
            CompressionTask.status == status.value
        )
        statement = statement.where(CompressionTask.status == status.value)
    count = session.exec(count_statement).one()
    tasks = session.exec(
        statement
        .order_by(col(CompressionTask.created_at).desc())
        .offset(skip)
        .limit(limit)
    ).all()
    return CompressionTasksPublic(
        data=[_to_public(task) for task in tasks], count=count
    )


@router.post("/", response_model=CompressionTaskPublic)
async def create_compression_task(
    session: SessionDep,
    current_user: CurrentUser,
    task_in: CompressionTaskCreate,
) -> Any:
    """
    Create a compression task and enqueue it for the worker.
    """
    model = session.get(Model, task_in.model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    if not current_user.is_superuser and (model.owner_id != current_user.id):
        raise HTTPException(status_code=403, detail="Not enough permissions")
    if model.status != "ready":
        raise HTTPException(
            status_code=422,
            detail=f"Model is not ready (status: {model.status})",
        )

    params = task_in.params or {}
    method = task_in.method.value
    if method == CompressionMethod.quantize_onnx.value and model.framework != "onnx":
        raise HTTPException(
            status_code=422,
            detail="quantize_onnx requires an ONNX model",
        )
    if method == CompressionMethod.llm_quantize.value and model.framework != "huggingface":
        raise HTTPException(
            status_code=422,
            detail="llm_quantize requires a HuggingFace model (repo id as source URL)",
        )

    task = CompressionTask(
        model_id=model.id,
        method=method,
        params=params,
        status=TaskStatus.pending.value,
        owner_id=current_user.id,
    )
    session.add(task)
    session.commit()
    session.refresh(task)
    await enqueue_job("run_compression", task.id)
    task.model = model
    return _to_public(task)

@router.get("/{task_id}", response_model=CompressionTaskPublic)
def read_compression_task(
    session: SessionDep, current_user: CurrentUser, task_id: uuid.UUID
) -> Any:
    """
    Get a compression task by ID.
    """
    return _to_public(_get_user_task(session, current_user, task_id))


@router.get("/{task_id}/progress-history")
def read_progress_history(
    session: SessionDep, current_user: CurrentUser, task_id: uuid.UUID
) -> Any:
    """
    Live progress samples published by the worker (falls back to the final
    ``progress_history`` stored in metrics once the task has finished).
    """
    task = _get_user_task(session, current_user, task_id)
    history: list[list[float]] = []
    try:
        import redis

        from app.core.config import settings

        client = redis.Redis.from_url(str(settings.REDIS_URL), decode_responses=True)
        raw = client.lrange(f"platform:progress:{task.id}", 0, -1)
        if raw:
            history = [[round(float(i) * 0.5, 1), float(value)] for i, value in enumerate(raw)]
    except Exception:
        pass
    if not history and task.metrics:
        history = task.metrics.get("progress_history") or []
    return {
        "task_id": str(task.id),
        "status": task.status,
        "progress": task.progress,
        "history": history,
    }


@router.get("/{task_id}/download")
def download_compression_result(
    session: SessionDep, current_user: CurrentUser, task_id: uuid.UUID
) -> FileResponse:
    """
    Download the compressed model (single file, or a zip for directories).
    """
    task = _get_user_task(session, current_user, task_id)
    if task.status != TaskStatus.succeeded.value:
        raise HTTPException(
            status_code=422, detail=f"Task is not finished (status: {task.status})"
        )
    if not task.output_path or not Path(task.output_path).exists():
        raise HTTPException(status_code=404, detail="Result file not found")

    path = Path(task.output_path)
    if path.is_dir():
        zip_path = path.with_suffix(".zip")
        if not zip_path.exists():
            shutil.make_archive(str(path), "zip", path)
        return FileResponse(
            zip_path,
            filename=f"{task.method}_result.zip",
            media_type="application/zip",
        )
    return FileResponse(
        path, filename=path.name, media_type="application/octet-stream"
    )


def _artifact_runtime(artifact: Path) -> str:
    if artifact.is_dir() and (artifact / "model.xml").exists():
        return "openvino>=2025.0 onnxruntime"
    return "onnxruntime"


_SERVE_STUB = """\
import os
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Model Server")

# Load the model once at startup (CPU).
is_ov = os.path.isdir("model") and os.path.exists("model/model.xml")
if is_ov:
    import openvino as ov
    compiled = ov.Core().compile_model("model", "CPU")
    INPUT_NAME = next(iter(compiled.inputs)).any_name
    OUTPUT_KEY = next(iter(compiled.outputs))
    def predict(data):
        return compiled({INPUT_NAME: np.asarray(data, dtype=np.float32)})[OUTPUT_KEY]
else:
    import onnxruntime as ort
    model_file = os.listdir("model")[0]
    session = ort.InferenceSession(
        os.path.join("model", model_file), providers=["CPUExecutionProvider"]
    )
    INPUT_NAME = session.get_inputs()[0].name
    def predict(data):
        return session.run(None, {INPUT_NAME: np.asarray(data, dtype=np.float32)})[0]


class Req(BaseModel):
    inputs: list
    input_shape: list[int] | None = None


@app.post("/predict")
def predict_route(req: Req):
    data = req.inputs
    if req.input_shape:
        data = np.asarray(data).reshape([int(x) for x in req.input_shape])
    return {"output": np.asarray(predict(data)).tolist()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
"""


@router.get("/{task_id}/deploy-export")
def export_deployment_package(
    session: SessionDep, current_user: CurrentUser, task_id: uuid.UUID
) -> FileResponse:
    """
    Package the compressed artifact with a Dockerfile + model card + a minimal
    FastAPI server (serve.py) into a deployable zip.
    """
    import json

    task = _get_user_task(session, current_user, task_id)
    if task.status != TaskStatus.succeeded.value:
        raise HTTPException(
            status_code=422, detail=f"Task is not finished (status: {task.status})"
        )
    if not task.output_path or not Path(task.output_path).exists():
        raise HTTPException(status_code=404, detail="Result file not found")

    task.model = session.get(Model, task.model_id)
    model_name = task.model.name if task.model else str(task.model_id)

    artifact = Path(task.output_path)
    pkg = task_dir(task.id) / "deploy"
    pkg.mkdir(parents=True, exist_ok=True)
    for item in pkg.iterdir():
        if item.is_dir():
            shutil.rmtree(item, ignore_errors=True)
        else:
            item.unlink(missing_ok=True)

    model_dir = pkg / "model"
    model_dir.mkdir(exist_ok=True)
    if artifact.is_dir():
        shutil.copytree(artifact, model_dir, dirs_exist_ok=True)
    else:
        shutil.copy2(artifact, model_dir / artifact.name)

    card = {
        "model_id": str(task.model_id),
        "model_name": model_name,
        "method": task.method,
        "params": task.params,
        "metrics": task.metrics,
        "created_at": task.created_at.isoformat() if task.created_at else None,
    }
    (pkg / "model_card.json").write_text(
        json.dumps(card, ensure_ascii=False, indent=2)
    )
    runtime = _artifact_runtime(artifact)
    (pkg / "Dockerfile").write_text(
        "FROM python:3.11-slim\n"
        "WORKDIR /app\n"
        "COPY model ./model\n"
        f"RUN pip install --no-cache-dir fastapi uvicorn numpy {runtime}\n"
        "COPY serve.py ./serve.py\n"
        "EXPOSE 8000\n"
        'CMD ["python", "serve.py"]\n'
    )
    (pkg / "serve.py").write_text(_SERVE_STUB)
    (pkg / "README.md").write_text(
        f"# Deployment package\n\n"
        f"Model: {model_name}\nMethod: {task.method}\n\n"
        "Build: `docker build -t model-server .`\n"
        "Run: `docker run -p 8000:8000 model-server`\n"
        "Predict: `POST /predict` with {\"inputs\": [...], \"input_shape\": [...]}\n"
    )
    zip_path = task_dir(task.id) / "deploy_package.zip"
    shutil.make_archive(str(zip_path.with_suffix("")), "zip", pkg)
    return FileResponse(
        zip_path,
        filename=f"deploy_{model_name}.zip",
        media_type="application/zip",
    )


@router.post("/{task_id}/retry", response_model=CompressionTaskPublic)
async def retry_compression_task(
    session: SessionDep, current_user: CurrentUser, task_id: uuid.UUID
) -> Any:
    """
    Re-enqueue a failed or canceled task.
    """
    task = _get_user_task(session, current_user, task_id)
    if task.status not in (TaskStatus.failed.value, TaskStatus.canceled.value):
        raise HTTPException(
            status_code=422,
            detail=f"Only failed or canceled tasks can be retried (status: {task.status})",
        )
    task.status = TaskStatus.pending.value
    task.progress = 0.0
    task.error_message = None
    task.metrics = {}
    task.finished_at = None
    session.add(task)
    session.commit()
    session.refresh(task)
    await enqueue_job("run_compression", task.id)
    task.model = session.get(Model, task.model_id)
    return _to_public(task)


@router.delete("/{task_id}", response_model=Message)
def delete_compression_task(
    session: SessionDep, current_user: CurrentUser, task_id: uuid.UUID
) -> Any:
    """
    Delete a task. Pending/running tasks are marked as canceled first.
    """
    task = _get_user_task(session, current_user, task_id)
    if task.status in (TaskStatus.pending.value, TaskStatus.running.value):
        task.status = TaskStatus.canceled.value
        task.finished_at = _now()
        session.add(task)
        session.commit()
    dir_path = task_dir(task_id)
    if dir_path.exists():
        shutil.rmtree(dir_path, ignore_errors=True)
    session.delete(task)
    session.commit()
    return Message(message="Compression task deleted successfully")

