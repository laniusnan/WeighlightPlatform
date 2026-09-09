import shutil
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlmodel import col, func, select

from app.api.deps import CurrentUser, SessionDep
from app.core.config import settings
from app.core.redis import enqueue_job
from app.jobs.storage import (
    dir_size,
    enforce_user_quota,
    model_dir,
    original_model_path,
    safe_extension,
)
from app.models import (
    Message,
    Model,
    ModelCreate,
    ModelFramework,
    ModelPublic,
    ModelSourceType,
    ModelsPublic,
    ModelStatus,
)

router = APIRouter(prefix="/models", tags=["models"])

_MAX_UPLOAD_BYTES = 10 * 1024 * 1024 * 1024  # 10 GB


def _get_user_model(
    session: SessionDep, current_user: CurrentUser, model_id: uuid.UUID
) -> Model:
    model = session.get(Model, model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    if not current_user.is_superuser and (model.owner_id != current_user.id):
        raise HTTPException(status_code=403, detail="Not enough permissions")
    return model


@router.get("/", response_model=ModelsPublic)
def read_models(
    session: SessionDep,
    current_user: CurrentUser,
    skip: int = 0,
    limit: int = 100,
    status: ModelStatus | None = None,
) -> Any:
    """
    Retrieve models (admins see all, regular users see their own).
    """
    if current_user.is_superuser:
        count_statement = select(func.count()).select_from(Model)
        statement = select(Model)
    else:
        count_statement = (
            select(func.count())
            .select_from(Model)
            .where(Model.owner_id == current_user.id)
        )
        statement = select(Model).where(Model.owner_id == current_user.id)
    if status is not None:
        count_statement = count_statement.where(Model.status == status.value)
        statement = statement.where(Model.status == status.value)
    count = session.exec(count_statement).one()
    models = session.exec(
        statement.order_by(col(Model.created_at).desc()).offset(skip).limit(limit)
    ).all()
    return ModelsPublic(
        data=[ModelPublic.model_validate(model) for model in models], count=count
    )


@router.post("/upload", response_model=ModelPublic)
async def upload_model(
    session: SessionDep,
    current_user: CurrentUser,
    file: UploadFile = File(...),
    name: str = Form(...),
    framework: ModelFramework = Form(ModelFramework.pytorch),
    description: str | None = Form(default=None),
    arch_file: UploadFile | None = File(default=None),
) -> Any:
    """
    Upload a model file directly.

    - A plain model file is stored as-is.
    - ONNX models exported with torch>=2.9 split their weights into a sidecar
      ``*.onnx.data`` file; upload a zip containing both files and the platform
      inlines everything into a single self-contained ``original.onnx``.
    - PyTorch checkpoints saved with ``torch.save(model)`` reference the model
      class; upload the matching ``arch.py`` (the model class definition) so the
      worker can load them (also required to rebuild ``state_dict`` checkpoints).
    """
    model = Model(
        name=name,
        framework=framework.value,
        description=description,
        source_type=ModelSourceType.upload.value,
        status=ModelStatus.pending.value,
        owner_id=current_user.id,
    )
    # Model registry: same-name uploads become the next version.
    existing = session.exec(
        select(Model)
        .where(Model.name == name, Model.owner_id == current_user.id)
        .order_by(col(Model.version).desc())
        .limit(1)
    ).first()
    if existing:
        model.version = existing.version + 1
    session.add(model)
    session.commit()
    session.refresh(model)

    try:
        if _is_onnx_archive(file):
            dest = await _store_onnx_archive(file, model.id)
        else:
            extension = safe_extension(file.filename or "")
            dest = await _store_upload(file, original_model_path(model.id, extension))
        if arch_file is not None:
            await _store_arch_file(arch_file, model.id)
        # Enforce the per-user storage quota (rows still have size_bytes=0 here).
        try:
            enforce_user_quota(session, current_user.id, dest.stat().st_size)
        except ValueError as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc
    except Exception:
        import shutil

        shutil.rmtree(model_dir(model.id), ignore_errors=True)
        session.delete(model)
        session.commit()
        raise

    model.file_path = str(dest)
    model.size_bytes = dest.stat().st_size
    model.status = ModelStatus.ready.value
    session.add(model)
    session.commit()
    session.refresh(model)
    return model

async def _store_upload(file: UploadFile, dest: Path) -> Path:
    """Stream an uploaded file to ``dest``, enforcing the size limit."""
    total = 0
    with dest.open("wb") as fh:
        while chunk := await file.read(1024 * 1024):
            total += len(chunk)
            if total > _MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"Upload exceeds the {_MAX_UPLOAD_BYTES} bytes limit",
                )
            fh.write(chunk)

    # Detect ONNX files that reference missing external data files and tell the
    # user to upload a zip instead (see _store_onnx_archive). Also validates the
    # ONNX graph structure (basic security check against malformed models).
    if dest.suffix.lower() == ".onnx":
        import onnx

        try:
            proto = onnx.load(str(dest), load_external_data=False)
            onnx.checker.check_model(proto)
            has_external = any(
                tensor.HasField("data_location")
                and tensor.data_location == onnx.TensorProto.EXTERNAL
                for tensor in proto.graph.initializer
            )
            if has_external:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "This ONNX model stores its weights in external data "
                        "files. Upload a zip containing the .onnx file together "
                        "with its .onnx.data file(s)."
                    ),
                )
        except HTTPException:
            raise
        except Exception:
            pass  # let downstream jobs report loading issues
    elif dest.suffix.lower() in {".pt", ".pth"}:
        import zipfile

        if not zipfile.is_zipfile(dest):
            raise HTTPException(
                status_code=422,
                detail=(
                    "Not a valid PyTorch checkpoint: expected a torch.save() "
                    "output (a zip container)."
                ),
            )
    return dest


def _is_onnx_archive(file: UploadFile) -> bool:
    """True when the upload is a zip archive (used for multi-file ONNX models)."""
    return (file.filename or "").lower().endswith(".zip")


async def _store_onnx_archive(file: UploadFile, model_id: uuid.UUID) -> Path:
    """Extract an ONNX model from a zip and store it inlined into one file.

    torch>=2.9 exports may split the weights into a sidecar ``*.onnx.data``
    file; this inlines everything back into a single self-contained
    ``original.onnx`` so downstream jobs and downloads work with one file.
    """
    import shutil
    import tempfile
    import zipfile

    import onnx

    staging = Path(tempfile.mkdtemp(prefix="onnx_archive_"))
    try:
        zip_path = staging / "model.zip"
        total = 0
        with zip_path.open("wb") as fh:
            while chunk := await file.read(1024 * 1024):
                total += len(chunk)
                if total > _MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Upload exceeds the {_MAX_UPLOAD_BYTES} bytes limit",
                    )
                fh.write(chunk)

        extract_dir = staging / "extracted"
        with zipfile.ZipFile(zip_path) as zf:
            for member in zf.infolist():
                target = (extract_dir / member.filename).resolve()
                if extract_dir.resolve() not in target.parents:
                    raise HTTPException(
                        status_code=400, detail="Unsafe path in archive"
                    )
            zf.extractall(extract_dir)

        onnx_files = sorted(extract_dir.rglob("*.onnx"))
        if not onnx_files:
            raise HTTPException(
                status_code=400,
                detail="No .onnx file found in the archive",
            )
        # External-data models have a small .onnx plus a large .data file; the
        # one with the largest size has the weights inline.
        main = max(onnx_files, key=lambda p: p.stat().st_size)

        dest = original_model_path(model_id, ".onnx")
        proto = onnx.load(str(main), load_external_data=True)
        onnx.save_model(proto, str(dest), save_as_external_data=False)
        return dest
    finally:
        shutil.rmtree(staging, ignore_errors=True)


async def _store_arch_file(arch_file: UploadFile, model_id: uuid.UUID) -> None:
    """Store the optional companion ``arch.py`` next to the model checkpoint."""
    if not (arch_file.filename or "").lower().endswith(".py"):
        raise HTTPException(
            status_code=422, detail="Architecture file must be a .py file"
        )
    dest = model_dir(model_id) / "arch.py"
    total = 0
    with dest.open("wb") as fh:
        while chunk := await arch_file.read(1024 * 1024):
            total += len(chunk)
            if total > 1024 * 1024:
                raise HTTPException(
                    status_code=413,
                    detail="Architecture file too large (max 1 MB)",
                )
            fh.write(chunk)


@router.post("/pull", response_model=ModelPublic)
async def pull_model_from_url(
    session: SessionDep,
    current_user: CurrentUser,
    model_in: ModelCreate,
) -> Any:
    """
    Create a model from a URL (or a HuggingFace repo id) and download it in the
    background worker.
    """
    source_url = (model_in.source_url or "").strip()
    if not source_url:
        raise HTTPException(status_code=422, detail="source_url is required")

    model = Model(
        name=model_in.name,
        framework=model_in.framework.value,
        description=model_in.description,
        source_type=ModelSourceType.url.value,
        source_url=source_url,
        status=ModelStatus.pending.value,
        owner_id=current_user.id,
    )
    session.add(model)
    session.commit()
    session.refresh(model)
    await enqueue_job("pull_model", model.id)
    return model


class ImportLocalModelRequest(BaseModel):
    """Import a HF-format model directory that lives on the server host."""

    name: str | None = None
    # Path *inside the container* under one of the read-only local roots,
    # e.g. /models-local/Qwen2.5-0.5B-Instruct
    server_path: str
    framework: ModelFramework = ModelFramework.huggingface
    description: str | None = None


@router.post("/import-local", response_model=ModelPublic)
async def import_local_model(
    session: SessionDep,
    current_user: CurrentUser,
    body: ImportLocalModelRequest,
) -> Any:
    """
    Import a HF-format model directory that is already present on the server.

    The host directory is mounted read-only into the container (see
    ``LOCAL_MODEL_DIRS`` / ``compose.cpu.yml``). The platform copies it into
    its own storage and registers a ready ``Model`` row with
    ``source_type=local`` so methods such as ``llm_quantize`` (engine=nncf)
    can run on it directly.
    """
    allowed = [Path(p.strip()) for p in settings.LOCAL_MODEL_DIRS.split(";") if p.strip()]
    if not allowed:
        raise HTTPException(status_code=422, detail="LOCAL_MODEL_DIRS is empty")
    raw = Path(body.server_path).expanduser().resolve()
    if not any(raw.is_relative_to(root.resolve()) for root in allowed):
        roots = ", ".join(str(r) for r in allowed)
        raise HTTPException(
            status_code=422,
            detail=f"server_path must be inside one of: {roots}",
        )
    if not raw.is_dir():
        raise HTTPException(status_code=404, detail="server_path is not a directory")

    # Auto-descend one level when the repo root wraps a single HF directory.
    source = raw
    if not (source / "config.json").is_file():
        children = [p for p in source.iterdir() if p.is_dir()]
        hf_dirs = [p for p in children if (p / "config.json").is_file()]
        if len(hf_dirs) == 1:
            source = hf_dirs[0]
    if not (source / "config.json").is_file():
        raise HTTPException(
            status_code=422, detail="Not a HuggingFace model dir: config.json missing"
        )
    weights = [
        (source / "model.safetensors").is_file(),
        (source / "pytorch_model.bin").is_file(),
        (source / "model.bin").is_file(),
    ]
    if not any(weights):
        raise HTTPException(
            status_code=422,
            detail="No weights found (model.safetensors / pytorch_model.bin / model.bin)",
        )

    try:
        enforce_user_quota(session, current_user.id, dir_size(source))
    except ValueError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc

    default_name = source.name or "local-model"
    model = Model(
        name=body.name or default_name,
        framework=body.framework.value,
        description=body.description,
        source_type=ModelSourceType.local.value,
        source_url=None,
        status=ModelStatus.ready.value,
        owner_id=current_user.id,
    )
    existing = session.exec(
        select(Model)
        .where(Model.name == model.name, Model.owner_id == current_user.id)
        .order_by(col(Model.version).desc())
        .limit(1)
    ).first()
    if existing:
        model.version = existing.version + 1
    session.add(model)
    session.commit()
    session.refresh(model)

    dest = model_dir(model.id)
    try:
        shutil.copytree(source, dest, dirs_exist_ok=True)
        model.file_path = str(dest)
        model.size_bytes = dir_size(dest)
        model.status = ModelStatus.ready.value
        session.add(model)
        session.commit()
        session.refresh(model)
    except Exception:
        shutil.rmtree(dest, ignore_errors=True)
        session.delete(model)
        session.commit()
        raise
    return model


def _local_model_roots() -> list[Path]:
    return [
        Path(p.strip()).resolve()
        for p in settings.LOCAL_MODEL_DIRS.split(";")
        if p.strip()
    ]


@router.get("/local-dirs")
async def list_local_model_dirs() -> dict[str, Any]:
    """List candidate HF model directories under the read-only local roots.

    Lets the “import local model” dialog offer choices (each one-level child
    directory of a configured root) while still allowing a free-form path.
    """
    roots = _local_model_roots()
    dirs: list[dict[str, Any]] = []
    for root in roots:
        if not root.is_dir():
            continue
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            dirs.append(
                {
                    "name": child.name,
                    "path": str(child),
                    "has_config": (child / "config.json").is_file(),
                    "weight_files": sum(
                        1
                        for name in ("model.safetensors", "pytorch_model.bin", "model.bin")
                        if (child / name).is_file()
                    ),
                }
            )
    return {"roots": [str(r) for r in roots], "dirs": dirs}


@router.get("/{model_id}", response_model=ModelPublic)
def read_model(
    session: SessionDep, current_user: CurrentUser, model_id: uuid.UUID
) -> Any:
    """
    Get a model by ID.
    """
    return _get_user_model(session, current_user, model_id)


@router.get("/{model_id}/download")
def download_model(
    session: SessionDep, current_user: CurrentUser, model_id: uuid.UUID
) -> FileResponse:
    """
    Download the original model file.
    """
    model = _get_user_model(session, current_user, model_id)
    if not model.file_path or not Path(model.file_path).exists():
        raise HTTPException(status_code=404, detail="Model file not found")
    return FileResponse(
        model.file_path,
        filename=f"{model.name}{Path(model.file_path).suffix}",
        media_type="application/octet-stream",
    )


@router.delete("/{model_id}", response_model=Message)
def delete_model(
    session: SessionDep, current_user: CurrentUser, model_id: uuid.UUID
) -> Any:
    """
    Delete a model and all of its stored files.
    """
    model = _get_user_model(session, current_user, model_id)
    dir_path = model_dir(model_id)
    if dir_path.exists():
        import shutil

        shutil.rmtree(dir_path, ignore_errors=True)
    session.delete(model)
    session.commit()
    return Message(message="Model deleted successfully")

