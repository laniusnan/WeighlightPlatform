"""Dataset management: upload / list / delete calibration & evaluation datasets."""
import os
import shutil
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from sqlmodel import col, func, select

from app.api.deps import CurrentUser, SessionDep
from app.core.config import settings
from app.jobs.storage import dir_size, enforce_user_quota
from app.models import Dataset, DatasetPublic, DatasetsPublic, Message

router = APIRouter(prefix="/datasets", tags=["datasets"])

_MAX_UPLOAD_BYTES = 5 * 1024 * 1024 * 1024  # 5 GB

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_TEXT_SUFFIXES = {
    ".txt",
    ".csv",
    ".tsv",
    ".json",
    ".jsonl",
    ".parquet",
    ".xlsx",
    ".xls",
    ".arrow",
}
_ARCHIVE_SUFFIXES = {".zip", ".tar", ".gz", ".tgz", ".bz2", ".xz", ".7z"}
_SCAN_MAX_FILES = 400_000


def _dataset_root() -> Path:
    from app.jobs.storage import models_root

    path = models_root() / "datasets"
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Server-local datasets (open datasets that already live on the host and are
# mounted read-only into the containers, e.g.
# /home/zhangxuefei/Dataset -> /datasets-local).
# ---------------------------------------------------------------------------
def _local_dataset_roots() -> list[Path]:
    return [
        Path(p.strip()).resolve()
        for p in settings.LOCAL_DATASET_DIRS.split(";")
        if p.strip()
    ]


def _is_under_local_roots(path: Path) -> bool:
    """True when ``path`` points inside one of the read-only dataset roots."""
    try:
        resolved = path.resolve()
    except OSError:  # pragma: no cover - broken symlink
        return False
    return any(resolved.is_relative_to(root) for root in _local_dataset_roots())


def _scan_tree(path: Path) -> tuple[int, int, bool]:
    """Return ``(size_bytes, file_count, truncated)`` for a directory tree."""
    total = 0
    files = 0
    for root, _dirs, names in os.walk(path):
        for name in names:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:  # pragma: no cover - race with deletion
                continue
            files += 1
            if files >= _SCAN_MAX_FILES:
                return total, files, True
    return total, files, False


def _describe_local_dataset(path: Path) -> dict[str, Any]:
    """Classify a server-local directory into something the platform can use.

    ``kind`` is one of:

    * ``imagefolder`` - per-class image folders, as expected by data-driven
      pruning, QAT, distillation and classification evaluation;
    * ``text`` - instruction/preference corpora (jsonl/csv/txt/...) for SFT,
      DPO and perplexity runs;
    * ``hf`` - a HuggingFace dataset directory (``config.json``);
    * ``archive`` - only raw archives are present (extract them first);
    * ``unknown`` - nothing recognisable.
    """
    size_bytes, files, truncated = _scan_tree(path)
    subdirs = sorted((p for p in path.iterdir() if p.is_dir()), key=lambda p: p.name)
    top_files = sorted((p for p in path.iterdir() if p.is_file()), key=lambda p: p.name)

    class_names: list[str] = []
    samples = 0
    for child in subdirs:
        count = 0
        for entry in child.iterdir():
            if entry.is_file() and entry.suffix.lower() in _IMAGE_SUFFIXES:
                count += 1
        if count:
            class_names.append(child.name)
            samples += count

    text_files = [f for f in top_files if f.suffix.lower() in _TEXT_SUFFIXES]
    archive_files = [f for f in top_files if f.suffix.lower() in _ARCHIVE_SUFFIXES]

    if class_names:
        kind = "imagefolder"
    elif (path / "config.json").is_file():
        kind = "hf"
    elif text_files:
        kind = "text"
    elif archive_files:
        kind = "archive"
    else:
        kind = "unknown"

    return {
        "name": path.name,
        "path": str(path),
        "kind": kind,
        "classes": len(class_names),
        "class_names": class_names[:12],
        "samples": samples,
        "text_files": [f.name for f in text_files[:8]],
        "size_bytes": size_bytes,
        "files": files,
        "truncated": truncated,
    }


class ImportLocalDatasetRequest(BaseModel):
    """Register a dataset that already lives on the server host."""

    name: str | None = None
    # Path *inside the container* under one of LOCAL_DATASET_DIRS, e.g.
    # /datasets-local/CIFAR10-train
    server_path: str
    # reference: keep the bytes where they are (no copy - ideal for read-only
    #            open datasets such as CIFAR/MNIST);
    # copy:      import into platform storage (counts against the user quota).
    mode: Literal["reference", "copy"] = "reference"
    description: str | None = None


def _get_user_dataset(
    session: SessionDep, current_user: CurrentUser, dataset_id: object
) -> Dataset:
    dataset = session.get(Dataset, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if not current_user.is_superuser and (dataset.owner_id != current_user.id):
        raise HTTPException(status_code=403, detail="Not enough permissions")
    return dataset


@router.get("/", response_model=DatasetsPublic)
def read_datasets(
    session: SessionDep,
    current_user: CurrentUser,
    skip: int = 0,
    limit: int = 100,
) -> Any:
    """List datasets (admins see all, regular users see their own)."""
    if current_user.is_superuser:
        count_statement = select(func.count()).select_from(Dataset)
        statement = select(Dataset)
    else:
        count_statement = (
            select(func.count())
            .select_from(Dataset)
            .where(Dataset.owner_id == current_user.id)
        )
        statement = select(Dataset).where(Dataset.owner_id == current_user.id)
    count = session.exec(count_statement).one()
    datasets = session.exec(
        statement.order_by(col(Dataset.created_at).desc()).offset(skip).limit(limit)
    ).all()
    return DatasetsPublic(
        data=[DatasetPublic.model_validate(d) for d in datasets], count=count
    )


@router.get("/local-dirs")
def list_local_dataset_dirs(session: SessionDep, current_user: CurrentUser) -> Any:
    """List candidate datasets under the read-only server-local roots.

    Powers the “import dataset from server” dialog: every one-level child
    directory of a configured root is described (kind / classes / samples /
    size) so operators can pick CIFAR-10, MNIST, a wikitext dump, ... without
    typing paths. Already-registered entries are flagged with their dataset id.
    """
    roots = _local_dataset_roots()
    registered: dict[str, Dataset] = {}
    if roots:
        for dataset in session.exec(select(Dataset)).all():
            key = (dataset.source_url or "").rstrip("/")
            if key:
                registered[key] = dataset

    entries: list[dict[str, Any]] = []
    for root in roots:
        if not root.is_dir():
            continue
        for child in sorted(root.iterdir(), key=lambda p: p.name):
            if not child.is_dir() or child.name.startswith((".", "_")):
                continue
            info = _describe_local_dataset(child)
            existing = registered.get(str(child).rstrip("/"))
            info["imported"] = existing is not None
            info["dataset_id"] = str(existing.id) if existing else None
            entries.append(info)
    return {"roots": [str(r) for r in roots], "datasets": entries}


@router.post("/import-local", response_model=DatasetPublic)
def import_local_dataset(
    session: SessionDep,
    current_user: CurrentUser,
    body: ImportLocalDatasetRequest,
) -> Any:
    """Import a dataset that is already on the server (no upload needed).

    ``mode="reference"`` (default) stores the *container path* in
    ``Dataset.file_path`` without copying anything: the API and the worker
    mount the same read-only root, so pruning / QAT / distillation /
    calibration / perplexity jobs can consume it directly while the bytes keep
    living on the host (0 extra disk usage, and deleting the row never touches
    the host data). ``mode="copy"`` snapshots it into platform storage and
    counts it against the per-user quota.
    """
    roots = _local_dataset_roots()
    if not roots:
        raise HTTPException(status_code=422, detail="LOCAL_DATASET_DIRS is empty")

    source = Path(body.server_path).expanduser()
    if not source.is_absolute():
        raise HTTPException(status_code=422, detail="server_path must be absolute")
    resolved = source.resolve()
    if not any(resolved.is_relative_to(root) for root in roots):
        allowed = ", ".join(str(r) for r in roots)
        raise HTTPException(
            status_code=422, detail=f"server_path must be inside one of: {allowed}"
        )
    if not resolved.is_dir():
        raise HTTPException(status_code=404, detail="server_path is not a directory")

    info = _describe_local_dataset(resolved)
    if info["kind"] == "unknown":
        raise HTTPException(
            status_code=422,
            detail=(
                "Unrecognised dataset layout: expected per-class image folders, "
                "a text/JSONL corpus, or a HuggingFace dataset directory"
            ),
        )

    # One row per server path keeps the scanner's ``imported`` flag unambiguous.
    already = session.exec(
        select(Dataset).where(Dataset.source_url == str(resolved))
    ).first()
    if already:
        raise HTTPException(
            status_code=409,
            detail=f"Already imported as dataset {already.id} ({already.name})",
        )

    dataset = Dataset(
        name=(body.name.strip() if body.name and body.name.strip() else resolved.name),
        description=body.description,
        source_type="local",
        source_url=str(resolved),
        status="ready",
        owner_id=current_user.id,
    )
    session.add(dataset)
    session.commit()
    session.refresh(dataset)

    if body.mode == "copy":
        try:
            enforce_user_quota(session, current_user.id, info["size_bytes"])
        except ValueError as exc:
            session.delete(dataset)
            session.commit()
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        dest = _dataset_root() / str(dataset.id)
        try:
            shutil.copytree(resolved, dest, dirs_exist_ok=True)
        except Exception:
            shutil.rmtree(dest, ignore_errors=True)
            session.delete(dataset)
            session.commit()
            raise
        dataset.file_path = str(dest)
        dataset.size_bytes = dir_size(dest)
    else:
        # Reference mode: point at the read-only mount, no bytes are copied.
        dataset.file_path = str(resolved)
        dataset.size_bytes = info["size_bytes"]

    session.add(dataset)
    session.commit()
    session.refresh(dataset)
    return dataset


@router.post("/upload", response_model=DatasetPublic)
async def upload_dataset(
    session: SessionDep,
    current_user: CurrentUser,
    file: UploadFile = File(...),
    name: str = Form(...),
    description: str | None = Form(default=None),
) -> Any:
    """Upload an archive (zip/tar) that will be extracted and stored on disk.

    Datasets can be referenced by ``dataset_id`` from compression tasks for
    calibration or evaluation, avoiding the need to host a public URL.
    """
    import zipfile

    dataset = Dataset(
        name=name,
        description=description,
        source_type="upload",
        status="ready",
        owner_id=current_user.id,
    )
    session.add(dataset)
    session.commit()
    session.refresh(dataset)

    dest_dir = _dataset_root() / str(dataset.id)
    dest_dir.mkdir(parents=True, exist_ok=True)
    archive_path = dest_dir / "_upload.zip"
    total = 0
    try:
        with archive_path.open("wb") as fh:
            while chunk := await file.read(1024 * 1024):
                total += len(chunk)
                if total > _MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Upload exceeds the {_MAX_UPLOAD_BYTES} bytes limit",
                    )
                fh.write(chunk)
        if zipfile.is_zipfile(archive_path):
            with zipfile.ZipFile(archive_path) as archive:
                archive.extractall(dest_dir)
        elif archive_path.suffix.lower() in {".tar", ".tar.gz", ".tgz"}:
            import tarfile

            with tarfile.open(archive_path) as archive:
                archive.extractall(dest_dir, filter="data")
        else:
            # Plain file (e.g. a text corpus) - keep as-is.
            plain = dest_dir / (file.filename or "data.bin")
            shutil.move(str(archive_path), str(plain))
        archive_path.unlink(missing_ok=True)
        # Enforce the per-user storage quota (row still has size_bytes=0 here).
        try:
            enforce_user_quota(session, current_user.id, total)
        except ValueError as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc
    except Exception:
        shutil.rmtree(dest_dir, ignore_errors=True)
        session.delete(dataset)
        session.commit()
        raise

    dataset.file_path = str(dest_dir)
    dataset.size_bytes = total
    session.add(dataset)
    session.commit()
    session.refresh(dataset)
    return dataset


@router.delete("/{dataset_id}", response_model=Message)
def delete_dataset(
    session: SessionDep, current_user: CurrentUser, dataset_id: object
) -> Any:
    """Unregister a dataset.

    Server-local datasets (``source_type="local"``, backed by the read-only
    mount) are only removed from the registry: the files on the host are never
    touched. Uploaded/copied datasets have their platform copy deleted.
    """
    dataset = _get_user_dataset(session, current_user, dataset_id)
    dir_path = Path(dataset.file_path) if dataset.file_path else None
    keep_files = dataset.source_type == "local" or (
        dir_path is not None and _is_under_local_roots(dir_path)
    )
    if dir_path and dir_path.exists() and not keep_files:
        shutil.rmtree(dir_path, ignore_errors=True)
    session.delete(dataset)
    session.commit()
    return Message(
        message=(
            "Dataset unregistered (server-local files were kept on disk)"
            if keep_files
            else "Dataset deleted successfully"
        )
    )
