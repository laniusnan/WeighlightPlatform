"""Dataset management: upload / list / delete calibration & evaluation datasets."""
import shutil
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from sqlmodel import col, func, select

from app.api.deps import CurrentUser, SessionDep
from app.jobs.storage import enforce_user_quota
from app.models import Dataset, DatasetPublic, DatasetsPublic, Message

router = APIRouter(prefix="/datasets", tags=["datasets"])

_MAX_UPLOAD_BYTES = 5 * 1024 * 1024 * 1024  # 5 GB


def _dataset_root() -> Path:
    from app.jobs.storage import models_root

    path = models_root() / "datasets"
    path.mkdir(parents=True, exist_ok=True)
    return path


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
    dataset = _get_user_dataset(session, current_user, dataset_id)
    dir_path = Path(dataset.file_path) if dataset.file_path else None
    if dir_path and dir_path.exists():
        shutil.rmtree(dir_path, ignore_errors=True)
    session.delete(dataset)
    session.commit()
    return Message(message="Dataset deleted successfully")
