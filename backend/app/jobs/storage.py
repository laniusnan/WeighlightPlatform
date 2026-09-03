"""Filesystem layout for model files and compression outputs.

Everything lives under ``settings.MODEL_STORAGE_PATH``::

    models-storage/
      models/<model_uuid>/original<ext>      # uploaded / pulled original model
      tasks/<task_uuid>/...                   # compression outputs (file or dir)
"""
from __future__ import annotations

import uuid
from pathlib import Path

from sqlmodel import Session

from app.core.config import settings


def models_root() -> Path:
    path = Path(settings.MODEL_STORAGE_PATH)
    path.mkdir(parents=True, exist_ok=True)
    return path


def model_dir(model_id: uuid.UUID) -> Path:
    path = models_root() / "models" / str(model_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def original_model_path(model_id: uuid.UUID, extension: str = "") -> Path:
    """Path where the original model file is stored (extension may be empty)."""
    path = model_dir(model_id) / f"original{extension}"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def task_dir(task_id: uuid.UUID) -> Path:
    path = models_root() / "tasks" / str(task_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def dir_size(path: Path) -> int:
    """Total size of a file or directory tree in bytes."""
    if path.is_file():
        return path.stat().st_size
    if path.is_dir():
        return sum(
            p.stat().st_size for p in path.rglob("*") if p.is_file()
        )
    return 0


def enforce_user_quota(session: Session, user_id: object, extra_bytes: int) -> None:
    """Raise if adding ``extra_bytes`` would exceed the user's storage quota.

    ``extra_bytes`` is the size of the file currently being stored; the row it
    belongs to must still have ``size_bytes == 0`` in the DB so the sum below
    does not double-count it.
    """
    from sqlmodel import func, select

    from app.core.config import settings
    from app.models import Dataset, Model

    if settings.USER_STORAGE_QUOTA_BYTES <= 0:
        return
    used = int(
        session.exec(
            select(func.coalesce(func.sum(Model.size_bytes), 0)).where(
                Model.owner_id == user_id
            )
        ).one()
    )
    used += int(
        session.exec(
            select(func.coalesce(func.sum(Dataset.size_bytes), 0)).where(
                Dataset.owner_id == user_id
            )
        ).one()
    )
    limit = int(settings.USER_STORAGE_QUOTA_BYTES)
    if used + int(extra_bytes) > limit:
        raise ValueError(
            f"Storage quota exceeded: used {used} bytes, uploading "
            f"{int(extra_bytes)} bytes, limit is {limit} bytes"
        )


def safe_extension(filename: str) -> str:
    """Return a lowercased, sanitized file extension (with dot) from a filename."""
    suffix = Path(filename or "").suffix.lower()
    if not suffix or len(suffix) > 16:
        return ".model"
    return suffix
