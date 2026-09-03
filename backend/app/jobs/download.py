"""Streaming download helpers (httpx) with hard size limits."""
from __future__ import annotations

import tarfile
import zipfile
from pathlib import Path

import httpx

from app.core.config import settings


def download_file(
    url: str,
    dest: Path,
    max_bytes: int | None = None,
    chunk_size: int = 1024 * 1024,
) -> Path:
    """Stream a file from ``url`` to ``dest``. Raises if it exceeds ``max_bytes``."""
    max_bytes = max_bytes or settings.MAX_MODEL_DOWNLOAD_BYTES
    dest.parent.mkdir(parents=True, exist_ok=True)
    with httpx.stream(
        "GET", url, follow_redirects=True, timeout=httpx.Timeout(60.0)
    ) as response:
        response.raise_for_status()
        total = 0
        with dest.open("wb") as fh:
            for chunk in response.iter_bytes(chunk_size=chunk_size):
                total += len(chunk)
                if total > max_bytes:
                    raise ValueError(
                        f"Download exceeded the {max_bytes} bytes limit"
                    )
                fh.write(chunk)
    return dest


def download_and_extract_archive(
    url: str,
    dest_dir: Path,
    max_bytes: int | None = None,
) -> Path:
    """Download a .zip / .tar(.gz) archive and extract it into ``dest_dir``."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    tmp = dest_dir / "_archive.bin"
    download_file(url, tmp, max_bytes=max_bytes)
    try:
        if zipfile.is_zipfile(tmp):
            with zipfile.ZipFile(tmp) as archive:
                archive.extractall(dest_dir)
        elif tarfile.is_tarfile(tmp):
            with tarfile.open(tmp) as archive:
                archive.extractall(dest_dir, filter="data")
        else:
            raise ValueError(
                "Downloaded evaluation data is not a zip or tar archive"
            )
    finally:
        tmp.unlink(missing_ok=True)
    return dest_dir
