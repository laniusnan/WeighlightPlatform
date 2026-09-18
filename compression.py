#!/usr/bin/env python3
"""Single-file launcher so the CLI can be run as a plain script.

    python compression.py --help
    python compression.py quantize --model Qwen2.5-0.5B --method nncf_int4 --bits 4 --watch
    python compression.py task status <task_id> --json

It simply locates ``backend/`` (next to this file, or ``/app/backend`` inside
the platform image) and delegates to :func:`app.cli.compression.main`, so the
CLI needs no installation, no ``-m`` package form and no ``PYTHONPATH``.

Equivalent (all supported):

    python compression.py            <sub> [--flags]     # this launcher
    python backend/app/cli/compression.py <sub> [--flags]
    python -m app.cli.compression    <sub> [--flags]     # from backend/
    compression                      <sub> [--flags]     # after `uv sync` (console script)
    scripts/compression              <sub> [--flags]     # bash wrapper
"""
from __future__ import annotations

import pathlib
import sys


def _drop_self_dir() -> None:
    """Remove this file's own directory from ``sys.path``.

    The file is named ``compression.py``; when executed as a script Python puts
    its directory first on ``sys.path``, which would **shadow the stdlib
    ``compression`` package** that ``bz2`` -> ``shutil`` -> ``tempfile`` ->
    ``urllib`` import. Dropping the directory keeps the stdlib importable (this
    file is already loaded as ``__main__``), so plain ``python compression.py``
    works exactly like the package form.
    """
    here = pathlib.Path(__file__).resolve().parent
    sys.path[:] = [p for p in sys.path if pathlib.Path(p or ".").resolve() != here]


_drop_self_dir()


def _find_backend() -> pathlib.Path:
    here = pathlib.Path(__file__).resolve().parent
    candidates = [
        here / "backend",              # repo root: ./compression.py
        here,                          # already inside backend/
        pathlib.Path("/app/backend"),  # platform container
    ]
    for candidate in candidates:
        if (candidate / "app" / "cli" / "compression.py").is_file():
            return candidate
    raise SystemExit(
        "cannot locate backend/app/cli/compression.py - run this script from the "
        "repository root or keep it next to the backend/ directory"
    )


def main() -> int:
    backend = _find_backend()
    if str(backend) not in sys.path:
        sys.path.insert(0, str(backend))
    sys.argv[0] = "compression"
    from app.cli.compression import main as cli_main

    return cli_main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
