"""``models`` / ``datasets`` inventory helpers (used to resolve refs by name)."""
from __future__ import annotations

import argparse
from typing import Any

from . import common


def register(subparsers: Any) -> None:
    models = subparsers.add_parser("models", help="list registered models")
    models.add_argument("--framework", help="filter by framework (pytorch/onnx/huggingface/tensorflow)")
    models.add_argument("--limit", type=int, default=50)
    common.add_global_flags(models)
    models.set_defaults(handler=_models)

    datasets = subparsers.add_parser("datasets", help="list datasets (incl. server-local open datasets)")
    datasets.add_argument("--limit", type=int, default=100)
    common.add_global_flags(datasets)
    datasets.set_defaults(handler=_datasets)

    local = subparsers.add_parser("local-datasets", help="scan the read-only server-local dataset roots")
    common.add_global_flags(local)
    local.set_defaults(handler=_local_datasets)


def _models(args: argparse.Namespace) -> dict[str, Any]:
    client = common.ApiClient()
    printer = common.Printer(getattr(args, "json", False))
    page = client.get("/api/v1/models/", params={"limit": common.resolve_int("limit", args.limit, 50)})
    rows = page.get("data", []) if isinstance(page, dict) else page
    if args.framework:
        rows = [m for m in rows if m.get("framework") == args.framework]
    payload = {
        "count": len(rows),
        "models": [
            {
                "model_id": m.get("id"),
                "name": m.get("name"),
                "framework": m.get("framework"),
                "status": m.get("status"),
                "size_bytes": m.get("size_bytes"),
                "version": m.get("version"),
            }
            for m in rows
        ],
    }

    def table(_: Any) -> str:
        return common.render_table(
            [
                (
                    m["model_id"][:8],
                    m["name"],
                    m["framework"],
                    m["status"],
                    common.human_bytes(m["size_bytes"]),
                    f"v{m['version']}",
                )
                for m in payload["models"]
            ],
            ("MODEL", "NAME", "FRAMEWORK", "STATUS", "SIZE", "VER"),
        )

    printer.emit(payload, table)
    return payload


def _datasets(args: argparse.Namespace) -> dict[str, Any]:
    client = common.ApiClient()
    printer = common.Printer(getattr(args, "json", False))
    page = client.get("/api/v1/datasets/", params={"limit": common.resolve_int("limit", args.limit, 100)})
    rows = page.get("data", []) if isinstance(page, dict) else page
    payload = {
        "count": len(rows),
        "datasets": [
            {
                "dataset_id": d.get("id"),
                "name": d.get("name"),
                "source_type": d.get("source_type"),
                "size_bytes": d.get("size_bytes"),
            }
            for d in rows
        ],
    }

    def table(_: Any) -> str:
        return common.render_table(
            [
                (d["dataset_id"][:8], d["name"], d["source_type"], common.human_bytes(d["size_bytes"]))
                for d in payload["datasets"]
            ],
            ("DATASET", "NAME", "SOURCE", "SIZE"),
        )

    printer.emit(payload, table)
    return payload


def _local_datasets(args: argparse.Namespace) -> dict[str, Any]:
    client = common.ApiClient()
    printer = common.Printer(getattr(args, "json", False))
    payload = client.get("/api/v1/datasets/local-dirs")

    def table(data: Any) -> str:
        rows = [
            (
                entry.get("name"),
                entry.get("kind"),
                entry.get("classes") or "-",
                entry.get("samples") or entry.get("files") or "-",
                common.human_bytes(entry.get("size_bytes")),
                "yes" if entry.get("imported") else "",
            )
            for entry in data.get("datasets", [])
        ]
        return common.render_table(rows, ("NAME", "KIND", "CLASSES", "SAMPLES", "SIZE", "IMPORTED"))

    printer.emit(payload, table)
    return payload
