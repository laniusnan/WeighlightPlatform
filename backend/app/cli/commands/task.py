"""``task`` subcommand: async task lifecycle (status / list / wait / download / deploy).

The submit subcommands (``quantize``/``prune``/``distill``/``finetune``/``export``)
return a ``task_id`` immediately; this module is how the inference module polls
progress, fetches artifacts and produces a deployment bundle.
"""
from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

from . import common

TERMINAL = common.TASK_TERMINAL


def register(subparsers: Any) -> None:
    parser = subparsers.add_parser("task", help="inspect / wait for / fetch compression tasks")
    sub = parser.add_subparsers(dest="task_command", required=True)

    status = sub.add_parser("status", help="show one task (status, progress, metrics)")
    status.add_argument("task_id")
    common.add_global_flags(status)
    status.set_defaults(handler=_status)

    listing = sub.add_parser("list", help="list recent tasks (newest first)")
    listing.add_argument("--status", choices=("pending", "running", "succeeded", "failed", "canceled"))
    listing.add_argument("--method", help="filter by platform method (e.g. llm_quantize)")
    listing.add_argument("--limit", type=int, default=20)
    common.add_global_flags(listing)
    listing.set_defaults(handler=_list)

    wait = sub.add_parser("wait", help="block until a task reaches a terminal state")
    wait.add_argument("task_id")
    common.add_global_flags(wait)
    wait.set_defaults(handler=_wait)

    for name, helptext in (("download", "download the artifact zip"), ("deploy", "download the deployment bundle")):
        cmd = sub.add_parser(name, help=helptext)
        cmd.add_argument("task_id")
        cmd.add_argument("--out-dir", dest="out_dir", default=None, help="local directory (default ./exports)")
        common.add_global_flags(cmd)
        cmd.set_defaults(handler=_download, kind=name)

    remove = sub.add_parser("delete", help="delete a task (cancel it first when running)")
    remove.add_argument("task_id")
    remove.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    common.add_global_flags(remove)
    remove.set_defaults(handler=_delete)

    retry = sub.add_parser("retry", help="re-run a failed/canceled task with the same params")
    retry.add_argument("task_id")
    common.add_global_flags(retry)
    retry.set_defaults(handler=_retry)

    reevaluate = sub.add_parser("reevaluate", help="re-run eval/benchmark for a finished task")
    reevaluate.add_argument("task_id")
    reevaluate.add_argument("--kind", choices=("eval", "benchmark"), default="eval",
                            help="which stored config to re-run (default eval)")
    common.add_global_flags(reevaluate)
    reevaluate.set_defaults(handler=_reevaluate)


def _status(args: argparse.Namespace) -> dict[str, Any]:
    client = common.ApiClient()
    printer = common.Printer(getattr(args, "json", False))
    payload = common.task_payload(common.fetch_task(client, args.task_id))
    printer.emit(payload, lambda _: common.render_task(common.fetch_task(client, args.task_id)))
    return payload


def _list(args: argparse.Namespace) -> dict[str, Any]:
    client = common.ApiClient()
    printer = common.Printer(getattr(args, "json", False))
    page = client.get("/api/v1/compressions/", params={"limit": common.resolve_int("limit", args.limit, 20)})
    rows = page.get("data", []) if isinstance(page, dict) else page
    if args.status:
        rows = [t for t in rows if t.get("status") == args.status]
    if args.method:
        rows = [t for t in rows if t.get("method") == args.method]
    payload = {"count": len(rows), "tasks": [common.task_payload(t) for t in rows]}

    def table(_: Any) -> str:
        return common.render_table(
            [
                (
                    t["task_id"][:8],
                    t.get("model_name"),
                    t["method"],
                    t["status"],
                    f"{t.get('progress')}%",
                    common.human_bytes(t.get("output_size_bytes")),
                    (t.get("created_at") or "")[:19],
                )
                for t in payload["tasks"]
            ],
            ("TASK", "MODEL", "METHOD", "STATUS", "PROG", "OUTPUT", "CREATED"),
        )

    printer.emit(payload, table)
    return payload


def _wait(args: argparse.Namespace) -> dict[str, Any]:
    client = common.ApiClient()
    printer = common.Printer(getattr(args, "json", False))
    interval = common.resolve_float("poll-interval", args.poll_interval, 5.0)
    timeout = common.resolve_float("wait-timeout", args.wait_timeout, 1800.0)
    counter = {"n": 0}

    def on_tick(current: dict[str, Any]) -> None:
        counter["n"] += 1
        if counter["n"] % 3 == 1 or current.get("status") in TERMINAL:
            printer.note(f"  {current.get('status'):<9} {current.get('progress'):>5}%")

    task = common.wait_for_task(client, args.task_id, interval, timeout, on_tick)
    payload = common.task_payload(task)
    printer.emit(payload, common.render_task)
    return payload


def _download(args: argparse.Namespace, kind: str = "download") -> dict[str, Any]:
    client = common.ApiClient()
    printer = common.Printer(getattr(args, "json", False))
    out_dir = pathlib.Path(common.resolve("out-dir", args.out_dir, "exports"))
    endpoint = "deploy-export" if kind == "deploy" else "download"
    suffix = "deploy" if kind == "deploy" else "artifact"
    dest = out_dir / f"{suffix}_{args.task_id}.zip"
    path = client.download(f"/api/v1/compressions/{args.task_id}/{endpoint}", dest)
    payload = {
        "task_id": args.task_id,
        "endpoint": endpoint,
        "saved_to": path,
        "size_bytes": pathlib.Path(path).stat().st_size,
    }
    printer.emit(payload)
    return payload


def _retry(args: argparse.Namespace) -> dict[str, Any]:
    """Re-submit a failed/canceled task with the same method + params."""
    client = common.ApiClient()
    printer = common.Printer(getattr(args, "json", False))
    created = client.post(f"/api/v1/compressions/{args.task_id}/retry")
    if not isinstance(created, dict) or "id" not in created:
        raise common.fail(f"unexpected retry response: {created}", common.EXIT_ERROR)
    printer.note(f"retried  task_id={created['id']}  status={created.get('status')}")
    task = created
    if args.watch:
        interval = common.resolve_float("poll-interval", args.poll_interval, 5.0)
        timeout = common.resolve_float("wait-timeout", args.wait_timeout, 1800.0)

        def on_tick(current: dict[str, Any]) -> None:
            if current.get("status") in TERMINAL:
                printer.note(f"  {current.get('status')} {current.get('progress')}%")

        task = common.wait_for_task(client, created["id"], interval, timeout, on_tick)
    payload = common.task_payload(task)
    printer.emit(payload, common.render_task)
    return payload


def _reevaluate(args: argparse.Namespace) -> dict[str, Any]:
    """Re-run the stored eval / benchmark config of a finished task."""
    client = common.ApiClient()
    printer = common.Printer(getattr(args, "json", False))
    result = client.post(f"/api/v1/compressions/{args.task_id}/reevaluate", {"kind": args.kind})
    payload = {"task_id": args.task_id, "kind": args.kind, **(result or {})}
    printer.note(f"queued {args.kind} re-run for task {args.task_id}")
    if args.watch:
        interval = common.resolve_float("poll-interval", args.poll_interval, 5.0)
        timeout = common.resolve_float("wait-timeout", args.wait_timeout, 1800.0)
        task = common.wait_for_task(client, args.task_id, interval, timeout)
        payload = common.task_payload(task)
    printer.emit(payload, None if args.watch else lambda _: json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def _delete(args: argparse.Namespace) -> dict[str, Any]:
    client = common.ApiClient()
    printer = common.Printer(getattr(args, "json", False))
    if not args.yes:
        answer = input(f"delete task {args.task_id}? [y/N] ").strip().lower()
        if answer not in {"y", "yes"}:
            raise common.fail("aborted", common.EXIT_OK)
    result = client.delete(f"/api/v1/compressions/{args.task_id}")
    printer.emit(result if isinstance(result, dict) else {"result": result})
    return result
