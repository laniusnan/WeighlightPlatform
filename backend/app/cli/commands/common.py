"""Shared plumbing for the compression CLI (Python standard library only).

Design notes
------------
* The CLI is a thin **async client** over the platform HTTP API: submitting a
  job returns a ``task_id`` immediately, progress can be polled later
  (:func:`submit_task` / :func:`wait_for_task`). This is what the inference
  module needs: fire-and-forget from a web request, then poll from the UI.
* Every flag can be supplied through an environment variable, so the website
  can drive the CLI purely with env config: ``--bits`` -> ``COMPRESSION_BITS``,
  ``--api`` -> ``COMPRESSION_API_BASE``, ... CLI flags win over env vars.
* Two output modes: human-readable tables (default) and machine-readable JSON
  (``--json``) so the frontend can parse the output directly.

Exit codes (see :data:`EXIT_CODES`): 0 ok, 1 generic, 2 usage, 3 auth,
4 not found, 5 validation, 6 task failed, 7 timeout, 8 network, 9 local IO.
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Iterable, Sequence

# --------------------------------------------------------------------------- #
# Exit codes
# --------------------------------------------------------------------------- #
EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2
EXIT_AUTH = 3
EXIT_NOT_FOUND = 4
EXIT_VALIDATION = 5
EXIT_TASK_FAILED = 6
EXIT_TIMEOUT = 7
EXIT_NETWORK = 8
EXIT_LOCAL_IO = 9

EXIT_CODES: dict[int, str] = {
    EXIT_OK: "success",
    EXIT_ERROR: "generic error",
    EXIT_USAGE: "usage / argument error",
    EXIT_AUTH: "authentication or permission error (401/403)",
    EXIT_NOT_FOUND: "model/dataset/task not found (404)",
    EXIT_VALIDATION: "server rejected the request (422)",
    EXIT_TASK_FAILED: "the compression task itself failed",
    EXIT_TIMEOUT: "timed out while waiting for the task",
    EXIT_NETWORK: "network / transport error",
    EXIT_LOCAL_IO: "local file or directory error",
}

ENV_PREFIX = "COMPRESSION_"
API_PREFIX = "/api/v1"


class CliError(Exception):
    """Error carrying the process exit code."""

    def __init__(self, message: str, code: int = EXIT_ERROR) -> None:
        super().__init__(message)
        self.code = code


def fail(message: str, code: int = EXIT_ERROR) -> CliError:
    return CliError(message, code)


# --------------------------------------------------------------------------- #
# Environment-variable overrides
# --------------------------------------------------------------------------- #
def env_name(flag: str) -> str:
    """``--group-size`` -> ``COMPRESSION_GROUP_SIZE``."""
    return ENV_PREFIX + flag.lstrip("-").replace("-", "_").upper()


def env_value(flag: str, default: Any = None) -> Any:
    value = os.environ.get(env_name(flag))
    return default if value in (None, "") else value


def resolve(flag: str, cli_value: Any, default: Any = None) -> Any:
    """Precedence: CLI flag > ``COMPRESSION_<FLAG>`` > built-in default."""
    if cli_value is not None:
        return cli_value
    return env_value(flag, default)


def resolve_int(flag: str, cli_value: Any, default: int) -> int:
    value = resolve(flag, cli_value, default)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        label = flag.lstrip("-")
        raise fail(f"--{label} must be an integer (got {value!r})", EXIT_USAGE) from exc


def resolve_float(flag: str, cli_value: Any, default: float) -> float:
    value = resolve(flag, cli_value, default)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        label = flag.lstrip("-")
        raise fail(f"--{label} must be a number (got {value!r})", EXIT_USAGE) from exc


def resolve_bool(flag: str, cli_value: Any, default: bool = False) -> bool:
    value = resolve(flag, cli_value, default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on", "y"}


def parse_shape(value: str) -> list[int]:
    """``1x3x32x32`` / ``1,3,32,32`` / ``[1,3,32,32]`` -> ``[1, 3, 32, 32]``."""
    import json as _json

    text = value.strip()
    if text.startswith("["):
        try:
            return [int(dim) for dim in _json.loads(text)]
        except (ValueError, TypeError) as exc:
            raise fail(f"invalid --input-shape {value!r}", EXIT_USAGE) from exc
    parts = [p for p in text.replace("x", ",").replace("X", ",").split(",") if p.strip()]
    try:
        return [int(p) for p in parts]
    except ValueError as exc:
        raise fail(f"invalid --input-shape {value!r}", EXIT_USAGE) from exc


def parse_kv(pairs: Sequence[str] | None) -> dict[str, Any]:
    """``--set a=1 --set b=true`` -> ``{"a": 1, "b": True}`` (JSON scalars)."""
    out: dict[str, Any] = {}
    for item in pairs or []:
        if "=" not in item:
            raise fail(f"--set expects key=value, got {item!r}", EXIT_USAGE)
        key, raw = item.split("=", 1)
        try:
            out[key.strip()] = json.loads(raw)
        except json.JSONDecodeError:
            out[key.strip()] = raw
    return out


# --------------------------------------------------------------------------- #
# Output helpers
# --------------------------------------------------------------------------- #
def _display_width(text: str) -> int:
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def _pad(text: str, width: int) -> str:
    return text + " " * max(width - _display_width(text), 0)


def render_table(rows: Iterable[Sequence[Any]], headers: Sequence[str]) -> str:
    """Render a borderless, CJK-aware table without external dependencies."""
    body = [["" if cell is None else str(cell) for cell in row] for row in rows]
    widths = [
        max([_display_width(headers[i])] + [_display_width(row[i]) for row in body])
        for i in range(len(headers))
    ]
    lines = ["  ".join(_pad(h, widths[i]) for i, h in enumerate(headers)).rstrip()]
    lines.append("  ".join("-" * width for width in widths))
    for row in body:
        lines.append("  ".join(_pad(cell, widths[i]) for i, cell in enumerate(row)).rstrip())
    return "\n".join(lines)


def human_bytes(size: Any) -> str:
    try:
        value = float(size or 0)
    except (TypeError, ValueError):
        return str(size)
    if value < 1024:
        return f"{int(value)}B"
    for unit in ("KB", "MB", "GB", "TB"):
        value /= 1024
        if value < 1024 or unit == "TB":
            return f"{value:.1f}{unit}"
    return f"{value:.1f}TB"


class Printer:
    """Write either a table (default) or JSON (``--json``) to stdout."""

    def __init__(self, json_mode: bool = False) -> None:
        self.json_mode = json_mode

    def emit(self, payload: dict[str, Any], table: Callable[[Any], str] | None = None) -> None:
        if self.json_mode or table is None:
            print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        else:
            print(table(payload))

    def note(self, message: str) -> None:
        """Diagnostics: stderr, suppressed in --json mode to keep stdout parseable."""
        if not self.json_mode:
            print(message, file=sys.stderr)


# --------------------------------------------------------------------------- #
# HTTP client
# --------------------------------------------------------------------------- #
class ApiClient:
    """Minimal bearer-token client for the platform API (stdlib urllib)."""

    def __init__(
        self,
        base: str | None = None,
        token: str | None = None,
        email: str | None = None,
        password: str | None = None,
        timeout: float | None = None,
    ) -> None:
        raw_base = str(resolve("api", base, "http://127.0.0.1:8001") or "").rstrip("/")
        # ``--api`` may be given with or without the ``/api/v1`` suffix.
        self.base = raw_base if raw_base.endswith(API_PREFIX) else raw_base + API_PREFIX
        self.timeout = float(resolve("timeout", timeout, 60))
        self.token = resolve("token", token)
        self.email = resolve("email", email)
        self.password = resolve("password", password)

    @staticmethod
    def _normalise(path: str) -> str:
        """Accept both ``/api/v1/x`` and ``/x`` spellings."""
        if path.startswith(API_PREFIX):
            path = path[len(API_PREFIX):]
        return path or "/"

    # -- auth ------------------------------------------------------------- #
    def login(self, force: bool = False) -> str | None:
        if self.token and not force:
            return self.token
        if not self.email:
            return None
        if not self.password:
            if not sys.stdin.isatty():
                raise fail(
                    "no API token available: pass --token/--password "
                    "(or set COMPRESSION_TOKEN / COMPRESSION_PASSWORD)",
                    EXIT_AUTH,
                )
            self.password = getpass.getpass(f"Password for {self.email}: ")
        payload = urllib.parse.urlencode(
            {"username": self.email, "password": self.password}
        ).encode()
        response = self._request(
            "POST",
            "/login/access-token",
            raw=payload,
            content_type="application/x-www-form-urlencoded",
            authenticated=False,
        )
        self.token = response["access_token"]
        return self.token

    # -- transport -------------------------------------------------------- #
    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        raw: bytes | None = None,
        content_type: str = "application/json",
        authenticated: bool = True,
        params: dict[str, Any] | None = None,
    ) -> Any:
        url = f"{self.base}{self._normalise(path)}"
        if params:
            clean = {k: v for k, v in params.items() if v is not None}
            if clean:
                url = f"{url}?{urllib.parse.urlencode(clean)}"
        headers: dict[str, str] = {}
        data = raw
        if payload is not None:
            data = json.dumps(payload).encode()
        if data is not None:
            headers["Content-Type"] = content_type
        if authenticated:
            if not self.token:
                self.login()
            if self.token:
                headers["Authorization"] = f"Bearer {self.token}"
        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read()
                return json.loads(body) if body else {}
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:400]
            raise self._http_error(exc.code, detail) from exc
        except urllib.error.URLError as exc:
            raise fail(f"cannot reach {url}: {exc.reason}", EXIT_NETWORK) from exc
        except TimeoutError as exc:  # pragma: no cover - socket timeout
            raise fail(f"request to {url} timed out", EXIT_TIMEOUT) from exc

    @staticmethod
    def _http_error(status: int, detail: str) -> CliError:
        if status in (401, 403):
            return fail(f"authentication/permission error ({status}): {detail}", EXIT_AUTH)
        if status == 404:
            return fail(f"not found (404): {detail}", EXIT_NOT_FOUND)
        if status == 422:
            return fail(f"validation error (422): {detail}", EXIT_VALIDATION)
        return fail(f"API error ({status}): {detail}", EXIT_ERROR)

    # -- convenience ------------------------------------------------------ #
    def get(self, path: str, **kwargs: Any) -> Any:
        return self._request("GET", path, **kwargs)

    def post(self, path: str, payload: dict[str, Any] | None = None, **kwargs: Any) -> Any:
        return self._request("POST", path, payload=payload or {}, **kwargs)

    def delete(self, path: str, **kwargs: Any) -> Any:
        return self._request("DELETE", path, **kwargs)

    def download(self, path: str, destination: os.PathLike[str] | str) -> str:
        """Stream a binary endpoint (artifact zip) to ``destination``."""
        import pathlib

        dest = pathlib.Path(destination)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not self.token:
            # Binary endpoints are authenticated too.
            self.login()
        request = urllib.request.Request(f"{self.base}{self._normalise(path)}")
        if self.token:
            request.add_header("Authorization", f"Bearer {self.token}")
        try:
            with urllib.request.urlopen(request, timeout=max(self.timeout, 300)) as response:
                with dest.open("wb") as handle:
                    while chunk := response.read(1 << 20):
                        handle.write(chunk)
        except urllib.error.HTTPError as exc:
            raise self._http_error(
                exc.code, exc.read().decode("utf-8", errors="replace")[:400]
            ) from exc
        except urllib.error.URLError as exc:
            raise fail(f"download failed: {exc.reason}", EXIT_NETWORK) from exc
        if not dest.exists() or dest.stat().st_size == 0:
            raise fail(f"downloaded file is empty: {dest}", EXIT_LOCAL_IO)
        return str(dest)


# --------------------------------------------------------------------------- #
# Task lifecycle helpers (async submit + poll)
# --------------------------------------------------------------------------- #
TASK_TERMINAL = {"succeeded", "failed", "canceled"}


def submit_task(
    client: ApiClient,
    model_ref: str,
    method: str,
    params: dict[str, Any],
    dry_run: bool = False,
) -> dict[str, Any]:
    """Submit a compression task; returns the created task (contains ``id``)."""
    payload = {"model_id": model_ref if _looks_like_uuid(model_ref) else None,
               "method": method, "params": params}
    if payload["model_id"] is None:
        payload["model_id"] = resolve_model(client, model_ref)["id"]
    if dry_run:
        return {"__dry_run__": True, "request": {"method": method, "params": params,
                                                "model_id": payload["model_id"]}}
    task = client.post("/api/v1/compressions/", payload)
    if not isinstance(task, dict) or "id" not in task:
        raise fail(f"unexpected submit response: {task}", EXIT_ERROR)
    return task


def fetch_task(client: ApiClient, task_id: str) -> dict[str, Any]:
    task = client.get(f"/api/v1/compressions/{task_id}")
    if not isinstance(task, dict) or "id" not in task:
        raise fail(f"unexpected task response: {task}", EXIT_ERROR)
    return task


def wait_for_task(
    client: ApiClient,
    task_id: str,
    interval: float = 5.0,
    timeout: float = 1800.0,
    on_tick: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Poll until the task reaches a terminal state.

    Raises :class:`CliError` with ``EXIT_TIMEOUT`` / ``EXIT_TASK_FAILED``.
    """
    deadline = time.monotonic() + max(timeout, 1.0)
    last: dict[str, Any] = {}
    while True:
        last = fetch_task(client, task_id)
        status = last.get("status")
        if on_tick:
            on_tick(last)
        if status in TASK_TERMINAL:
            break
        if time.monotonic() >= deadline:
            raise fail(
                f"task {task_id} still {status} after {timeout:.0f}s "
                "(re-run `task status` later)", EXIT_TIMEOUT,
            )
        time.sleep(max(interval, 0.5))
    if last.get("status") == "failed":
        raise fail(
            f"task {task_id} failed: {last.get('error_message') or 'unknown error'}",
            EXIT_TASK_FAILED,
        )
    return last


def _looks_like_uuid(value: str) -> bool:
    parts = value.split("-")
    return len(parts) == 5 and all(len(p) in (8, 4, 4, 4, 12) for p in parts)


def resolve_model(client: ApiClient, reference: str) -> dict[str, Any]:
    """Accept a model UUID, an exact name, or a unique name prefix."""
    if _looks_like_uuid(reference):
        return client.get(f"/api/v1/models/{reference}")
    page = client.get("/api/v1/models/", params={"limit": 200})
    rows = page.get("data", page) if isinstance(page, dict) else page
    exact = [m for m in rows if m.get("name") == reference]
    if len(exact) == 1:
        return exact[0]
    prefix = [m for m in rows if str(m.get("name", "")).startswith(reference)]
    if len(prefix) == 1:
        return prefix[0]
    if not prefix and not exact:
        raise fail(f"no model matches {reference!r} (use `models list` to inspect)", EXIT_NOT_FOUND)
    names = ", ".join(sorted(m["name"] for m in (exact or prefix)))
    raise fail(f"{reference!r} is ambiguous, candidates: {names}", EXIT_USAGE)


def resolve_dataset(client: ApiClient, reference: str) -> dict[str, Any]:
    """Accept a dataset UUID, an exact name, or a unique name prefix."""
    if _looks_like_uuid(reference):
        # The dataset API has no GET /{id}; a UUID can be used as-is.
        return {"id": reference, "name": reference}
    page = client.get("/api/v1/datasets/", params={"limit": 200})
    rows = page.get("data", page) if isinstance(page, dict) else page
    exact = [d for d in rows if d.get("name") == reference]
    if len(exact) == 1:
        return exact[0]
    prefix = [d for d in rows if str(d.get("name", "")).startswith(reference)]
    if len(prefix) == 1:
        return prefix[0]
    if not prefix:
        raise fail(f"no dataset matches {reference!r}", EXIT_NOT_FOUND)
    names = ", ".join(sorted(d["name"] for d in prefix))
    raise fail(f"{reference!r} is ambiguous, candidates: {names}", EXIT_USAGE)


# --------------------------------------------------------------------------- #
# Presentation of tasks / artifacts
# --------------------------------------------------------------------------- #
ARTIFACT_TABLE_KEYS = (
    "precision", "model_kind", "tool", "outtype", "stage", "method",
    "total_filters", "pruned_filters", "params_after", "head_layers_adjusted",
    "forward_check", "topology_warning", "size_reduction_pct", "ratio",
    "trainable_params", "preference_pairs", "dpo_ready", "onnx_exported",
)


def summarize_metrics(task: dict[str, Any]) -> dict[str, Any]:
    metrics = task.get("metrics") or {}
    summary = {k: v for k, v in metrics.items() if k in ARTIFACT_TABLE_KEYS and v is not None}
    if isinstance(metrics.get("eval"), dict):
        summary["eval"] = metrics["eval"]
    if isinstance(metrics.get("benchmark"), dict):
        summary["benchmark"] = metrics["benchmark"]
    return summary


def task_payload(task: dict[str, Any]) -> dict[str, Any]:
    """Canonical JSON payload for one task (stable keys for the frontend)."""
    payload = {
        "task_id": task.get("id"),
        "model_id": task.get("model_id"),
        "model_name": task.get("model_name"),
        "method": task.get("method"),
        "status": task.get("status"),
        "progress": task.get("progress"),
        "error_message": task.get("error_message"),
        "output_path": task.get("output_path"),
        "output_size_bytes": task.get("output_size_bytes"),
        "input_size_bytes": task.get("input_size_bytes"),
        "metrics": task.get("metrics") or {},
        "created_at": task.get("created_at"),
        "finished_at": task.get("finished_at"),
    }
    if payload["output_path"]:
        base = f"/api/v1/compressions/{task.get('id')}"
        payload["download_url"] = f"{base}/download"
        payload["deploy_export_url"] = f"{base}/deploy-export"
    # The public task schema does not expose ``output_path``; a finished task
    # with a non-zero output size is what guarantees a downloadable artifact.
    artifact_ready = (
        payload["status"] == "succeeded" and (payload.get("output_size_bytes") or 0) > 0
    )
    payload["artifact_available"] = artifact_ready
    if artifact_ready:
        base = f"/api/v1/compressions/{task.get('id')}"
        payload.setdefault("download_url", f"{base}/download")
        payload.setdefault("deploy_export_url", f"{base}/deploy-export")
    return payload


def render_task(task: dict[str, Any]) -> str:
    payload = task_payload(task)
    rows = [
        ("task_id", payload["task_id"]),
        ("model", f"{payload.get('model_name')} ({payload.get('model_id')})"),
        ("method", payload["method"]),
        ("status", payload["status"]),
        ("progress", f"{payload.get('progress')}%"),
        ("artifact", payload.get("output_path")
         or ("available (see download_url)" if payload.get("artifact_available") else "-")),
        ("size", f"{human_bytes(payload.get('output_size_bytes'))} "
                 f"(input {human_bytes(payload.get('input_size_bytes'))})"),
        ("error", payload.get("error_message") or "-"),
    ]
    metrics = summarize_metrics(task)
    table = render_table(rows, ("FIELD", "VALUE"))
    if metrics:
        metric_rows = []
        for key, value in metrics.items():
            if isinstance(value, dict):
                value = json.dumps(value, ensure_ascii=False)
            metric_rows.append((key, value))
        table += "\n\n" + render_table(metric_rows, ("METRIC", "VALUE"))
    if payload.get("download_url"):
        table += (
            f"\n\nartifact zip : {payload['download_url']}"
            f"\ndeploy bundle: {payload['deploy_export_url']}"
        )
    return table


def run_submit_flow(args: argparse.Namespace, model_ref: str, method: str,
                    params: dict[str, Any]) -> dict[str, Any]:
    """Shared body of every compression subcommand.

    Submits the job, optionally watches it, prints table/JSON and returns the
    task payload (used by tests and by ``--json`` consumers).
    """
    client = ApiClient()
    printer = Printer(json_mode=getattr(args, "json", False))
    interval = resolve_float("poll-interval", getattr(args, "poll_interval", None), 5.0)
    wait_timeout = resolve_float("wait-timeout", getattr(args, "wait_timeout", None), 1800.0)

    task = submit_task(client, model_ref, method, params, dry_run=getattr(args, "dry_run", False))
    if task.get("__dry_run__"):
        printer.emit({"dry_run": True, "request": task["request"]})
        return {"dry_run": True, **task["request"]}

    printer.note(f"submitted  task_id={task['id']}  method={method}  status={task.get('status')}")
    if getattr(args, "watch", False):
        ticks = {"n": 0}

        def on_tick(current: dict[str, Any]) -> None:
            ticks["n"] += 1
            if ticks["n"] % 3 == 1 or current.get("status") in TASK_TERMINAL:
                printer.note(
                    f"  {current.get('status'):<9} {current.get('progress'):>5}%  "
                    f"{current.get('error_message') or ''}"
                )

        try:
            task = wait_for_task(client, task["id"], interval, wait_timeout, on_tick)
        except CliError as exc:
            # Surface the final state even when the job failed/timed out.
            try:
                printer.emit(task_payload(fetch_task(client, task["id"])))
            except CliError:
                pass
            raise exc

    payload = task_payload(task)
    printer.emit(payload, render_task)
    return payload


# --------------------------------------------------------------------------- #
# Shared argparse flags
# --------------------------------------------------------------------------- #
def add_global_flags(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Connection / output flags shared by every subcommand.

    Every flag is also honoured through ``COMPRESSION_<FLAG>`` env vars, e.g.
    ``--api`` -> ``COMPRESSION_API`` ... plus the alias ``COMPRESSION_API_BASE``
    which is what the website sets.
    """
    group = parser.add_argument_group("connection / output")
    group.add_argument("--api", metavar="URL", default=env_value("api", env_value("api-base")),
                       help=f"API base URL (env {env_name('api')} / COMPRESSION_API_BASE, default http://127.0.0.1:8001)")
    group.add_argument("--token", default=env_value("token"), help=f"Bearer token (env {env_name('token')})")
    group.add_argument("--email", default=env_value("email"), help=f"login email (env {env_name('email')})")
    group.add_argument("--password", default=env_value("password"), help=f"login password (env {env_name('password')})")
    group.add_argument("--timeout", type=float, default=None, help="HTTP timeout seconds (env COMPRESSION_TIMEOUT)")
    group.add_argument("--poll-interval", type=float, default=None, help="polling interval seconds (env COMPRESSION_POLL_INTERVAL)")
    group.add_argument("--wait-timeout", type=float, default=None, help="max seconds to wait with --watch (env COMPRESSION_WAIT_TIMEOUT, default 1800)")
    group.add_argument("--watch", action=argparse.BooleanOptionalAction, default=None,
                       help="poll until the task finishes (env COMPRESSION_WATCH=1)")
    group.add_argument("--json", action="store_true", default=False, help="machine-readable JSON output")
    group.add_argument("--output-format", dest="output_format", choices=("table", "json"), default=None,
                       help="output format (same as --json; env COMPRESSION_OUTPUT_FORMAT)")
    group.add_argument("--dry-run", action="store_true", default=False, help="print the resolved payload without submitting")
    group.add_argument("--set", dest="extra_params", action="append", metavar="KEY=VALUE",
                       help="extra method params passed through to the platform (repeatable)")
    return parser


def preprocess_args(args: argparse.Namespace) -> argparse.Namespace:
    """Apply env overrides that argparse cannot express (bool env vars)."""
    if getattr(args, "output_format", None) == "json" or env_value("output-format") == "json":
        args.json = True
    if getattr(args, "watch", None) is None:
        args.watch = resolve_bool("watch", None, False)
    return args


def merged_params(args: argparse.Namespace, params: dict[str, Any]) -> dict[str, Any]:
    """Method params + ``--set`` overrides (env ``COMPRESSION_PARAMS`` = JSON)."""
    import json as _json

    base: dict[str, Any] = {}
    raw_env = os.environ.get(f"{ENV_PREFIX}PARAMS")
    if raw_env:
        try:
            loaded = _json.loads(raw_env)
        except _json.JSONDecodeError as exc:
            raise fail(f"COMPRESSION_PARAMS is not valid JSON: {exc}", EXIT_USAGE) from exc
        if not isinstance(loaded, dict):
            raise fail("COMPRESSION_PARAMS must be a JSON object", EXIT_USAGE)
        base.update(loaded)
    base.update({k: v for k, v in params.items() if v is not None})
    base.update(parse_kv(getattr(args, "extra_params", None)))
    return base

