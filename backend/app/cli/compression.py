"""Unified compression CLI for the inference module.

Usage
-----
    python -m app.cli.compression <subcommand> [options]

Subcommands
-----------
    quantize    ONNX / OpenVINO / NNCF INT4 / GGUF / QAT quantization
    prune       structured pruning (l1/l2/bnslim/autoslim/group_fisher/dmcp) + sparsify
    distill     knowledge distillation (logits KD / FitNets features)
    finetune    LoRA SFT / DPO preference tuning
    export      ONNX / OpenVINO / GGUF export + deploy bundle download
    task        async lifecycle: status / list / wait / download / deploy / delete
    models      list registered models
    datasets    list datasets (incl. server-local open datasets)
    local-datasets  scan the read-only /datasets-local roots
    exit-codes  print the exit-code contract
    env         print the resolved environment overrides

Everything is also configurable through ``COMPRESSION_*`` environment
variables; see ``docs/compression-cli.md`` for the full flag/param tables.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Sequence

from .commands import common, distill, export, finetune, inventory, prune, quantize, task as task_cmd

__all__ = ["main", "build_parser"]

SUBCOMMANDS = (quantize, prune, distill, finetune, export, task_cmd, inventory)

VERSION = "1.0.0"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="compression",
        description="Model compression CLI (async task client for the platform API)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  compression quantize --model Qwen2.5-0.5B --method nncf_int4 --bits 4 --watch\n"
            "  compression quantize --model CifarCNN --method onnx_dynamic_int8 --json\n"
            "  compression prune --model CifarCNN --method group_fisher --rate 0.3 --dataset CIFAR10-train --watch\n"
            "  compression export --model Qwen2.5-0.5B --format gguf --outtype q4_K_M --watch\n"
            "  compression task status <task_id> --json\n"
        ),
    )
    parser.add_argument("--version", action="version", version=f"compression-cli {VERSION}")
    parser.add_argument("--json", action="store_true", help="machine-readable JSON output (top-level)")
    parser.add_argument("--api", default=None, help="API base URL (env COMPRESSION_API)")
    parser.add_argument("--token", default=None, help="bearer token (env COMPRESSION_TOKEN)")
    parser.add_argument("--email", default=None, help="login email (env COMPRESSION_EMAIL)")
    parser.add_argument("--password", default=None, help="login password (env COMPRESSION_PASSWORD)")

    subparsers = parser.add_subparsers(dest="command", metavar="<subcommand>")
    for module in SUBCOMMANDS:
        module.register(subparsers)

    meta = subparsers.add_parser("exit-codes", help="print the exit-code contract")
    meta.set_defaults(handler=_exit_codes)

    env = subparsers.add_parser("env", help="print resolved environment overrides")
    env.set_defaults(handler=_env)
    return parser


def _exit_codes(_args: argparse.Namespace) -> dict[int, str]:
    for code, meaning in sorted(common.EXIT_CODES.items()):
        print(f"{code:>2}  {meaning}")
    return common.EXIT_CODES


def _env(_args: argparse.Namespace) -> dict[str, str]:
    values = {
        key: value
        for key, value in sorted(os.environ.items())
        if key.startswith(common.ENV_PREFIX)
    }
    redacted = {
        key: ("***" if key.endswith(("TOKEN", "PASSWORD")) else value)
        for key, value in values.items()
    }
    print(json.dumps(redacted or {"note": "no COMPRESSION_* variables set"}, indent=2))
    return redacted


def _top_level_overrides(argv: Sequence[str]) -> dict[str, object]:
    """Read the connection/output flags when they are given *before* the subcommand."""
    top = argparse.ArgumentParser(add_help=False)
    top.add_argument("--json", action="store_true", default=False)
    top.add_argument("--api", default=None)
    top.add_argument("--token", default=None)
    top.add_argument("--email", default=None)
    top.add_argument("--password", default=None)
    known, _unknown = top.parse_known_args(argv)
    return {key: value for key, value in vars(known).items() if value not in (None, False)}


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(argv if argv is not None else sys.argv[1:])
    parser = build_parser()
    args = parser.parse_args(arguments)
    if not getattr(args, "handler", None):
        parser.print_help()
        return common.EXIT_USAGE

    # Flags may appear before or after the subcommand; the subcommand-level values
    # (when explicitly set) win over the top-level ones.
    for key, value in _top_level_overrides(arguments).items():
        if getattr(args, key, None) in (None, False):
            setattr(args, key, value)
    common.preprocess_args(args)

    try:
        args.handler(args)
    except common.CliError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return exc.code or common.EXIT_ERROR
    except KeyboardInterrupt:  # pragma: no cover - interactive cancel
        print("interrupted", file=sys.stderr)
        return 130
    return common.EXIT_OK


if __name__ == "__main__":  # pragma: no cover - script entry
    raise SystemExit(main())
