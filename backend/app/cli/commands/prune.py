"""``prune`` subcommand: structured filter pruning + unstructured sparsify.

================  ================  ==================================================
--method          platform method   notes
================  ================  ==================================================
l1                prune             structured pruning, |w| L1-norm criterion
l2                prune             structured pruning, |w| L2-norm criterion (default)
bnslim            prune             BN-Slim: importance = |BN gamma|
autoslim          prune             L1 sparsification then gamma-based pruning (needs data)
group_fisher      prune             Group Fisher information (needs data)
dmcp              prune             DMCP first-order Taylor importance (needs data)
sparsify          sparsify          unstructured magnitude pruning ({uniform, global})
================  ================  ==================================================

PyTorch models need a companion ``arch.py`` next to the checkpoint (uploaded
with the model) so the worker can rebuild the module.
"""
from __future__ import annotations

import argparse
from typing import Any

from . import common

METHODS: dict[str, tuple[str, str]] = {
    "l1": ("prune", "structured pruning, L1 filter norm"),
    "l2": ("prune", "structured pruning, L2 filter norm"),
    "bnslim": ("prune", "structured pruning via BN gamma (BN-Slim)"),
    "autoslim": ("prune", "sparse training + gamma pruning (AutoSlim)"),
    "group_fisher": ("prune", "data-driven Group Fisher importance"),
    "dmcp": ("prune", "data-driven DMCP Taylor importance"),
    "sparsify": ("sparsify", "unstructured magnitude pruning"),
}

DATA_METHODS = {"autoslim", "group_fisher", "dmcp"}
STRUCTURED = {"l1", "l2", "bnslim", "autoslim", "group_fisher", "dmcp"}

PARAMS: list[tuple[str, type, str]] = [
    ("rate", float, "prune ratio 0<rate<1 (default 0.3)"),
    ("dataset", str, "platform dataset id/name (required by data-driven methods)"),
    ("fisher-image-size", int, "image size for Fisher estimation (default 224)"),
    ("fisher-iters", int, "Fisher iterations (default 12)"),
    ("fisher-batch-size", int, "Fisher batch size (default 8)"),
    ("autoslim-epochs", int, "AutoSlim sparse-training epochs (default 2)"),
    ("autoslim-l1", float, "AutoSlim L1 regularisation strength (default 1e-2)"),
    ("autoslim-lr", float, "AutoSlim learning rate (default 3e-3)"),
    ("strategy", str, "sparsify strategy: uniform | global (default uniform)"),
    ("input-shape", str, "forward-check / ONNX export input shape, e.g. 1x3x32x32"),
    ("opset", int, "ONNX opset for the pruned export (default 12)"),
    ("export-onnx", str, "also export the pruned model to ONNX true|false (default true)"),
]


def register(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "prune",
        help="prune a PyTorch model (structured / unstructured)",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="model id / name / name-prefix / UUID")
    parser.add_argument("--method", required=True, choices=sorted(METHODS), help="pruning method")
    for flag, cast, helptext in PARAMS:
        parser.add_argument(f"--{flag}", dest=flag.replace("-", "_"), type=cast, default=None, help=helptext)
    common.add_global_flags(parser)
    parser.set_defaults(handler=run)


def build_params(args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    platform_method, _ = METHODS[args.method]
    client = common.ApiClient()
    params: dict[str, Any] = {"rate": common.resolve_float("rate", args.rate, 0.3)}

    if platform_method == "prune":
        params["algorithm"] = args.method
    if args.method == "sparsify":
        params["strategy"] = common.resolve("strategy", args.strategy, "uniform")
        params.pop("algorithm", None)
    if args.method in DATA_METHODS:
        if not args.dataset:
            raise common.fail(
                f"method {args.method} is data-driven: pass --dataset <name|id>", common.EXIT_USAGE
            )
        params["dataset_id"] = common.resolve_dataset(client, args.dataset)["id"]
        if args.method in {"group_fisher", "dmcp", "autoslim"}:
            params["fisher_image_size"] = common.resolve_int("fisher-image-size", args.fisher_image_size, 224)
            params["fisher_iters"] = common.resolve_int("fisher-iters", args.fisher_iters, 12)
            params["fisher_batch_size"] = common.resolve_int("fisher-batch-size", args.fisher_batch_size, 8)
        if args.method == "autoslim":
            params["autoslim_epochs"] = common.resolve_int("autoslim-epochs", args.autoslim_epochs, 2)
            params["autoslim_l1"] = common.resolve_float("autoslim-l1", args.autoslim_l1, 1e-2)
            params["autoslim_lr"] = common.resolve_float("autoslim-lr", args.autoslim_lr, 3e-3)
    elif args.dataset:
        params["dataset_id"] = common.resolve_dataset(client, args.dataset)["id"]

    if args.input_shape:
        params["input_shape"] = common.parse_shape(args.input_shape)
    if args.opset is not None:
        params["opset"] = common.resolve_int("opset", args.opset, 12)
    if args.export_onnx is not None:
        params["export_onnx"] = common.resolve_bool("export-onnx", args.export_onnx, True)

    return platform_method, common.merged_params(args, params)


def run(args: argparse.Namespace) -> dict[str, Any]:
    method, params = build_params(args)
    return common.run_submit_flow(args, args.model, method, params)
