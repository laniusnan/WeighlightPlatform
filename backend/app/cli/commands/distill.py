"""``distill`` subcommand: knowledge distillation (many flavours, one entry).

================  =================  ===========================================================
--method          platform params    maths
================  =================  ===========================================================
logits            method=logits      Hinton KD:  L = (1-α)·CE(y, p_s) + α·T²·KL(p_t/T ‖ p_s/T)
features          method=features    FitNets:    L = CE(y, p_s) + α·MSE(f_t, W·f_s) + KD term
================  =================  ===========================================================

The task's ``--model`` **is the teacher** (a PyTorch checkpoint with a sibling
``arch.py``). The student class must live in that same ``arch.py``:

* a class named ``Student``, or
* the first class whose ``__init__`` accepts ``width_mult`` otherwise.

``--student-arch`` is accepted for interface completeness and forwarded as an
informational param; the platform always reads ``arch.py`` from the teacher's
model directory.
"""
from __future__ import annotations

import argparse
from typing import Any

from . import common

METHODS: dict[str, str] = {
    "logits": "Hinton logits KD (temperature-scaled soft targets)",
    "features": "FitNets feature-map alignment (CNN teacher/student)",
}

PARAMS: list[tuple[str, type, str]] = [
    ("dataset", str, "per-class image dataset id/name (required)"),
    ("temperature", float, "distillation temperature T (default 4.0)"),
    ("alpha", float, "KD weight α in [0,1] (default 0.7)"),
    ("width-mult", float, "student width multiplier (default 0.5)"),
    ("epochs", int, "training epochs (default 5)"),
    ("lr", float, "learning rate (default 1e-3)"),
    ("batch-size", int, "batch size (default 8)"),
    ("image-size", int, "image size (default 224)"),
    ("input-shape", str, "explicit input shape, e.g. 1x3x32x32"),
    ("student-arch", str, "informational: student class source (platform reads teacher's arch.py)"),
]


def register(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "distill",
        help="knowledge distillation (logits KD / FitNets features)",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", "--teacher", dest="model", required=True,
                        help="teacher model id / name / UUID (needs a sibling arch.py)")
    parser.add_argument("--method", required=True, choices=sorted(METHODS), help="distillation flavour")
    for flag, cast, helptext in PARAMS:
        parser.add_argument(f"--{flag}", dest=flag.replace("-", "_"), type=cast, default=None, help=helptext)
    common.add_global_flags(parser)
    parser.set_defaults(handler=run)


def build_params(args: argparse.Namespace) -> dict[str, Any]:
    client = common.ApiClient()
    if not args.dataset:
        raise common.fail("distill requires --dataset <name|id> (per-class image folders)", common.EXIT_USAGE)
    params: dict[str, Any] = {
        "method": args.method,
        "dataset_id": common.resolve_dataset(client, args.dataset)["id"],
        "temperature": common.resolve_float("temperature", args.temperature, 4.0),
        "alpha": common.resolve_float("alpha", args.alpha, 0.7),
        "width_mult": common.resolve_float("width-mult", args.width_mult, 0.5),
        "epochs": common.resolve_int("epochs", args.epochs, 5),
        "lr": common.resolve_float("lr", args.lr, 1e-3),
        "batch_size": common.resolve_int("batch-size", args.batch_size, 8),
        "image_size": common.resolve_int("image-size", args.image_size, 224),
    }
    if args.input_shape:
        params["input_shape"] = common.parse_shape(args.input_shape)
    if args.student_arch:
        params["student_arch"] = args.student_arch
    return common.merged_params(args, params)


def run(args: argparse.Namespace) -> dict[str, Any]:
    return common.run_submit_flow(args, args.model, "distill", build_params(args))
