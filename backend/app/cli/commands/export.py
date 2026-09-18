"""``export`` subcommand: convert a model into a deployable runtime format.

============  ============================  ==================================================
--format      platform method               artifact
============  ============================  ==================================================
onnx          export_onnx                    model.onnx (onnxsim-simplified, weights inline)
onnx_int8     quantize_onnx                  model_quantized.onnx (dynamic INT8)
openvino      openvino_int8 (precision)      model.xml/.bin (fp16 or int8 IR)
gguf          export_gguf                    model.gguf (f16/bf16/q8_0/q4_0/q4_K*/q6_K)
deploy_zip    ------- (no task) -------      deployment package via /deploy-export
============  ============================  ==================================================

``deploy_zip`` does not submit a new task: it downloads the deployment bundle
(Dockerfile + ``serve.py`` + ``model_card.json`` + weights) of an already
finished task, which is exactly what the inference module ships to production.
"""
from __future__ import annotations

import argparse
import pathlib
from typing import Any

from . import common

FORMATS: dict[str, str] = {
    "onnx": "PyTorch -> ONNX (export_onnx)",
    "onnx_int8": "PyTorch -> ONNX dynamic INT8 (quantize_onnx)",
    "openvino": "PyTorch/ONNX -> OpenVINO IR (fp16 or int8)",
    "gguf": "HuggingFace LLM -> GGUF (llama.cpp / ollama)",
    "deploy_zip": "download the deployment package of a finished task",
}

PARAMS: list[tuple[str, type, str]] = [
    ("input-shape", str, "explicit input shape, e.g. 1x3x32x32 (default 1x3x224x224)"),
    ("opset", int, "ONNX opset (default 12)"),
    ("dynamic-axes", str, "dynamic batch axis true|false (default false)"),
    ("model-kind", str, "cnn | yolo"),
    ("imgsz", int, "YOLO input size (default 640)"),
    ("precision", str, "OpenVINO precision int8|fp16 (default fp16 for export)"),
    ("outtype", str, "GGUF outtype f16|bf16|q8_0|q4_0|q4_K|q4_K_S|q4_K_M|q6_K (default f16)"),
    ("task", str, "task UUID for --format deploy_zip"),
    ("out-dir", str, "local directory for --format deploy_zip (default ./exports)"),
]


def register(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "export",
        help="export a model (ONNX / OpenVINO / GGUF) or fetch a deploy package",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", help="model id / name / name-prefix / UUID (not needed for deploy_zip)")
    parser.add_argument("--format", dest="fmt", required=True, choices=sorted(FORMATS), help="target format")
    for flag, cast, helptext in PARAMS:
        parser.add_argument(f"--{flag}", dest=flag.replace("-", "_"), type=cast, default=None, help=helptext)
    common.add_global_flags(parser)
    parser.set_defaults(handler=run)


def _download_deploy(args: argparse.Namespace) -> dict[str, Any]:
    if not args.task:
        raise common.fail("--format deploy_zip requires --task <task_uuid>", common.EXIT_USAGE)
    client = common.ApiClient()
    out_dir = pathlib.Path(common.resolve("out-dir", args.out_dir, "exports"))
    dest = out_dir / f"deploy_{args.task}.zip"
    path = client.download(f"/api/v1/compressions/{args.task}/deploy-export", dest)
    return {"task_id": args.task, "format": "deploy_zip", "saved_to": path,
            "size_bytes": pathlib.Path(path).stat().st_size}


def build_params(args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    params: dict[str, Any] = {}
    if args.fmt == "onnx":
        method = "export_onnx"
        params["opset"] = common.resolve_int("opset", args.opset, 12)
        params["export_onnx"] = True
    elif args.fmt == "onnx_int8":
        method = "quantize_onnx"
        params["opset"] = common.resolve_int("opset", args.opset, 12)
    elif args.fmt == "openvino":
        method = "openvino_int8"
        params["precision"] = common.resolve("precision", args.precision, "fp16")
    else:  # gguf
        method = "export_gguf"
        params["outtype"] = common.resolve("outtype", args.outtype, "f16")

    if args.input_shape:
        params["input_shape"] = common.parse_shape(args.input_shape)
    if args.model_kind:
        params["model_kind"] = common.resolve("model-kind", args.model_kind, "cnn")
    if args.imgsz is not None:
        params["imgsz"] = common.resolve_int("imgsz", args.imgsz, 640)
    if args.dynamic_axes is not None:
        params["dynamic_axes"] = common.resolve_bool("dynamic-axes", args.dynamic_axes, False)
    return method, common.merged_params(args, params)


def run(args: argparse.Namespace) -> dict[str, Any]:
    printer = common.Printer(json_mode=getattr(args, "json", False))
    if args.fmt == "deploy_zip":
        payload = _download_deploy(args)
        printer.emit(payload)
        return payload
    if not args.model:
        raise common.fail("--model is required (except for --format deploy_zip)", common.EXIT_USAGE)
    method, params = build_params(args)
    return common.run_submit_flow(args, args.model, method, params)
