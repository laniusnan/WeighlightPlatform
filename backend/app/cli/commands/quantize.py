"""``quantize`` subcommand: INT8/INT4/GGUF quantization methods.

Friendly method names map onto the platform's compression methods:

======================  ==========================  =========================================
--method                platform method             artifact
======================  ==========================  =========================================
onnx_dynamic_int8       quantize_onnx               model_quantized.onnx (weights-only INT8)
onnx_static_int8        quantize_onnx_static        model_quantized.onnx (+ calibration)
openvino_int8           openvino_int8               model.xml/.bin (OpenVINO IR INT8/fp16)
nncf_int4               llm_quantize                OpenVINO IR LLM (INT4 group-wise)
gguf                    export_gguf                 model.gguf (q8_0/q4_0/q4_K_M/Q6_K/...)
qat_int8                qat                         model_qat.onnx (FX QDQ or LSQ)
======================  ==========================  =========================================
"""
from __future__ import annotations

import argparse
from typing import Any

from . import common

METHODS: dict[str, tuple[str, str]] = {
    "onnx_dynamic_int8": ("quantize_onnx", "ONNX dynamic INT8 (weights only, no calibration data)"),
    "onnx_static_int8": ("quantize_onnx_static", "ONNX static INT8 QDQ (needs calibration images)"),
    "openvino_int8": ("openvino_int8", "OpenVINO IR + NNCF INT8 post-training quantization"),
    "nncf_int4": ("llm_quantize", "LLM INT4/INT8 group-wise -> OpenVINO IR"),
    "gguf": ("export_gguf", "HuggingFace LLM -> GGUF for llama.cpp / ollama"),
    "qat_int8": ("qat", "Quantization-aware fine-tuning -> INT8 QDQ ONNX"),
}

# flag, type, help, applies-to (empty set = every method)
PARAMS: list[tuple[str, type, str, set[str]]] = [
    ("opset", int, "ONNX opset (default 12)", {"onnx_dynamic_int8"}),
    ("image-size", int, "calibration image size (default 224)", {"onnx_static_int8", "qat_int8"}),
    ("input-shape", str, "explicit input shape, e.g. 1x3x32x32", {"onnx_static_int8", "openvino_int8", "qat_int8"}),
    ("calibration-samples", int, "calibration samples (16 static ONNX / 8 OpenVINO)", {"onnx_static_int8", "openvino_int8"}),
    ("calib-dataset", str, "platform dataset id/name used for calibration", {"onnx_static_int8", "openvino_int8"}),
    ("precision", str, "OpenVINO precision: int8 | fp16 (fp16 = plain FP16 IR)", {"openvino_int8"}),
    ("model-kind", str, "cnn | yolo", {"openvino_int8"}),
    ("imgsz", int, "YOLO input size (default 640)", {"openvino_int8"}),
    ("bits", int, "LLM weight bits 4|8 (default 4)", {"nncf_int4"}),
    ("group-size", int, "INT4 group size 32/64/128/256 (default 128)", {"nncf_int4"}),
    ("sym", str, "symmetric weights true|false (default true)", {"nncf_int4"}),
    ("engine", str, "nncf | optimum | auto", {"nncf_int4"}),
    ("outtype", str, "GGUF outtype f16|bf16|q8_0|q4_0|q4_K|q4_K_S|q4_K_M|q6_K (default f16)", {"gguf"}),
    ("scheme", str, "QAT scheme fx|lsq (default fx)", {"qat_int8"}),
    ("lsq-bits", int, "LSQ weight bits (default 8)", {"qat_int8"}),
    ("epochs", int, "fine-tuning epochs (default 3)", {"qat_int8"}),
    ("lr", float, "learning rate (default 1e-3)", {"qat_int8"}),
    ("dataset", str, "platform dataset id/name (per-class image folder)", {"qat_int8"}),
]


def register(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "quantize",
        help="quantize a model (ONNX / OpenVINO / NNCF / GGUF / QAT)",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="model id / name / name-prefix / UUID")
    parser.add_argument("--method", required=True, choices=sorted(METHODS), help="quantization method")
    for flag, cast, helptext, _applies in PARAMS:
        parser.add_argument(f"--{flag}", dest=flag.replace("-", "_"), type=cast, default=None, help=helptext)
    common.add_global_flags(parser)
    parser.set_defaults(handler=run)


def build_params(args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    platform_method, _description = METHODS[args.method]
    client = common.ApiClient()
    params: dict[str, Any] = {}

    if args.method == "onnx_dynamic_int8":
        params["opset"] = common.resolve_int("opset", args.opset, 12)
    elif args.method == "onnx_static_int8":
        params["image_size"] = common.resolve_int("image-size", args.image_size, 224)
        params["calibration_samples"] = common.resolve_int("calibration-samples", args.calibration_samples, 16)
    elif args.method == "openvino_int8":
        params["precision"] = common.resolve("precision", args.precision, "int8")
        params["model_kind"] = common.resolve("model-kind", args.model_kind, "cnn")
        params["calibration_samples"] = common.resolve_int("calibration-samples", args.calibration_samples, 8)
    elif args.method == "nncf_int4":
        params["engine"] = common.resolve("engine", args.engine, "nncf")
        params["bits"] = common.resolve_int("bits", args.bits, 4)
        params["group_size"] = common.resolve_int("group-size", args.group_size, 128)
        params["sym"] = str(common.resolve_bool("sym", args.sym, True)).lower()
    elif args.method == "qat_int8":
        params["scheme"] = common.resolve("scheme", args.scheme, "fx")
        params["epochs"] = common.resolve_int("epochs", args.epochs, 3)
        params["lr"] = common.resolve_float("lr", args.lr, 1e-3)
    elif args.method == "gguf":
        params["outtype"] = common.resolve("outtype", args.outtype, "f16")

    if args.input_shape:
        params["input_shape"] = common.parse_shape(args.input_shape)
    if args.imgsz is not None:
        params["imgsz"] = common.resolve_int("imgsz", args.imgsz, 640)
    if args.calib_dataset:
        params["calib_dataset_id"] = common.resolve_dataset(client, args.calib_dataset)["id"]
    if args.dataset:
        params["dataset_id"] = common.resolve_dataset(client, args.dataset)["id"]
    if args.method == "qat_int8" and args.lsq_bits is not None:
        params["lsq_bits"] = common.resolve_int("lsq-bits", args.lsq_bits, 8)

    return platform_method, common.merged_params(args, params)


def run(args: argparse.Namespace) -> dict[str, Any]:
    method, params = build_params(args)
    return common.run_submit_flow(args, args.model, method, params)

