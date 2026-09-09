#!/usr/bin/env python3
"""optimum-free feasibility: torch MLP -> ONNX fp32 -> OpenVINO IR
-> nncf.compress_weights(INT4_ASYM, group_size=32, example small dims) -> save & run CPU.

GPT2/decoder-specific dynamic export stays an engineering task; this proves
the INT4 weight-compression + OpenVINO path without optimum.
"""
import os, tempfile, shutil
from pathlib import Path
import torch, torch.nn as nn

os.environ.setdefault("FASTAPI_ENV", "development")
print("1/5 build MLP", flush=True)
net = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 128), nn.Linear(128, 32))
net.eval()
work = Path(tempfile.mkdtemp())
x = torch.randn(1, 64)

print("2/5 export fp32 ONNX", flush=True)
onnx_path = work / "mlp_fp32.onnx"
torch.onnx.export(net, x, str(onnx_path), input_names=["input"], output_names=["output"], opset_version=14, dynamo=False)

import openvino as ov
import nncf

print("3/5 convert to OpenVINO fp32 IR", flush=True)
ov_model = ov.convert_model(str(onnx_path))
ir_dir = work / "ir_fp32"
ov.save_model(ov_model, ir_dir / "model.xml")
print("4/5 nncf INT4 group compression", flush=True)
compressed = nncf.compress_weights(ov_model, mode=nncf.CompressWeightsMode.INT4_ASYM, group_size=32, ratio=1.0)
q_dir = work / "ir_int4"
ov.save_model(compressed, q_dir / "model.xml")
print("5/5 run on CPU", flush=True)
compiled = ov.Core().compile_model(q_dir / "model.xml", "CPU")
out = compiled({compiled.input(0).any_name: x.numpy()})

def size_of(d: Path) -> int:
    return sum(f.stat().st_size for f in d.iterdir() if f.is_file())

print("E2E_NNCF_INT4_OK", "fp32_bytes=", size_of(ir_dir), "int4_bytes=", size_of(q_dir), "out_shape=", str(list(out[compiled.output(0)].shape)))
shutil.rmtree(work, ignore_errors=True)
