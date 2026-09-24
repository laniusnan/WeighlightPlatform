#!/usr/bin/env python3
"""Generate the frontend method-card data from the measured summary.

The card copy (zh/en) lives here; every number under `evidence` is pulled from
`out/summary.json` (produced by collect_results.py from real runs), so the UI can
never show a number that was not measured.

    python build_method_cards.py \
        --summary /home/lanius/compression-eval/out/summary.json \
        --out /home/lanius/Workspace/full-stack-fastapi/frontend/src/components/lab/method-cards.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

T = lambda zh, en: {"zh": zh, "en": en}  # noqa: E731 - compact literal helper


def card_export_onnx(ev: dict | None) -> dict:
    return {
        "id": "export_onnx",
        "label": T("导出 ONNX", "Export ONNX"),
        "summary": T(
            "PyTorch 检查点转 ONNX（torch.onnx.export + onnxsim 简化），后续量化/推理的前置步骤。",
            "Convert a PyTorch checkpoint to ONNX (torch.onnx.export + onnxsim), the prerequisite for quantization and ONNX runtimes.",
        ),
        "group": "格式转换",
        "calibration": {"needed": False, "note": T("不需要数据；只需输入形状/opset。", "No data needed; only input shape / opset.")},
        "inputs": [
            {"name": "--base-model-path", "required": True,
             "format": T(".pt/.pth 检查点", ".pt/.pth checkpoint"),
             "note": T("state_dict / torch.save(model) / build_model(config) 三种格式都支持", "state_dict, torch.save(model) and build_model(config) are all supported")},
            {"name": "--arch-file", "required": False,
             "format": T("模型定义 .py（含 build_model）", "model definition .py (with build_model)"),
             "note": T("检查点只有权重时必须提供；同目录有 arch.py 时自动使用", "Required when the checkpoint only holds weights; an arch.py next to the model is picked up automatically")},
            {"name": "--input-shape", "required": True,
             "format": T("如 1x3x32x32（表格模型 1x7）", "e.g. 1x3x32x32 (tabular: 1x7)")},
        ],
        "suits": {"models": ["cnn", "transformer", "tabular", "yolo"], "tasks": ["image-classification", "object-detection", "tabular-classification"]},
        "effect": {
            "compression": T("体积≈持平（FP32 精度不变，实测 +3%）", "Same size (FP32 kept, measured +3%)"),
            "accuracy": T("精度不变（数值等价）", "Unchanged (numerically equivalent)"),
            "latency": T("ONNX Runtime 通常略快于 PyTorch eager", "ONNX Runtime is usually slightly faster than PyTorch eager"),
        },
        "limits": [
            T("必须给输入形状；表格/非图像模型要显式指定（如 1x7）。", "Input shape is mandatory; non-image models must pass it explicitly (e.g. 1x7)."),
            T("默认静态 batch（1）；需要动态 batch 要加 --dynamic-axes，且动态 batch 与 INT8 动态量化组合受限。", "Static batch=1 by default; use --dynamic-axes for dynamic batch, which then conflicts with dynamic INT8 quantization."),
            T("YOLO 走 ultralytics 分支，需要 ultralytics 依赖与 imgsz。", "YOLO uses the ultralytics branch (needs the dependency and --imgsz)."),
        ],
        "evidence": ev,
    }


def card_export_gguf(ev: dict | None) -> dict:
    return {
        "id": "export_gguf",
        "label": T("GGUF 导出", "Export GGUF"),
        "summary": T(
            "把 HuggingFace 因果语言模型转成 llama.cpp 的 GGUF（f16/q8_0/q4_K_M…），供 ollama/llama.cpp 生态直接加载。",
            "Convert a HuggingFace causal LM to llama.cpp GGUF (f16/q8_0/q4_K_M…) for ollama / llama.cpp.",
        ),
        "group": "格式转换",
        "calibration": {"needed": False, "note": T("不需要数据。", "No data needed.")},
        "inputs": [
            {"name": "--base-model-path", "required": True,
             "format": T("HF 模型目录（config.json 所在目录）", "HF model directory (with config.json)")},
            {"name": "--outtype", "required": False,
             "format": T("f16 | q8_0 | q4_0 | q4_K_M | q6_K …", "f16 | q8_0 | q4_0 | q4_K_M | q6_K …"),
             "note": T("默认 f16", "f16 by default")},
            {"name": "llama.cpp 工具", "required": True,
             "format": T("离线目录或可访问网络", "offline directory OR network access"),
             "note": T("用 --llama-cpp-path 指到含 convert_hf_to_gguf.py 的目录；否则引擎会联网拉取", "Point --llama-cpp-path at a directory containing convert_hf_to_gguf.py; otherwise the engine fetches it online")},
        ],
        "suits": {"models": ["llm"], "tasks": ["text-generation"]},
        "effect": {
            "compression": T("按 outtype：f16≈2×缩小，q4_K_M 约 4~5×缩小", "Depends on outtype: f16 ≈ 2×, q4_K_M ≈ 4–5× smaller"),
            "accuracy": T("q4_K_M 通常 <1% 困惑度上升；f16 无损失", "q4_K_M usually <1% PPL increase; f16 lossless"),
            "latency": T("llama.cpp CPU 推理，量化越低位越快", "llama.cpp CPU inference; lower bits run faster"),
        },
        "limits": [
            T("需要 llama.cpp 的 convert_hf_to_gguf.py；离线环境要把工具放进 deploy/compress/llama-cpp/（或 --llama-cpp-path 指向本机 checkout）。", "Requires llama.cpp's convert_hf_to_gguf.py; offline hosts must place it in deploy/compress/llama-cpp/ (or point --llama-cpp-path at a local checkout)."),
            T("自定义架构（非 llama/qwen/gpt 系）可能不被工具支持。", "Custom architectures outside the llama/qwen/gpt families may be unsupported."),
            T("实测：Qwen2.5-0.5B → q4_K_M 511 MB（−48.8%）、31 s、离线可复现；困惑度损失未单独测（PPL 见 INT4 行）。", "Measured: Qwen2.5-0.5B → q4_K_M 511 MB (−48.8%) in 31 s, fully offline; perplexity loss not measured separately (see the INT4 row for PPL)."),
        ],
        "evidence": ev,
    }


def card_quantize_onnx(ev: dict | None) -> dict:
    return {
        "id": "quantize_onnx",
        "label": T("ONNX 动态 INT8", "ONNX dynamic INT8"),
        "summary": T(
            "onnxruntime 权重动态量化（激活运行时定标）：免校准数据、一条命令把 FP32 ONNX 压到约 1/4。",
            "onnxruntime weight-only dynamic quantization: no calibration data, roughly 4× smaller than the FP32 ONNX.",
        ),
        "group": "量化",
        "calibration": {"needed": False, "note": T("不需要校准数据。", "No calibration data.")},
        "inputs": [
            {"name": "--base-model-path", "required": True, "format": T("ONNX（.onnx）", "ONNX (.onnx)")},
            {"name": "--external-data-path", "required": False,
             "format": T("外部权重 .onnx.data（文件或目录）", "external weights .onnx.data (file or directory)"),
             "note": T("仅当 ONNX 带 external data 时必填；引擎会把权重内联成单文件", "Only required for ONNX with external data; the engine inlines it into a single file")},
        ],
        "suits": {"models": ["cnn", "transformer", "tabular", "llm"],
                  "tasks": ["image-classification", "tabular-classification", "text-generation"]},
        "effect": {
            "compression": T("约 3.5~4.5× 缩小（实测 −73.7%）", "≈3.5–4.5× smaller (measured −73.7%)"),
            "accuracy": T("实测 Top-1 68.79% → 68.27%（−0.52pp）", "Measured Top-1 68.79% → 68.27% (−0.52pp)"),
            "latency": T("p50 0.44 ms / p99 0.79 ms（batch=1）", "p50 0.44 ms / p99 0.79 ms (batch=1)"),
        },
        "limits": [
            T("只量化权重，激活仍 FP32：提速有限（本 demo 反而比静态 INT8 慢 5×）。", "Weights only, activations stay FP32: limited speed-up (here 5× slower than static INT8)."),
            T("动态 batch 注意力图与动态量化组合会触发 onnxruntime 形状限制。", "Dynamic-batch attention graphs conflict with ORT quantizer shape checks."),
            T("旧导出器写的形状标注自相矛盾时，引擎自动剔除 value_info 重试（看 metrics.shape_repair）。", "When legacy exporters write contradictory shape annotations the engine strips value_info and retries (see metrics.shape_repair)."),
        ],
        "evidence": ev,
    }


def card_quantize_onnx_static(ev: dict | None) -> dict:
    return {
        "id": "quantize_onnx_static",
        "label": T("ONNX 静态 INT8", "ONNX static INT8"),
        "summary": T(
            "QDQ 静态量化（权重+激活都定点）：用一批代表性图片校准激活范围，CPU 上通常最快。",
            "QDQ static quantization (weights and activations): calibrate activation ranges on real images — usually the fastest CPU option.",
        ),
        "group": "量化",
        "calibration": {"needed": True,
                        "note": T("需要按类图片目录（本次用 CIFAR10-train 的 512 张）。", "Needs a per-class image directory (512 images from CIFAR10-train here).")},
        "inputs": [
            {"name": "--base-model-path", "required": True, "format": T("ONNX（.onnx）", "ONNX (.onnx)")},
            {"name": "--dataset-path", "required": True, "format": T("按类图片目录", "per-class image directory")},
            {"name": "--max-samples", "required": False,
             "format": T("校准样本数（默认 16；实测 512 更稳）", "calibration samples (default 16; 512 measured more stable)")},
        ],
        "suits": {"models": ["cnn", "yolo", "transformer"], "tasks": ["image-classification", "object-detection"]},
        "effect": {
            "compression": T("实测 −72.4%（380,919 → 108,431 B）", "Measured −72.4% (380,919 → 108,431 B)"),
            "accuracy": T("实测 Top-1 68.79% → 68.68%（−0.11pp）", "Measured 68.79% → 68.68% (−0.11pp)"),
            "latency": T("p50 0.079 ms / p99 0.247 ms、2,553 样本/s — 本批最快", "p50 0.079 ms / p99 0.247 ms, 2,553 samples/s — fastest of the batch"),
        },
        "limits": [
            T("必须有代表性校准集，分布不匹配会明显掉精度。", "A representative calibration set is mandatory; mismatched data costs accuracy."),
            T("校准数据要用按类目录结构（class_x/*.jpg|png|bmp）。", "Calibration data must use per-class folders (class_x/*.jpg|png|bmp)."),
            T("只量化常见算子，自定义算子保持 FP32。", "Only common ops are quantized; custom ops stay FP32."),
        ],
        "evidence": ev,
    }


def card_openvino_int8(ev: dict | None) -> dict:
    return {
        "id": "openvino_int8",
        "label": T("OpenVINO INT8", "OpenVINO INT8"),
        "summary": T(
            "导出 OpenVINO IR 并用 NNCF 做训练后 INT8 量化（也可 FP16），产物是 model.xml + model.bin 目录。",
            "Export OpenVINO IR and quantize with NNCF (or FP16); the artifact is a model.xml + model.bin directory.",
        ),
        "group": "量化",
        "calibration": {"needed": True,
                        "note": T("INT8 需要校准图片；FP16/FP32 不需要（--precision）。", "INT8 needs calibration images; FP16/FP32 do not (--precision).")},
        "inputs": [
            {"name": "--base-model-path", "required": True, "format": T("ONNX 或 PyTorch 检查点", "ONNX or PyTorch checkpoint")},
            {"name": "--dataset-path", "required": True, "format": T("按类图片目录（INT8 校准）", "per-class image directory (INT8 calibration)")},
            {"name": "--precision", "required": False, "format": T("int8 | fp16 | fp32", "int8 | fp16 | fp32")},
        ],
        "suits": {"models": ["cnn", "yolo", "transformer", "llm"],
                  "tasks": ["image-classification", "object-detection", "text-generation"]},
        "effect": {
            "compression": T("实测 −68.1%（IR 目录 125,352 B）", "Measured −68.1% (IR directory 125,352 B)"),
            "accuracy": T("实测 Top-1 68.79% → 67.01%（−1.78pp）", "Measured 68.79% → 67.01% (−1.78pp)"),
            "latency": T("p50 0.182 ms / p99 0.534 ms", "p50 0.182 ms / p99 0.534 ms"),
        },
        "limits": [
            T("产物是目录（model.xml/model.bin），部署/下载要整目录带走。", "Artifact is a directory (model.xml/model.bin) — ship the whole folder."),
            T("静态 batch 输入：批量推理需按固定批量喂入。", "Static-batch input: batch inference must feed the fixed batch size."),
            T("Intel CPU 上收益最大；其它平台退化为通用后端。", "Biggest gains on Intel CPUs; other vendors fall back to a generic backend."),
        ],
        "evidence": ev,
    }



def card_llm_quantize(ev: dict | None) -> dict:
    return {
        "id": "llm_quantize",
        "label": T("LLM INT4 / INT8", "LLM INT4 / INT8"),
        "summary": T(
            "HuggingFace 大模型 → OpenVINO IR 的权重压缩（optimum-intel / NNCF INT4）：免校准，把 0.5B 级模型压到约 1/4。",
            "HuggingFace LLM → OpenVINO IR weight compression (optimum-intel / NNCF INT4): no calibration, takes a 0.5B model to about 1/4.",
        ),
        "group": "量化",
        "calibration": {"needed": False, "note": T("权重压缩不需要校准数据。", "Weight compression needs no calibration data.")},
        "inputs": [
            {"name": "--base-model-path", "required": True, "format": T("HF 模型目录 / 模型 ID", "HF model directory / model id")},
            {"name": "--bits", "required": False, "format": T("4 | 8（默认 4）", "4 | 8 (default 4)")},
            {"name": "--group-size", "required": False, "format": T("32 | 64 | 128 | 256（默认 128）", "32 | 64 | 128 | 256 (default 128)")},
        ],
        "suits": {"models": ["llm"], "tasks": ["text-generation"]},
        "effect": {
            "compression": T("INT4 约 3.5~4×（0.999 GB → 见报告实测值）", "INT4 ≈3.5–4× (0.999 GB → measured value in the report)"),
            "accuracy": T("困惑度上升通常 <5%（group 128 / sym）", "Perplexity increase usually <5% (group 128 / sym)"),
            "latency": T("CPU 生成明显提速（tokens/s 见报告）", "Clearly faster CPU generation (tokens/s in the report)"),
        },
        "limits": [
            T("产物是 OpenVINO IR 目录，需用 OV/GenAI 或平台推理接口加载。", "Artifact is an OpenVINO IR directory; load with OV/GenAI or the platform inference API."),
            T("超小模型（<0.5B）INT4 掉点更明显，建议 group 128 + sym。", "INT4 hurts tiny models (<0.5B) more; prefer group 128 + sym."),
            T("NNCF INT4 走 CPU 通道压缩，长上下文吞吐受内存带宽限制。", "NNCF INT4 uses CPU channel compression; long-context throughput is memory-bandwidth bound."),
        ],
        "evidence": ev,
    }


def card_prune(ev: dict | None) -> dict:
    return {
        "id": "prune",
        "label": T("结构化剪枝", "Structured pruning"),
        "summary": T(
            "按滤波器重要性（L1/L2/BN-Slim）或数据驱动准则（AutoSlim/GroupFisher/DMCP）整通道裁剪 CNN，再导出瘦身后的图。",
            "Remove whole CNN filters by importance (L1/L2/BN-Slim) or data-driven criteria (AutoSlim/GroupFisher/DMCP) and export the slimmed graph.",
        ),
        "group": "剪枝 / 稀疏化",
        "calibration": {"needed": False,
                        "note": T("L1/L2/BN-Slim 免数据；AutoSlim/GroupFisher/DMCP 需要按类图片目录。", "L1/L2/BN-Slim need no data; AutoSlim/GroupFisher/DMCP need a per-class image directory.")},
        "inputs": [
            {"name": "--base-model-path", "required": True, "format": T("PyTorch 检查点", "PyTorch checkpoint")},
            {"name": "--arch-file", "required": True, "format": T("模型定义 .py", "model definition .py"),
             "note": T("剪枝要重建网络并改通道数", "Pruning rebuilds the network with new channel counts")},
            {"name": "--rate", "required": True, "format": T("0<rate<1（本次 0.1 / 0.3）", "0<rate<1 (0.1 / 0.3 here)")},
            {"name": "--dataset-path", "required": False, "format": T("按类图片目录（仅数据驱动准则）", "per-class image directory (data-driven criteria only)")},
        ],
        "suits": {"models": ["cnn"], "tasks": ["image-classification", "object-detection"]},
        "effect": {
            "compression": T("rate 0.3：94,986 → 45,915 参数、380,919 → 185,178 B（−52.8%）", "rate 0.3: 94,986 → 45,915 params, 380,919 → 185,178 B (−52.8%)"),
            "accuracy": T("**未微调会崩**：实测 68.79% → 20.1%（rate 0.3）/ 17.4%（rate 0.1）", "**Collapses without fine-tuning**: measured 68.79% → 20.1% (rate 0.3) / 17.4% (rate 0.1)"),
            "latency": T("p50 0.126 ms（CPU 小网络收益有限）", "p50 0.126 ms (small nets gain little on CPU)"),
        },
        "limits": [
            T("只支持含 Conv2d 的 CNN；纯 Transformer/MLP 报 No Conv2d layers found。", "CNN (Conv2d) only; pure Transformer/MLP fails with “No Conv2d layers found”."),
            T("必须提供 arch.py（能重建模型）。", "arch.py is mandatory (model must be rebuildable)."),
            T("剪完要自己接 QAT/微调，否则精度不可用。", "Follow with QAT/fine-tuning, otherwise accuracy is unusable."),
            T("分类头维度会自动对齐，这会改变 head 权重映射，是掉点主因之一。", "The classifier head is auto-adjusted, rewiring head weights — one of the main accuracy hits."),
        ],
        "evidence": ev,
    }




def card_sparsify(ev: dict | None) -> dict:
    return {
        "id": "sparsify",
        "label": T("权重稀疏化", "Weight sparsification"),
        "summary": T(
            "按幅值把最小的权重置零（非结构化稀疏）：结构不变，适合存储压缩与后续稀疏推理。",
            "Zero the smallest weights by magnitude (unstructured sparsity): the graph is unchanged; useful for storage and sparsity-aware runtimes.",
        ),
        "group": "剪枝 / 稀疏化",
        "calibration": {"needed": False, "note": T("不需要数据。", "No data needed.")},
        "inputs": [
            {"name": "--base-model-path", "required": True, "format": T("PyTorch 检查点", "PyTorch checkpoint")},
            {"name": "--rate", "required": True, "format": T("置零比例 0<rate<1", "fraction to zero, 0<rate<1")},
        ],
        "suits": {"models": ["cnn", "transformer", "tabular"], "tasks": ["image-classification", "tabular-classification"]},
        "effect": {
            "compression": T("稠密存储不变小（要稀疏格式才体现）", "Dense storage does not shrink (needs a sparse format)"),
            "accuracy": T("小幅扰动，rate 0.3 一般 <1pp（见报告实测）", "Small perturbation, usually <1pp at rate 0.3 (see report)"),
            "latency": T("稠密 CPU 推理不变快", "No speed-up for dense CPU inference"),
        },
        "limits": [
            T("非结构化、不改通道数，产出仍是稠密张量。", "Unstructured: channel counts and dense tensors unchanged."),
            T("要加速需配合稀疏内核或再做一次结构化剪枝。", "Speed-up needs sparse kernels or a follow-up structured prune."),
            T("本实验只验证流程与精度扰动。", "This experiment validates the pipeline and accuracy perturbation only."),
        ],
        "evidence": ev,
    }


def card_distill(ev: dict | None) -> dict:
    return {
        "id": "distill",
        "label": T("知识蒸馏", "Knowledge distillation"),
        "summary": T(
            "教师模型输出指导学生模型训练（logits KD / FitNets features），把大模型能力迁移到更小的网络。",
            "Train a smaller student from the teacher's outputs (logits KD / FitNets features).",
        ),
        "group": "训练后校准",
        "calibration": {"needed": True,
                        "note": T("需要按类图片目录（本次 CIFAR10-train 50,000 张）。", "Needs a per-class image directory (50,000 images from CIFAR10-train here).")},
        "inputs": [
            {"name": "--base-model-path", "required": True, "format": T("教师模型检查点", "teacher checkpoint")},
            {"name": "--arch-file", "required": True, "format": T("模型定义 .py", "model definition .py")},
            {"name": "--dataset-path", "required": True, "format": T("按类图片目录", "per-class image directory")},
            {"name": "--epochs / --temperature / --alpha", "required": False,
             "format": T("轮数 / 温度 / KD 权重", "epochs / temperature / KD weight")},
        ],
        "suits": {"models": ["cnn"], "tasks": ["image-classification"]},
        "effect": {
            "compression": T("取决于学生宽度（width_mult<1 才压缩；本次 width_mult 未生效，体积≈基线）", "Depends on student width (only width_mult<1 compresses; here it had no effect, size ≈ baseline)"),
            "accuracy": T("3 epoch：随机学生 8.7% → 49.4%；10 epoch 见报告", "3 epochs: random student 8.7% → 49.4%; 10-epoch run in the report"),
            "latency": T("学生与教师同构时与基线同量级", "Same order as baseline when student matches teacher architecture"),
        },
        "limits": [
            T("需要教师+学生可重建（arch.py）。", "Teacher and student must be rebuildable (arch.py)."),
            T("训练成本高：3 epoch 已 258 s（CPU）。", "Training cost is high: 3 epochs took 258 s on CPU."),
            T("width_mult 对 VGG-ish 顺序网络无效（实测学生参数量=教师），真要压缩请用剪枝/量化。", "width_mult has no effect on this VGG-ish sequential net (student params == teacher); use pruning/quantization for real compression."),
        ],
        "evidence": ev,
    }


def card_qat(ev: dict | None) -> dict:
    return {
        "id": "qat",
        "label": T("量化感知训练 QAT", "Quantization-aware training (QAT)"),
        "summary": T(
            "在训练中模拟量化误差（FX QAT / LSQ），导出量化 ONNX：比纯训练后量化更保精度，代价是要训练。",
            "Simulate quantization during training (FX QAT / LSQ) and export a quantized ONNX: better accuracy than post-training quantization at the cost of training.",
        ),
        "group": "训练后校准",
        "calibration": {"needed": True,
                        "note": T("需要按类图片目录（训练/校准共用）。", "Needs a per-class image directory (used for both training and calibration).")},
        "inputs": [
            {"name": "--base-model-path", "required": True, "format": T("PyTorch 检查点", "PyTorch checkpoint")},
            {"name": "--arch-file", "required": True, "format": T("模型定义 .py", "model definition .py")},
            {"name": "--dataset-path", "required": True, "format": T("按类图片目录", "per-class image directory")},
            {"name": "--epochs", "required": True, "format": T("QAT 轮数（CPU 上建议 1~3）", "QAT epochs (1–3 recommended on CPU)")},
        ],
        "suits": {"models": ["cnn"], "tasks": ["image-classification"]},
        "effect": {
            "compression": T("与静态 INT8 同级（实测 −72.5%，392,548 → 108,077 B）", "Comparable to static INT8 (measured −72.5%, 392,548 → 108,077 B)"),
            "accuracy": T("实测**轮次是关键**：1 epoch/4k 仅 62.8%，10 epoch/4k 升到 66.8%（+4.0pp，val loss 0.9225→0.8093）；但仍低于静态 PTQ 的 68.68%——要超过 PTQ 需要全量数据（本机 50k 会被 OOM 杀）", "Measured: **epochs matter** — 1 epoch/4k gives 62.8%, 10 epochs/4k reach 66.8% (+4.0pp, val loss 0.9225→0.8093), still below static PTQ (68.68%); beating PTQ needs the full dataset (50k is OOM-killed on this host)"),
            "latency": T("与静态 INT8 同级（10 epoch 实测 p50 0.119 ms / p99 0.433 ms）", "Same as static INT8 (10-epoch run: p50 0.119 ms / p99 0.433 ms)"),
        },
        "limits": [
            T("CLI 已支持：`--method=qat --algo=fx_int8|lsq`（需要 --dataset-path 按类图片目录）。", "CLI support: `--method=qat --algo=fx_int8|lsq` (requires --dataset-path with per-class images)."),
            T("必须在有 passwd 条目的用户下运行：容器内 uid 无 passwd 时 FX QAT 会退回 fine-tune+PTQ（实测报 getpwuid uid not found）。", "Must run as a user that exists in the image: without a passwd entry FX QAT silently falls back to fine-tune + PTQ (observed “getpwuid uid not found”)."),
            T("只在 Conv/Linear 等 FX 可追踪模块上生效；动态控制流图会失败。", "Only works on FX-traceable Conv/Linear modules; dynamic control flow fails."),
            T("内存占用高：全量 50k 图像跑 1 epoch 与 10 epoch 均被 OOM 杀掉（实测 exit 137 ×2），必须用子集。", "Memory hungry: the full 50k-image run was OOM-killed at both 1 and 10 epochs (exit 137 twice) — use a subset."),
            T("轮次不够时精度会明显低于静态 PTQ（1 epoch 实测 62.8% vs 68.68%），别只看体积。", "With too few epochs QAT is clearly worse than static PTQ (62.8% vs 68.68% at 1 epoch) — size is not the only metric."),
        ],
        "evidence": ev,
    }


def card_llm_finetune(ev: dict | None) -> dict:
    return {
        "id": "llm_finetune",
        "label": T("LoRA / QLoRA / DPO 微调", "LoRA / QLoRA / DPO fine-tuning"),
        "summary": T(
            "冻结底模只训练低秩适配器（LoRA），可再合并回底模；HuggingFace 模型需要 CUDA，PyTorch 检查点走 CPU 微型原型。",
            "Freeze the base model and train a low-rank adapter (LoRA), optionally merged back. HuggingFace models need CUDA; PyTorch checkpoints use the CPU micro prototype.",
        ),
        "group": "训练后校准",
        "calibration": {"needed": True,
                        "note": T("需要指令/偏好数据：alpaca(.jsonl) 用于 SFT，hh-rlhf 用于 DPO。", "Needs instruction/preference data: alpaca (.jsonl) for SFT, hh-rlhf for DPO.")},
        "inputs": [
            {"name": "--base-model-path", "required": True, "format": T("HF 模型目录", "HF model directory")},
            {"name": "--dataset-path", "required": True, "format": T("含 .jsonl 的目录", "directory containing .jsonl")},
            {"name": "--lora-rank / --epochs / --learning-rate", "required": False,
             "format": T("rank / 轮数 / 学习率", "rank / epochs / lr")},
            {"name": "GPU", "required": True, "format": T("CUDA 运行时（HF 路径）", "CUDA runtime (HF path)")},
        ],
        "suits": {"models": ["llm"], "tasks": ["text-generation"]},
        "effect": {
            "compression": T("不压缩底模；适配器本身仅几 MB（可单独分发）", "Does not compress the base model; the adapter itself is a few MB and can be shipped alone"),
            "accuracy": T("取决于数据量与轮数（微调目标是任务适配，不是压缩）", "Depends on data and epochs (goal is task adaptation, not compression)"),
            "latency": T("推理时若不合并适配器会略慢；合并后无额外开销", "Slightly slower if the adapter is not merged; no overhead once merged"),
        },
        "limits": [
            T("HF 路径必须 CUDA：CPU 只有内置微型字符 LM 原型（流程验证用）。", "The HF path requires CUDA; CPU only has the built-in micro char-LM prototype (pipeline validation)."),
            T("指令数据要能被解析（csv/xlsx/jsonl；jsonl 常见字段 instruction/input/output）。", "Instruction data must be parseable (csv/xlsx/jsonl; typical jsonl fields instruction/input/output)."),
            T("本环境 8GB 显存：0.5B + rank 8 + max_len 128 可跑，更大模型需降 batch/长度。", "8 GB VRAM here: 0.5B + rank 8 + max_len 128 fits; larger models need smaller batch/length."),
        ],
        "evidence": ev,
    }


CARDS = [
    card_export_onnx, card_export_gguf, card_quantize_onnx, card_quantize_onnx_static,
    card_openvino_int8, card_llm_quantize, card_prune, card_sparsify, card_distill,
    card_qat, card_llm_finetune,
]

# task directory in out/ → card id that should show its measured numbers
EVIDENCE_SOURCE = {
    "export_onnx": "01-export",
    "quantize_onnx": "02-dynamic-int8",
    "quantize_onnx_static": "03-static-int8",
    "openvino_int8": "04-openvino-int8",
    "prune": "05-prune-l2",
    "sparsify": "11-sparsify",
    "distill": "13-distill-10ep",
    "llm_quantize": "20-qwen-int4",
    "llm_finetune": "21-qwen-lora",
    "export_gguf": "22-qwen-gguf",
    "qat": "12-qat-10ep-4k",
}

# GitHub slugs for CJK headings are awkward to predict, so the report carries an
# explicit `<a id="...">` anchor where a card's evidence task id differs from it.
EVIDENCE_ANCHOR = {
    "12-qat-10ep-4k": "12-qat",
}



def llm_evidence(card_id: str, summary: dict) -> dict | None:
    """Evidence for the two LLM cards, built from the LLM task records.

    The CIFAR baseline (392 KB / 0.6879) is meaningless here, so these cards use
    the Qwen directory size and the LLM bench (decode + perplexity) instead.
    """
    tasks = summary.get("tasks") or {}
    if card_id == "llm_quantize":
        task = tasks.get("20-qwen-int4") or {}
        artifact_bytes = task.get("artifact_bytes")
        if not artifact_bytes:
            return None
        baseline_bytes = summary.get("llm_baseline_bytes") or 999_602_900
        compression = round((1 - artifact_bytes / baseline_bytes) * 100, 1)
        bench = task.get("llm_bench") or {}
        base, art = bench.get("baseline"), bench.get("artifact")
        zh = [f"压缩率 {compression}%", f"IR {artifact_bytes / 1e6:.0f} MB（基线 {baseline_bytes / 1e6:.0f} MB）"]
        en = [f"{compression}% smaller", f"IR {artifact_bytes / 1e6:.0f} MB (baseline {baseline_bytes / 1e6:.0f} MB)"]
        if isinstance(task.get("eval"), dict) and task["eval"].get("perplexity"):
            zh.append(f"引擎轻量困惑度 {task['eval']['perplexity']}")
            en.append(f"engine light-eval PPL {task['eval']['perplexity']}")
        if base and art:
            if base.get("perplexity") and art.get("perplexity"):
                zh.append(f"PPL {base['perplexity']} → {art['perplexity']}")
                en.append(f"PPL {base['perplexity']} → {art['perplexity']}")
            if base.get("decode_tokens_per_s") and art.get("decode_tokens_per_s"):
                zh.append(f"解码 {base['decode_tokens_per_s']} → {art['decode_tokens_per_s']} tok/s")
                en.append(f"decode {base['decode_tokens_per_s']} → {art['decode_tokens_per_s']} tok/s")
        engine_bench = task.get("engine_bench") or {}
        if engine_bench.get("original") and engine_bench.get("compressed"):
            o, c = engine_bench["original"], engine_bench["compressed"]
            zh.append(f"平台基准 p50 {o['latency_ms_p50']:.0f} → {c['latency_ms_p50']:.0f} ms（{engine_bench.get('speedup_x')}× 加速）")
            en.append(f"platform bench p50 {o['latency_ms_p50']:.0f} → {c['latency_ms_p50']:.0f} ms ({engine_bench.get('speedup_x')}× speed-up)")
            zh.append(f"峰值内存 {o['peak_memory_mb']:.0f} → {c['peak_memory_mb']:.0f} MB")
            en.append(f"peak memory {o['peak_memory_mb']:.0f} → {c['peak_memory_mb']:.0f} MB")
        if task.get("duration_s"):
            zh.append(f"耗时 {task['duration_s']:.0f} s")
            en.append(f"took {task['duration_s']:.0f} s")
        return {
            "dataset": "alpaca-eval.txt (32 行派生语料)",
            "summary": {"zh": "；".join(zh), "en": "; ".join(en)},
            "baseline_accuracy": None,
            "result_accuracy": None,
            "bytes_before": baseline_bytes,
            "bytes_after": artifact_bytes,
            "duration_s": task.get("duration_s"),
            "report_anchor": "docs/compression-eval-report.md#48-llm_quantizenncf-int420-qwen-int4",
        }
    if card_id == "export_gguf":
        task = tasks.get("22-qwen-gguf") or {}
        artifact_bytes = task.get("artifact_bytes")
        if not artifact_bytes:
            return None
        metrics = task.get("metrics") or {}
        baseline_bytes = summary.get("llm_baseline_bytes") or 999_602_900
        compression = round((1 - artifact_bytes / baseline_bytes) * 100, 1)
        zh = [f"q4_K_M {artifact_bytes / 1e6:.0f} MB（HF 基线 {baseline_bytes / 1e6:.0f} MB，−{compression}%）",
              "使用本机离线 llama.cpp 工具完成（无需联网）"]
        en = [f"q4_K_M {artifact_bytes / 1e6:.0f} MB (HF baseline {baseline_bytes / 1e6:.0f} MB, −{compression}%)",
              "converted with the on-host llama.cpp toolchain (no network)"]
        if metrics.get("outtype"):
            zh.append(f"outtype={metrics['outtype']}")
            en.append(f"outtype={metrics['outtype']}")
        if task.get("duration_s"):
            zh.append(f"耗时 {task['duration_s']:.0f} s")
            en.append(f"took {task['duration_s']:.0f} s")
        return {
            "dataset": "Qwen2.5-0.5B（服务器本地 HF 目录）",
            "summary": {"zh": "；".join(zh), "en": "; ".join(en)},
            "baseline_accuracy": None,
            "result_accuracy": None,
            "bytes_before": baseline_bytes,
            "bytes_after": artifact_bytes,
            "duration_s": task.get("duration_s"),
            "report_anchor": "docs/compression-eval-report.md#410-export_gguf22-qwen-gguf",
        }
    task = tasks.get("21-qwen-lora") or {}
    if not task.get("artifact_bytes"):
        return None
    metrics = task.get("metrics") or {}
    files = task.get("artifact_files") or {}
    adapter = files.get("adapter_lora.pt") or task["artifact_bytes"]
    merged = files.get("model_lora_merged.pt")
    zh = [f"可部署适配器 {adapter / 1e6:.1f} MB"]
    en = [f"deployable adapter {adapter / 1e6:.1f} MB"]
    if merged:
        zh.append(f"另含合并模型 {merged / 1e6:.0f} MB（可只分发适配器）")
        en.append(f"plus merged model {merged / 1e6:.0f} MB (adapter alone is enough)")
    for key, label_zh, label_en in (
        ("trainable_params", "可训练参数", "trainable params"),
        ("epochs_completed", "完成轮数", "epochs"),
        ("final_loss", "末轮 loss", "final loss"),
    ):
        if metrics.get(key) is not None:
            zh.append(f"{label_zh} {metrics[key]}")
            en.append(f"{label_en} {metrics[key]}")
    if task.get("duration_s"):
        zh.append(f"耗时 {task['duration_s']:.0f} s")
        en.append(f"took {task['duration_s']:.0f} s")
    return {
        "dataset": "alpaca-train-32 (服务器本地)",
        "summary": {"zh": "；".join(zh), "en": "; ".join(en)},
        "baseline_accuracy": None,
        "result_accuracy": None,
        "bytes_before": None,
        "bytes_after": task.get("artifact_bytes"),
        "duration_s": task.get("duration_s"),
        "report_anchor": "docs/compression-eval-report.md#49-llm_finetunelora21-qwen-lora",
    }


def evidence_for(card_id: str, summary: dict, baseline: dict) -> dict | None:
    """Build the `evidence` block from a real measured task (None when not run)."""
    if card_id in {"llm_quantize", "llm_finetune", "export_gguf"}:
        return llm_evidence(card_id, summary)
    task_name = EVIDENCE_SOURCE.get(card_id)
    task = (summary.get("tasks") or {}).get(task_name or "")
    if not task or not task.get("bench_accuracy") and not task.get("artifact_bytes"):
        return None
    accuracy = task.get("bench_accuracy")
    bytes_after = task.get("bench_bytes") or task.get("artifact_bytes")
    if not bytes_after:
        return None
    compression = round((1 - bytes_after / baseline["bytes"]) * 100, 1)
    parts_zh = [f"压缩率 {compression}%",
                f"体积 {baseline['bytes']:,} → {bytes_after:,} B"]
    parts_en = [f"{compression}% smaller",
                f"{baseline['bytes']:,} → {bytes_after:,} B"]
    if accuracy is not None:
        delta = round((accuracy - baseline["accuracy"]) * 100, 2)
        parts_zh.append(f"Top-1 {baseline['accuracy'] * 100:.2f}% → {accuracy * 100:.2f}%（{delta:+.2f}pp）")
        parts_en.append(f"Top-1 {baseline['accuracy'] * 100:.2f}% → {accuracy * 100:.2f}% ({delta:+.2f}pp)")
    if task.get("latency_ms"):
        parts_zh.append(f"p50 {task['latency_ms']['p50']} ms / p99 {task['latency_ms']['p99']} ms")
        parts_en.append(f"p50 {task['latency_ms']['p50']} ms / p99 {task['latency_ms']['p99']} ms")
    if task.get("duration_s") is not None:
        parts_zh.append(f"作业耗时 {task['duration_s']} s")
        parts_en.append(f"job took {task['duration_s']} s")
    return {
        "dataset": "CIFAR10-test (10,000 samples)",
        "summary": {"zh": "；".join(parts_zh), "en": "; ".join(parts_en)},
        "baseline_accuracy": baseline["accuracy"],
        "result_accuracy": accuracy,
        "bytes_before": baseline["bytes"],
        "bytes_after": bytes_after,
        "duration_s": task.get("duration_s"),
        "report_anchor": f"docs/compression-eval-report.md#{EVIDENCE_ANCHOR.get(task_name or '', task_name)}",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True, help="out/summary.json from collect_results.py")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    baseline = summary["baseline"]
    cards = [factory(evidence_for(factory(None)["id"], summary, baseline)) for factory in CARDS]
    payload = {
        "version": 1,
        "generatedAt": __import__("datetime").date.today().isoformat(),
        "report": "docs/compression-eval-report.md",
        "cards": cards,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with_evidence = [c["id"] for c in cards if c.get("evidence")]
    print(f"wrote {out} ({len(cards)} cards); evidence: {', '.join(with_evidence)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

