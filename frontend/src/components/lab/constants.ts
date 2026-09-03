/**
 * components/lab/constants.ts
 * 压缩方法 / 框架 / 状态的中文元数据（仅展示层，与真实枚举对齐）。
 */
import type { CompressionMethod, ModelFramework, TaskStatus } from "@/services/types"
import type { PillTone } from "./ui"

export interface MethodMeta {
  value: CompressionMethod
  label: string
  description: string
  group: "格式转换" | "量化" | "剪枝 / 稀疏化" | "训练后校准"
}

export const METHOD_META: MethodMeta[] = [
  {
    value: "export_onnx",
    label: "导出 ONNX",
    description: "PyTorch 转 ONNX（onnxsim 简化）",
    group: "格式转换",
  },
  {
    value: "export_gguf",
    label: "GGUF 导出",
    description: "LLM 转 GGUF（llama.cpp 生态）",
    group: "格式转换",
  },
  {
    value: "quantize_onnx",
    label: "ONNX 动态 INT8",
    description: "onnxruntime 动态量化，免校准数据",
    group: "量化",
  },
  {
    value: "quantize_onnx_static",
    label: "ONNX 静态 INT8",
    description: "QDQ 静态量化，需校准数据集",
    group: "量化",
  },
  {
    value: "openvino_int8",
    label: "OpenVINO 转换",
    description: "OpenVINO IR + NNCF INT8 / FP16",
    group: "量化",
  },
  {
    value: "llm_quantize",
    label: "LLM INT4/INT8",
    description: "HuggingFace LLM 导出 OpenVINO IR",
    group: "量化",
  },
  {
    value: "prune",
    label: "L2 结构化剪枝",
    description: "按 L2 范数剪除卷积滤波器",
    group: "剪枝 / 稀疏化",
  },
  {
    value: "sparsify",
    label: "权重稀疏化",
    description: "按幅值零化最小权重",
    group: "剪枝 / 稀疏化",
  },
  {
    value: "qat",
    label: "量化感知训练 QAT",
    description: "FX QAT 微调后导出量化 ONNX",
    group: "训练后校准",
  },
]


export const FRAMEWORK_LABEL: Record<ModelFramework, string> = {
  pytorch: "PyTorch",
  onnx: "ONNX",
  huggingface: "HuggingFace",
  tensorflow: "TensorFlow",
}

export const TASK_STATUS_META: Record<
  TaskStatus,
  { label: string; tone: PillTone }
> = {
  pending: { label: "排队中", tone: "pending" },
  running: { label: "运行中", tone: "running" },
  succeeded: { label: "已完成", tone: "succeeded" },
  failed: { label: "失败", tone: "failed" },
  canceled: { label: "已取消", tone: "canceled" },
}

export const TASK_STATUS_ORDER: TaskStatus[] = [
  "pending",
  "running",
  "succeeded",
  "failed",
  "canceled",
]

/* ---------------------- 双语标签（EN） ---------------------- */
export const METHOD_LABEL_EN: Record<CompressionMethod, string> = {
  export_onnx: "Export ONNX",
  export_gguf: "Export GGUF",
  quantize_onnx: "ONNX Dynamic INT8",
  quantize_onnx_static: "ONNX Static INT8",
  openvino_int8: "OpenVINO Convert",
  llm_quantize: "LLM INT4/INT8",
  prune: "L2 Structured Prune",
  sparsify: "Weight Sparsify",
  qat: "QAT",
}

export const METHOD_DESC_EN: Record<CompressionMethod, string> = {
  export_onnx: "PyTorch to ONNX with onnxsim simplification",
  export_gguf: "LLM to GGUF for llama.cpp",
  quantize_onnx: "onnxruntime dynamic INT8, no calibration",
  quantize_onnx_static: "QDQ static INT8, requires calibration",
  openvino_int8: "OpenVINO IR + NNCF INT8 / FP16",
  llm_quantize: "HuggingFace LLM to OpenVINO IR",
  prune: "Prune conv filters by L2 norm",
  sparsify: "Zero smallest weights by magnitude",
  qat: "FX QAT fine-tune then export",
}

export const TASK_STATUS_LABEL_EN: Record<TaskStatus, string> = {
  pending: "Queued",
  running: "Running",
  succeeded: "Completed",
  failed: "Failed",
  canceled: "Canceled",
}

import { getCurrentLocale } from "./i18n"

export const methodLabel = (m: CompressionMethod): string => {
  const meta = METHOD_META.find((x) => x.value === m)
  if (!meta) return m
  return getCurrentLocale() === "en" ? METHOD_LABEL_EN[m] : meta.label
}

export const methodDescription = (m: CompressionMethod): string => {
  const meta = METHOD_META.find((x) => x.value === m)
  if (!meta) return ""
  return getCurrentLocale() === "en" ? METHOD_DESC_EN[m] : meta.description
}

export const taskStatusLabel = (status: TaskStatus): string =>
  getCurrentLocale() === "en"
    ? TASK_STATUS_LABEL_EN[status]
    : TASK_STATUS_META[status].label
