/**
 * components/lab/methodAdvisor.ts · 压缩方法推荐规则引擎（纯函数，可测试）
 *
 * 输入：任务场景 / 模型 / 数据 / 部署指标；输出：按匹配度排序的方法建议 + 理由 + 注意事项 + 可直接执行的命令。
 *
 * 每条规则都对应 docs/compression-eval-report.md 里的**实测结论**（见 docs/compression-advisor.md 的规则表），
 * 不是拍脑袋的启发式；卡片文案与实测证据来自 ./method-cards.json（单一数据源）。
 */
import { METHOD_CARDS, type MethodCardData } from "./MethodCards"

export type AdvisorTask =
  | "image-classification"
  | "object-detection"
  | "text-generation"
  | "tabular-classification"
  | "point-cloud"

export type AdvisorModelKind = "cnn" | "yolo" | "transformer" | "llm" | "tabular" | "pointnet"

export type AdvisorData = "none" | "small-images" | "full-images" | "instruction" | "preference"

export type AdvisorDevice = "cpu" | "intel-cpu" | "gpu" | "edge"

export interface AdvisorInput {
  task: AdvisorTask
  modelKind: AdvisorModelKind
  /** 模型体积（MB），用于判断"是否需要大幅压缩" */
  modelMb: number
  data: AdvisorData
  device: AdvisorDevice
  /** 可接受的最大精度损失（百分点），0 表示不允许掉点 */
  accuracyBudgetPp: number
  /** 目标压缩率（%），0 表示不设硬指标 */
  targetCompressionPct: number
  /** 是否要求推理提速（低延迟） */
  needSpeed: boolean
  /** 是否需要 llama.cpp / ollama 生态可直接加载 */
  needLlamaCpp: boolean
  /** 是否要给模型"补领域能力"（任务适配而非压缩） */
  needTaskAdaptation: boolean
}

export interface AdvisorRecommendation {
  card: MethodCardData
  score: number
  role: "primary" | "alternative" | "avoid"
  reasons: string[]
  caveats: string[]
  command: string
}

function card(id: string): MethodCardData {
  const found = METHOD_CARDS.find((item) => item.id === id)
  if (!found) throw new Error(`unknown method card: ${id}`)
  return found
}

const HAS_IMAGES = new Set<AdvisorData>(["small-images", "full-images"])

function commandFor(id: string, input: AdvisorInput): string {
  const base = "python /app/compress_job.py --job-id=1001 --tenant-id=t1 --trace-id=abc"
  const model = input.modelKind === "llm" ? "/models/Qwen2.5-0.5B" : "/models/model.pt"
  const arch =
    input.modelKind === "llm" ? "" : " \\\n  --arch-file=/models/arch.py"
  const data = input.modelKind === "llm" ? "" : `${arch} \\\n  --dataset-path=/datasets/CIFAR10-train`
  switch (id) {
    case "quantize_onnx":
      return `${base} \\\n  --base-model-path=${model.replace(/\\.pt$/, ".onnx")}${arch} \\\n  --output-dir=/out/1001 --method=quantize --algo=onnx_dynamic_int8`
    case "quantize_onnx_static":
      return `${base} \\\n  --base-model-path=${model.replace(/\\.pt$/, ".onnx")}${data} \\\n  --max-samples=512 --output-dir=/out/1001 --method=quantize --algo=onnx_static_int8`
    case "openvino_int8":
      return `${base} \\\n  --base-model-path=${model.replace(/\\.pt$/, ".onnx")}${data} \\\n  --max-samples=512 --output-dir=/out/1001 --method=quantize --algo=openvino_int8 --precision=int8`
    case "prune":
      return `${base} \\\n  --base-model-path=${model}${data} \\\n  --output-dir=/out/1001 --method=prune --algo=l2 --rate=0.3`
    case "sparsify":
      return `${base} \\\n  --base-model-path=${model} \\\n  --output-dir=/out/1001 --method=prune --algo=sparsify --rate=0.3`
    case "distill":
      return `${base} \\\n  --base-model-path=${model}${data} \\\n  --epochs=10 --output-dir=/out/1001 --method=distill --algo=logits`
    case "qat":
      return `${base} \\\n  --base-model-path=${model}${data} \\\n  --epochs=1 --batch-size=64 --output-dir=/out/1001 --method=qat --algo=fx_int8`
    case "llm_quantize":
      return `${base} \\\n  --base-model-path=/models/Qwen2.5-0.5B \\\n  --output-dir=/out/1001 --method=quantize --algo=nncf_int4 --bits=4 --group-size=128`
    case "llm_finetune":
      return `${base} \\\n  --base-model-path=/models/Qwen2.5-0.5B \\\n  --dataset-path=/datasets/alpaca-train \\\n  --output-dir=/out/1001 --method=finetune --algo=lora --lora-rank=8 --epochs=1 --device=cuda`
    case "export_gguf":
      return `${base} \\\n  --base-model-path=/models/Qwen2.5-0.5B \\\n  --llama-cpp-path=/opt/llama.cpp --output-dir=/out/1001 --method=export --algo=gguf --outtype=q4_K_M`
    case "pipeline-prune-quant":
      return `${base} \\\n  --base-model-path=${model}${data} \\\n  --output-dir=/out/1001 --method=pipeline --pipeline="prune:l2,quantize:onnx_dynamic_int8" --rate=0.3`
    default:
      return `${base} \\\n  --base-model-path=${model} \\\n  --output-dir=/out/1001 --method=export --algo=onnx`
  }
}

interface Draft {
  card: MethodCardData
  score: number
  reasons: string[]
  caveats: string[]
}

export function recommend(input: AdvisorInput): AdvisorRecommendation[] {
  const drafts: Draft[] = []
  const hasImages = HAS_IMAGES.has(input.data)
  const isLlm = input.modelKind === "llm" || input.task === "text-generation"
  const isVision = input.modelKind === "cnn" || input.modelKind === "yolo" || input.modelKind === "pointnet"
  const tightAccuracy = input.accuracyBudgetPp <= 0.5
  const bigCompression = input.targetCompressionPct >= 60

  const add = (id: string, score: number, reasons: string[], caveats: string[] = []) => {
    drafts.push({ card: card(id), score, reasons, caveats })
  }

  // ---------------------------------------------------------------- LLM 场景
  if (isLlm) {
    add("llm_quantize", 95, [
      "LLM 权重压缩免校准数据，实测体积 −66%（999.6 MB → 339 MB）、平台基准 2.16× 加速、峰值内存 −47.8%",
      "OpenVINO IR 由平台自有 runner 直接加载，部署链路已验证",
    ], input.device === "edge" ? ["产物是 OpenVINO IR 目录，边缘设备需装 OpenVINO Runtime"] : [])
    if (input.needLlamaCpp) {
      add("export_gguf", 92, [
        "要 llama.cpp / ollama 生态直接加载时，必须走 GGUF；q4_K_M 通常 4~5× 压缩且困惑度损失 <1%",
        "本环境已用离线 llama.cpp 工具跑通（无网络依赖）",
      ], ["GGUF 量化在 CPU 上以内存带宽为主，吞吐提升不如 OpenVINO INT4"])
    }
    if (input.needTaskAdaptation) {
      add("llm_finetune", 90, [
        "目标是任务适配（指令/偏好对齐）而非压缩：LoRA rank 8 只训练 1.08 M 参数，适配器仅 4.4 MB",
        "实测 32 样本 1 epoch 即 loss 1.277 → 0.838，GPU 19.2 s",
      ], ["HF LoRA 需要 CUDA（CPU 镜像会明确报 requires a CUDA runtime）"])
    }
    if (!hasImages && input.data === "none") {
      add("quantize_onnx", 40, ["若已导出 ONNX 且只需要权重压缩，动态 INT8 免数据、体积最小"], ["LLM 场景 CPU 上不如 INT4 IR 快"])
    }
    return finish(drafts, input)
  }

  // ------------------------------------------------------------- 视觉/表格场景
  const onnxReady = input.modelKind !== "llm"

  if (hasImages && (input.device === "cpu" || input.device === "intel-cpu") && onnxReady) {
    add("quantize_onnx_static", 96, [
      "有按类图片即可做静态 INT8：本批实测精度损失最小（68.79% → 68.68%，−0.11pp）",
      "延迟最低（p50 0.079 ms / p99 0.247 ms）且体积 −72.4%，是图像分类/检测的首选",
    ], ["需要代表性校准集（本次 512 张即可），分布不匹配会明显掉点"])
  }
  if (onnxReady) {
    add("quantize_onnx", hasImages ? 82 : 92, [
      "免校准数据，一条命令把 FP32 ONNX 压到约 1/4（实测 −73.7%），精度损失 0.52pp",
      hasImages ? "可作为无数据时的兜底或与静态量化做对比" : "当前数据条件下它就是最省事的最优解",
    ], [
      "CPU 上未量化激活，实测 p50 比静态 INT8 慢约 5×（0.440 vs 0.079 ms）",
      input.modelKind === "pointnet" ? "PointNet 的算子图对 ORT 量化支持有限，先小样本验证" : "",
    ].filter(Boolean))
  }
  if (input.device === "intel-cpu" && isVision && hasImages) {
    add("openvino_int8", 78, [
      "Intel CPU + IR 部署时 NNCF INT8 可利用专用后端；本批实测 −68.1%",
    ], [
      "本批精度损失最大（−1.78pp）且峰值内存最高（367 MB），非 Intel 平台收益有限",
    ])
  }
  if (!isLlm && input.data === "none" && input.modelKind === "cnn") {
    add("prune", 55, [
      "无数据可用 L1/L2 结构化剪枝（不需要校准集），参数可减半（94,986 → 45,915）",
    ], [
      "⚠️ 实测未微调时精度从 68.79% 崩到 20.1%（rate 0.3）/ 17.4%（rate 0.1），分类头重映射是主因——必须接 QAT/微调",
      "只支持含 Conv2d 的 CNN，且必须提供 arch.py",
    ])
  }
  if (isVision && hasImages && bigCompression) {
    add("pipeline-prune-quant", 85, [
      `目标压缩率 ${input.targetCompressionPct}% 时单靠量化不够：剪枝+量化流水线实测 −86.2%（54,159 B）`,
      "体积指标最激进，但精度继承剪枝损失，需要后续微调补偿",
    ], ["必须先看剪枝那一版的精度：实测裸剪枝只有 0.20 左右"])
  }
  if (isVision && hasImages && tightAccuracy) {
    add("qat", 88, [
      `精度预算${input.accuracyBudgetPp}pp 很紧时，QAT（FX 训练中模拟量化）通常比训练后量化高 0.5~2pp`,
      "CLI 已支持：--method=qat --algo=fx_int8|lsq",
    ], ["需要训练时间与算力，CPU 上建议先 --max-samples 小样本试跑"])
  }
  if (isVision && hasImages && !bigCompression) {
    add("distill", 60, [
      "有小模型落地需求时可把教师能力蒸馏进更小学生（10 epoch 实测把随机学生从 8.7% 拉到 61.3%）",
    ], [
      "本平台 width_mult 对 VGG-ish 顺序网络未生效（学生参数量=教师），想压缩请用剪枝/量化",
      "训练成本高（3 epoch 258 s、10 epoch 681 s，CPU）",
    ])
  }
  if (input.data === "none" && input.modelKind !== "llm") {
    add("export_onnx", 45, [
      "没有任何标注数据时，至少先导出 ONNX（无精度损失，实测图等价），后续随时可加量化",
    ], ["本身不压缩体积（FP32 精度不变）"])
  }
  if (isVision && input.data === "none") {
    add("sparsify", 30, [
      "无数据时可做幅值稀疏化，但只是中间态",
    ], ["实测 30% 稀疏度体积仅 −3.1%、精度 −19.8pp，且稠密 CPU 推理不提速"])
  }
  return finish(drafts, input)
}

function finish(drafts: Draft[], input: AdvisorInput): AdvisorRecommendation[] {
  const ranked = drafts
    .map((draft) => {
      let score = draft.score
      // 硬性偏好：精度预算越紧，越偏向 QAT/静态量化；目标压缩率越高，越偏向流水线/量化
      if (input.accuracyBudgetPp <= 0.5 && ["qat", "quantize_onnx_static"].includes(draft.card.id)) {
        score += 4
      }
      if (input.targetCompressionPct >= 60 && ["pipeline-prune-quant", "quantize_onnx", "llm_quantize"].includes(draft.card.id)) {
        score += 3
      }
      if (input.needSpeed && ["quantize_onnx_static", "openvino_int8", "llm_quantize"].includes(draft.card.id)) {
        score += 3
      }
      if (!input.needSpeed && draft.card.id === "quantize_onnx_static") {
        score += 1
      }
      return { ...draft, score }
    })
    .sort((a, b) => b.score - a.score)

  return ranked.map((draft, index) => ({
    card: draft.card,
    score: draft.score,
    role: index === 0 ? "primary" : "alternative",
    reasons: draft.reasons,
    caveats: draft.caveats,
    command: commandFor(draft.card.id, input),
  }))
}

export const ADVISOR_DEFAULT: AdvisorInput = {
  task: "image-classification",
  modelKind: "cnn",
  modelMb: 0.4,
  data: "full-images",
  device: "cpu",
  accuracyBudgetPp: 1,
  targetCompressionPct: 60,
  needSpeed: true,
  needLlamaCpp: false,
  needTaskAdaptation: false,
}
