/**
 * tests/unit/methodAdvisor.test.ts · 推荐规则引擎单测（纯函数，无需浏览器/后端）
 *
 *   cd frontend && bun test            # 全部单测
 *   cd frontend && npm run test:unit   # 同上（脚本别名）
 *
 * 断言覆盖：LLM/视觉两条主线、数据条件分水岭、精度与压缩率加权、Intel 分支、
 * 命令生成正确性，以及「推荐引用的方法卡必须真实存在」这一数据一致性不变量。
 */
import { describe, expect, test } from "bun:test"

import { METHOD_CARDS } from "@/components/lab/MethodCards"
import {
  ADVISOR_DEFAULT,
  type AdvisorInput,
  adviseForWizard,
  recommend,
  wizardInput,
} from "@/components/lab/methodAdvisor"

const base: AdvisorInput = { ...ADVISOR_DEFAULT }

const ids = (input: Partial<AdvisorInput>) =>
  recommend({ ...base, ...input }).map((item) => item.card.id)

describe("LLM 场景", () => {
  test("无数据时首选 llm_quantize（免校准）", () => {
    const result = recommend({ ...base, modelKind: "llm", task: "text-generation", data: "none" })
    expect(result[0].card.id).toBe("llm_quantize")
    expect(result[0].role).toBe("primary")
    expect(result[0].reasons.join(" ")).toContain("免校准")
  })

  test("需要 llama.cpp/ollama 时加入 export_gguf", () => {
    const result = ids({ modelKind: "llm", task: "text-generation", data: "none", needLlamaCpp: true })
    expect(result).toContain("export_gguf")
  })

  test("需要任务适配时加入 llm_finetune 并提示 CUDA 限制", () => {
    const result = recommend({
      ...base, modelKind: "llm", task: "text-generation", data: "instruction", needTaskAdaptation: true,
    })
    const finetune = result.find((item) => item.card.id === "llm_finetune")
    expect(finetune).toBeDefined()
    expect(finetune?.caveats.join(" ")).toContain("CUDA")
  })
})

describe("视觉场景", () => {
  test("有完整图片集 + CPU 时首选静态 INT8", () => {
    const result = recommend({ ...base, data: "full-images", device: "cpu", needSpeed: true })
    expect(result[0].card.id).toBe("quantize_onnx_static")
    expect(result[0].reasons.join(" ")).toContain("0.079")
  })

  test("无数据时不推荐静态量化（需校准集），但推荐免数据的动态量化和导出", () => {
    const result = ids({ data: "none", modelKind: "cnn" })
    expect(result).not.toContain("quantize_onnx_static")
    expect(result).toContain("quantize_onnx")
    expect(result).toContain("export_onnx")
    expect(result).toContain("prune")
    expect(result).toContain("sparsify")
  })

  test("无数据的剪枝建议必须带「不微调会崩精度」警告", () => {
    const prune = recommend({ ...base, data: "none", modelKind: "cnn" })
      .find((item) => item.card.id === "prune")
    expect(prune?.caveats.join(" ")).toContain("崩到 20.1%")
  })

  test("Intel CPU + 有图片时给出 OpenVINO 选项", () => {
    expect(ids({ data: "small-images", device: "intel-cpu" })).toContain("openvino_int8")
    expect(ids({ data: "small-images", device: "cpu" })).not.toContain("openvino_int8")
  })

  test("目标压缩率 ≥60% 时加入剪枝+量化流水线", () => {
    expect(ids({ data: "full-images", targetCompressionPct: 80 })).toContain("pipeline-prune-quant")
  })

  test("精度预算 ≤0.5pp 时 QAT 排名高于动态量化", () => {
    const result = ids({ data: "full-images", accuracyBudgetPp: 0.2 })
    expect(result.indexOf("qat")).toBeGreaterThanOrEqual(0)
    expect(result.indexOf("qat")).toBeLessThan(result.indexOf("quantize_onnx"))
  })

  test("宽松压缩目标下给出蒸馏，并说明 width_mult 不生效", () => {
    const distill = recommend({ ...base, data: "full-images", targetCompressionPct: 30 })
      .find((item) => item.card.id === "distill")
    expect(distill).toBeDefined()
    expect(distill?.caveats.join(" ")).toContain("width_mult")
  })
})

describe("输出契约", () => {
  test("全参数矩阵下都能解析出方法卡、给出理由与命令（回归：组合卡缺失曾直接抛错）", () => {
    const modelKinds = ["cnn", "yolo", "transformer", "llm", "tabular", "pointnet"] as const
    const dataKinds = ["none", "small-images", "full-images", "instruction", "preference"] as const
    const devices = ["cpu", "intel-cpu", "gpu", "edge"] as const
    const platformIds = new Set(METHOD_CARDS.map((card) => card.id))

    let checked = 0
    for (const modelKind of modelKinds) {
      for (const data of dataKinds) {
        for (const device of devices) {
          // 最激进的一档：既要求高压缩率，又打开所有可选诉求
          const result = recommend({
            ...base, modelKind, data, device,
            targetCompressionPct: 90, accuracyBudgetPp: 0.1,
            needSpeed: true, needLlamaCpp: true, needTaskAdaptation: true,
          })
          expect(result.length).toBeGreaterThan(0)
          expect(result[0].role).toBe("primary")
          for (const item of result) {
            // 引用平台方法的必须命中 method-cards.json；组合卡是本模块自带的实测卡
            expect(platformIds.has(item.card.id) || item.card.id === "pipeline-prune-quant").toBe(true)
            expect(item.card.label.zh.length).toBeGreaterThan(0)
            expect(item.card.summary.zh.length).toBeGreaterThan(0)
            expect(item.reasons.length).toBeGreaterThan(0)
            expect(item.command).toContain("--method=")
            expect(item.command).toContain("--output-dir=")
          }
          checked += 1
        }
      }
    }
    expect(checked).toBe(modelKinds.length * dataKinds.length * devices.length)
  })

  test("相同输入结果稳定（纯函数、可复现）", () => {
    const first = JSON.stringify(ids({ modelKind: "llm", data: "none" }))
    const second = JSON.stringify(ids({ modelKind: "llm", data: "none" }))
    expect(first).toBe(second)
  })

  test("分数单调不增（排序正确）", () => {
    const scores = recommend(base).map((item) => item.score)
    for (let i = 1; i < scores.length; i += 1) {
      expect(scores[i - 1]).toBeGreaterThanOrEqual(scores[i])
    }
  })
})

describe("向导集成 adviseForWizard", () => {
  test("HF/LLM 模型 + 已选数据集 → 首选是 LLM 压缩；选 llm_quantize 判定为推荐内", () => {
    const hint = adviseForWizard("llm_quantize", {
      framework: "huggingface", modelName: "Qwen2.5-0.5B", modelBytes: 999_600_000, hasDataset: true,
    })
    expect(hint.input.modelKind).toBe("llm")
    expect(hint.input.task).toBe("text-generation")
    expect(hint.input.data).toBe("instruction")
    expect(hint.primary.card.id).toBe("llm_quantize")
    expect(hint.selectedRecommended).toBe(true)
    expect(hint.warning).toBe("")
  })

  test("模型名兜底识别 LLM（framework 缺失时也能判对）", () => {
    expect(wizardInput({ modelName: "Qwen2.5-0.5B" }).modelKind).toBe("llm")
    expect(wizardInput({ modelName: "CifarCNN10" }).modelKind).toBe("cnn")
    expect(wizardInput({ framework: "onnx" }).modelKind).toBe("cnn")
  })

  test("CNN + 已选数据集 → 首选静态 INT8，且该选择判定为推荐内", () => {
    const hint = adviseForWizard("quantize_onnx_static", {
      framework: "pytorch", modelName: "CifarCNN10", modelBytes: 392_548, hasDataset: true,
    })
    expect(hint.input.data).toBe("full-images")
    expect(hint.primary.card.id).toBe("quantize_onnx_static")
    expect(hint.selectedRecommended).toBe(true)
  })

  test("没选数据集时选静态量化 → 明确提示缺校准集", () => {
    const hint = adviseForWizard("quantize_onnx_static", { framework: "pytorch", hasDataset: false })
    expect(hint.input.data).toBe("none")
    expect(hint.selectedRecommended).toBe(false)
    expect(hint.warning).toContain("校准集")
    expect(hint.primary.card.id).not.toBe("quantize_onnx_static")
  })

  test("没选数据集时选 QAT / 蒸馏 → 提示缺训练集", () => {
    expect(adviseForWizard("qat", { hasDataset: false }).warning).toContain("训练")
    expect(adviseForWizard("distill", { hasDataset: false }).warning).toContain("训练")
  })

  test("CNN 上选 LLM 专用方法 → 提示面向 LLM", () => {
    const hint = adviseForWizard("llm_quantize", { framework: "pytorch", hasDataset: true })
    expect(hint.selectedRecommended).toBe(false)
    expect(hint.warning).toContain("LLM")
  })

  test("LLM 上选 CNN 专用方法 → 提示需 arch.py", () => {
    const hint = adviseForWizard("prune", { framework: "huggingface", hasDataset: true })
    expect(hint.warning).toContain("arch.py")
  })

  test("组合方法（流水线）会被算进推荐；未推荐方法给出说明而不是抛错，备选最多 2 条", () => {
    // CNN + 完整图片 + 默认 60% 压缩目标 → 剪枝+量化流水线确实在推荐列表里
    const pipeline = adviseForWizard("pipeline-prune-quant", { framework: "pytorch", hasDataset: true })
    expect(pipeline.selectedRecommended).toBe(true)
    expect(pipeline.warning).toBe("")
    expect(pipeline.alternatives.length).toBeLessThanOrEqual(2)

    // 通用 CPU 上的 OpenVINO 分支不触发 → 给出通用说明
    const openvino = adviseForWizard("openvino_int8", { framework: "pytorch", hasDataset: true })
    expect(openvino.selectedRecommended).toBe(false)
    expect(openvino.warning.length).toBeGreaterThan(0)
  })

  test("向导可见的每个方法枚举都能得到判定（与 platform 枚举同源）", () => {
    for (const id of METHOD_CARDS.map((card) => card.id)) {
      const hint = adviseForWizard(id, { framework: "pytorch", hasDataset: true })
      expect(hint.primary.card.id.length).toBeGreaterThan(0)
      expect(typeof hint.selectedRecommended).toBe("boolean")
    }
  })
})
