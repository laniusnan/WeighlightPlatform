/**
 * routes/lab/selfcheck.tsx · 实验自检（一键运行真实 job 链路的冒烟实验）
 */
import { createFileRoute } from "@tanstack/react-router"
import { FlaskConical, Loader2, Play } from "lucide-react"
import { useState } from "react"

import { useI18n } from "@/components/lab/i18n"
import { Button, Card, Pill } from "@/components/lab/ui"
import { apiSelfcheckRun } from "@/services/api"
import { apiErrorMessage } from "@/services/http"

export const Route = createFileRoute("/lab/selfcheck")({
  component: SelfCheckPage,
  head: () => ({ meta: [{ title: "Self-check - AI Model Lightweight Platform" }] }),
})

interface CheckDef {
  key: string
  zh: string
  en: string
  descriptionZh: string
  descriptionEn: string
  run: () => Promise<Record<string, unknown>>
}

function SelfCheckPage() {
  const { t } = useI18n()
  const [running, setRunning] = useState(false)
  const [result, setResult] = useState<Record<string, unknown> | null>(null)
  const [error, setError] = useState("")

  const checks: CheckDef[] = [
    {
      key: "fitnets_distill",
      zh: "FitNets 知识蒸馏冒烟（自动生成微型教师 + 数据集）",
      en: "FitNets distillation smoke (auto teacher + dataset)",
      descriptionZh:
        "验证 上传模型→蒸馏(method=features)→产物 onnx 的真实算法链路，无需准备任何数据。",
      descriptionEn: "Exercises the real distill(FitNets) job path end to end.",
      run: () => apiSelfcheckRun("fitnets_distill"),
    },
    {
      key: "distill_logits",
      zh: "知识蒸馏 logits 冒烟（Hinton 软标签）",
      en: "Logits KD (Hinton) smoke",
      descriptionZh: "同一套自动生成教师+数据集，走 method=logits 训练 1 epoch 并导出 ONNX。",
      descriptionEn: "Runs distill(method=logits) end to end.",
      run: () => apiSelfcheckRun("distill_logits"),
    },
    {
      key: "prune_l1",
      zh: "结构化剪枝 L1 冒烟",
      en: "Structured prune (L1) smoke",
      descriptionZh: "对自动生成的 CNN 教师执行 L1 结构化剪枝（rate=0.25）并输出裁剪后模型。",
      descriptionEn: "Runs prune(algorithm=l1) end to end.",
      run: () => apiSelfcheckRun("prune_l1"),
    },
    {
      key: "prune_group_fisher",
      zh: "结构化剪枝 Group Fisher 冒烟",
      en: "Structured prune (Group Fisher) smoke",
      descriptionZh: "用自动生成的微型数据估计 Fisher 信息并执行 Group Fisher 剪枝（rate=0.25）。",
      descriptionEn: "Runs prune(algorithm=group_fisher) with auto Fisher data.",
      run: () => apiSelfcheckRun("prune_group_fisher"),
    },
    {
      key: "prune_autoslim",
      zh: "结构化剪枝 AutoSlim 冒烟",
      en: "Structured prune (AutoSlim) smoke",
      descriptionZh: "微型数据上做稀疏训练后按 BN γ 剪枝（rate=0.25）。",
      descriptionEn: "Runs prune(algorithm=autoslim) with sparse training.",
      run: () => apiSelfcheckRun("prune_autoslim"),
    },
    {
      key: "prune_dmcp",
      zh: "结构化剪枝 DMCP 冒烟",
      en: "Structured prune (DMCP) smoke",
      descriptionZh: "用一阶 Taylor（|权重×梯度|）近似通道重要性并剪枝（rate=0.25）。",
      descriptionEn: "Runs prune(algorithm=dmcp) with Taylor probe.",
      run: () => apiSelfcheckRun("prune_dmcp"),
    },
    {
      key: "qat_lsq",
      zh: "QAT/LSQ 冒烟",
      en: "QAT with LSQ learnable-step smoke",
      descriptionZh: "训练 Conv/Linear 可学习步长并落为全 INT8 QDQ（权重+激活校准），含 ORT 输出一致性校验。",
      descriptionEn: "Runs qat(scheme=lsq) end to end (full INT8 QDQ).",
      run: () => apiSelfcheckRun("qat_lsq"),
    },
    {
      key: "llm_finetune",
      zh: "LoRA 微调（CPU 原型）冒烟",
      en: "LoRA SFT (CPU prototype) smoke",
      descriptionZh: "冻结内置微型因果 LM，仅训练注入的 LoRA A/B，秒级验证 数据→LoRA→合并产物。",
      descriptionEn: "Runs llm_finetune end to end on demo QA data.",
      run: () => apiSelfcheckRun("llm_finetune"),
    },
    {
      key: "llm_finetune_excel",
      zh: "Excel/CSV → 指令集 → LoRA 冒烟",
      en: "CSV/Excel instruction set to LoRA smoke",
      descriptionZh: "解析两列指令文件（问题/回答）后直接进入 LoRA SFT。",
      descriptionEn: "Parses CSV/Excel instruction data then runs llm_finetune.",
      run: () => apiSelfcheckRun("llm_finetune_excel"),
    },
    {
      key: "llm_finetune_jsonl",
      zh: "JSONL 多轮/偏好 → 指令集冒烟",
      en: "JSONL chat/preference to instruction smoke",
      descriptionZh: "解析 messages 多轮对话与 preference(chosen) 数据后进入 LoRA SFT。",
      descriptionEn: "Parses JSONL chat/preference datasets then runs llm_finetune.",
      run: () => apiSelfcheckRun("llm_finetune_jsonl"),
    },
    {
      key: "llm_finetune_gpu",
      zh: "GPU HF LoRA（Qwen2.5-0.5B）冒烟",
      en: "GPU HF LoRA (Qwen2.5-0.5B) smoke",
      descriptionZh: "CUDA 主机上对真实 Qwen 快照跑 LoRA SFT；CPU-only 主机自动跳过。",
      descriptionEn: "Runs HF LoRA SFT on Qwen2.5-0.5B when CUDA is present.",
      run: () => apiSelfcheckRun("llm_finetune_gpu"),
    },
  ]

  const runCheck = async (check: CheckDef) => {
    setRunning(true)
    setError("")
    setResult(null)
    try {
      const res = await check.run()
      setResult(res)
    } catch (e) {
      setError(apiErrorMessage(e))
    } finally {
      setRunning(false)
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <p className="lb-section">{t("实验自检", "Self-check lab")}</p>
        <h1 className="mt-1 flex items-center gap-2 text-2xl font-bold tracking-tight text-[var(--lb-text)]">
          <FlaskConical size={22} className="text-[var(--lb-primary)]" />
          {t("实验 / 自检", "Experiments / Self-check")}
        </h1>
        <p className="lb-muted mt-1 text-sm">
          {t(
            "一键运行内部冒烟实验：走真实的 job 代码路径，但不需要上传模型/数据集，也不会写入任务列表。",
            "Run internal smoke experiments that exercise the real job code paths - no uploads, no task rows.",
          )}
        </p>
      </div>

      <Card className="p-5">
        {checks.map((check) => (
          <div
            key={check.key}
            className="flex flex-wrap items-center justify-between gap-4 rounded-lg border border-[var(--lb-border)] bg-[var(--lb-surface-2)] p-4"
          >
            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-2">
                <p className="text-sm font-semibold text-[var(--lb-text)]">
                  {t(check.zh, check.en)}
                </p>
                <Pill tone="info">{check.key}</Pill>
              </div>
              <p className="lb-muted mt-1 text-[13px]">
                {t(check.descriptionZh, check.descriptionEn)}
              </p>
            </div>
            <Button onClick={() => void runCheck(check)} disabled={running}>
              {running ? <Loader2 size={14} className="animate-spin" /> : <Play size={14} />}
              {running ? t("运行中…", "Running…") : t("一键运行", "Run")}
            </Button>
          </div>
        ))}
      </Card>

      {error && (
        <div className="rounded-md border border-[var(--lb-danger)]/30 bg-[var(--lb-danger-soft)] px-4 py-2.5 text-sm text-[var(--lb-danger)]">
          {error}
        </div>
      )}

      {result && (
        <Card className="p-5">
          <p className="mb-2 text-sm font-semibold text-[var(--lb-text)]">
            {t("运行结果", "Result")}
            {" · "}
            {result.ok ? (
              <span className="text-[var(--lb-success)]">{t("通过", "PASS")}</span>
            ) : (
              <span className="text-[var(--lb-danger)]">{t("失败", "FAIL")}</span>
            )}
          </p>
          <pre className="lb-mono overflow-x-auto rounded-lg border border-[var(--lb-border)] bg-[var(--lb-surface-2)] p-4 text-[12.5px] leading-relaxed text-[var(--lb-text-2)]">
            {JSON.stringify(result, null, 2)}
          </pre>
        </Card>
      )}
    </div>
  )
}
