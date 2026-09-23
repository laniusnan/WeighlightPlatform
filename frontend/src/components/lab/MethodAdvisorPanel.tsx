/**
 * components/lab/MethodAdvisorPanel.tsx · 「方法推荐」面板
 *
 * 输入任务场景 / 模型 / 数据 / 部署指标 → 输出排序后的方法建议 + 理由 + 注意事项 + 可执行命令。
 * 全部规则与文案来自 ./methodAdvisor.ts（纯函数）与 ./method-cards.json（实测证据）。
 */
import { ClipboardCheck, Compass, Sparkles, TriangleAlert } from "lucide-react"
import { useMemo, useState } from "react"

import { useI18n } from "@/components/lab/i18n"
import {
  ADVISOR_DEFAULT,
  type AdvisorData,
  type AdvisorDevice,
  type AdvisorInput,
  type AdvisorModelKind,
  type AdvisorTask,
  recommend,
} from "@/components/lab/methodAdvisor"
import { Button, Card, Pill, SelectField, TextField, cx } from "@/components/lab/ui"

const TASKS: { value: AdvisorTask; zh: string; en: string }[] = [
  { value: "image-classification", zh: "图像分类", en: "Image classification" },
  { value: "object-detection", zh: "目标检测", en: "Object detection" },
  { value: "text-generation", zh: "文本生成（LLM）", en: "Text generation (LLM)" },
  { value: "tabular-classification", zh: "表格分类", en: "Tabular classification" },
  { value: "point-cloud", zh: "点云", en: "Point cloud" },
]

const MODEL_KINDS: { value: AdvisorModelKind; zh: string; en: string }[] = [
  { value: "cnn", zh: "CNN（图像）", en: "CNN (vision)" },
  { value: "yolo", zh: "YOLO（检测）", en: "YOLO (detection)" },
  { value: "transformer", zh: "Transformer（非 LLM）", en: "Transformer (non-LLM)" },
  { value: "llm", zh: "LLM（HF 因果语言模型）", en: "LLM (HF causal LM)" },
  { value: "tabular", zh: "表格 / MLP", en: "Tabular / MLP" },
  { value: "pointnet", zh: "PointNet", en: "PointNet" },
]

const DATA_OPTIONS: { value: AdvisorData; zh: string; en: string }[] = [
  { value: "none", zh: "没有数据", en: "No data" },
  { value: "small-images", zh: "少量图片（<1k，可校准）", en: "Few images (<1k, calibration)" },
  { value: "full-images", zh: "完整标注图片集", en: "Full labelled image set" },
  { value: "instruction", zh: "指令数据（alpaca 等 JSONL）", en: "Instruction data (alpaca JSONL)" },
  { value: "preference", zh: "偏好对数据（hh-rlhf）", en: "Preference pairs (hh-rlhf)" },
]

const DEVICES: { value: AdvisorDevice; zh: string; en: string }[] = [
  { value: "cpu", zh: "通用 CPU", en: "Generic CPU" },
  { value: "intel-cpu", zh: "Intel CPU", en: "Intel CPU" },
  { value: "gpu", zh: "GPU 服务器", en: "GPU server" },
  { value: "edge", zh: "边缘设备", en: "Edge device" },
]

export function MethodAdvisorPanel() {
  const { t, locale } = useI18n()
  const [input, setInput] = useState<AdvisorInput>(ADVISOR_DEFAULT)
  const [copied, setCopied] = useState<string>("")

  const results = useMemo(() => recommend(input), [input])
  const patch = (part: Partial<AdvisorInput>) => setInput((current) => ({ ...current, ...part }))
  const tr = (zh: string, en: string) => (locale === "en" ? en : zh)

  return (
    <Card className="space-y-4">
      <div className="flex items-center gap-2">
        <Compass size={16} className="text-[var(--lb-primary)]" />
        <div>
          <p className="text-[13.5px] font-semibold text-[var(--lb-text)]">
            {tr("方法推荐", "Method advisor")}
          </p>
          <p className="lb-muted text-[12px]">
            {tr(
              "输入任务场景 / 模型 / 数据 / 部署指标，得到推荐方法与理由（依据实测数据）",
              "Describe task / model / data / deployment targets to get a ranked recommendation with reasons",
            )}
          </p>
        </div>
      </div>

      <div className="grid gap-3 lg:grid-cols-3">
        <SelectField label={tr("任务场景", "Task")} value={input.task}
          onChange={(e) => patch({ task: e.target.value as AdvisorTask })}>
          {TASKS.map((item) => (
            <option key={item.value} value={item.value}>{tr(item.zh, item.en)}</option>
          ))}
        </SelectField>
        <SelectField label={tr("模型类型", "Model kind")} value={input.modelKind}
          onChange={(e) => patch({ modelKind: e.target.value as AdvisorModelKind })}>
          {MODEL_KINDS.map((item) => (
            <option key={item.value} value={item.value}>{tr(item.zh, item.en)}</option>
          ))}
        </SelectField>
        <TextField label={tr("模型体积（MB）", "Model size (MB)")} value={String(input.modelMb)}
          onChange={(e) => patch({ modelMb: Number(e.target.value) || 0 })} />
        <SelectField label={tr("数据条件", "Data available")} value={input.data}
          onChange={(e) => patch({ data: e.target.value as AdvisorData })}>
          {DATA_OPTIONS.map((item) => (
            <option key={item.value} value={item.value}>{tr(item.zh, item.en)}</option>
          ))}
        </SelectField>
        <SelectField label={tr("部署目标", "Deployment target")} value={input.device}
          onChange={(e) => patch({ device: e.target.value as AdvisorDevice })}>
          {DEVICES.map((item) => (
            <option key={item.value} value={item.value}>{tr(item.zh, item.en)}</option>
          ))}
        </SelectField>
        <TextField label={tr("精度预算（可掉点 pp）", "Accuracy budget (pp)")}
          value={String(input.accuracyBudgetPp)}
          onChange={(e) => patch({ accuracyBudgetPp: Number(e.target.value) || 0 })} />
        <TextField label={tr("目标压缩率（%）", "Target size reduction (%)")}
          value={String(input.targetCompressionPct)}
          onChange={(e) => patch({ targetCompressionPct: Number(e.target.value) || 0 })} />
        <div className="flex flex-wrap items-end gap-3 lg:col-span-2">
          {(
            [
              ["needSpeed", tr("要求推理提速", "Need lower latency")],
              ["needLlamaCpp", tr("要 llama.cpp/ollama 直接跑", "Needs llama.cpp/ollama")],
              ["needTaskAdaptation", tr("要补领域能力（微调）", "Needs task adaptation")],
            ] as const
          ).map(([key, label]) => (
            <label key={key} className="flex items-center gap-1.5 text-[12px] text-[var(--lb-text-2)]">
              <input type="checkbox" checked={Boolean(input[key])}
                onChange={(e) => patch({ [key]: e.target.checked } as Partial<AdvisorInput>)} />
              {label}
            </label>
          ))}
        </div>
      </div>

      <div className="space-y-3">
        {results.map((item, index) => (
          <div
            key={item.card.id}
            className={cx(
              "rounded-lg border p-3.5",
              index === 0
                ? "border-[var(--lb-primary)] bg-[var(--lb-primary-soft)]"
                : "border-[var(--lb-border)] bg-[var(--lb-surface)]",
            )}
          >
            <div className="flex flex-wrap items-center gap-2">
              <Pill tone={index === 0 ? "ready" : "info"}>
                {index === 0 ? tr("首选", "Primary") : tr("备选", "Alternative")}
              </Pill>
              <span className="text-[13.5px] font-semibold text-[var(--lb-text)]">
                {tr(item.card.label.zh, item.card.label.en)}
              </span>
              <span className="lb-muted text-[11px]">{tr("匹配度", "score")} {item.score}</span>
              <span className="lb-muted text-[11px]">
                {tr("压缩率", "size")} {tr(item.card.effect.compression.zh, item.card.effect.compression.en)}
                {" · "}
                {tr("精度", "accuracy")} {tr(item.card.effect.accuracy.zh, item.card.effect.accuracy.en)}
              </span>
            </div>

            <ul className="mt-2 space-y-1 text-[12px] leading-snug text-[var(--lb-text-2)]">
              {item.reasons.map((reason, i) => (
                <li key={i} className="flex gap-1.5">
                  <Sparkles size={12} className="mt-0.5 shrink-0 text-[var(--lb-primary)]" />
                  <span>{reason}</span>
                </li>
              ))}
            </ul>
            {item.caveats.length > 0 && (
              <ul className="mt-1.5 space-y-1 text-[12px] leading-snug text-[var(--lb-text-2)]">
                {item.caveats.map((caveat, i) => (
                  <li key={i} className="flex gap-1.5">
                    <TriangleAlert size={12} className="mt-0.5 shrink-0 text-[var(--lb-danger)]" />
                    <span>{caveat}</span>
                  </li>
                ))}
              </ul>
            )}

            <div className="mt-2.5 flex items-start gap-2">
              <pre className="lb-muted flex-1 overflow-x-auto rounded-md border border-[var(--lb-border)] bg-[var(--lb-surface-2)] p-2.5 text-[11px] leading-relaxed">
                {item.command}
              </pre>
              <Button
                size="sm"
                variant="outline"
                onClick={() => {
                  void navigator.clipboard?.writeText(item.command)
                  setCopied(item.card.id)
                }}
              >
                <ClipboardCheck size={13} />
                {copied === item.card.id ? t("已复制", "Copied") : t("复制", "Copy")}
              </Button>
            </div>
          </div>
        ))}
      </div>

      <p className="lb-muted text-[11px]">
        {tr(
          "规则与实测依据见 docs/compression-advisor.md；所有数字来自 docs/compression-eval-report.md 的真实运行。",
          "Rules and measurements: docs/compression-advisor.md, backed by real runs in docs/compression-eval-report.md.",
        )}
      </p>
    </Card>
  )
}
