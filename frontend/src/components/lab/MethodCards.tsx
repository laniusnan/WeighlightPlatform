/**
 * components/lab/MethodCards.tsx · 压缩方法说明卡（数据驱动）
 *
 * 卡片内容全部来自 `./method-cards.json`（schema 见 docs/method-cards.schema.json），
 * 本文件只负责类型、分组与渲染，不在 JSX 里硬编码任何方法文案。
 */
import { ChevronDown, ChevronRight, Info, Layers } from "lucide-react"
import { useMemo, useState } from "react"

import { useI18n } from "@/components/lab/i18n"
import type { PillTone } from "@/components/lab/ui"
import { Button, Card, cx, Pill } from "@/components/lab/ui"

import rawCards from "./method-cards.json"

export interface I18nText {
  zh: string
  en: string
}

export interface MethodCardInput {
  name: string
  required: boolean
  format: I18nText
  note?: I18nText
}

export interface MethodCardEvidence {
  dataset: string
  summary: I18nText
  baseline_accuracy?: number | null
  result_accuracy?: number | null
  bytes_before?: number | null
  bytes_after?: number | null
  duration_s?: number | null
  report_anchor?: string
}

export interface MethodCardData {
  id: string
  label: I18nText
  summary: I18nText
  group: string
  calibration: { needed: boolean; note: I18nText }
  inputs: MethodCardInput[]
  suits: { models: string[]; tasks: string[] }
  effect: { compression: I18nText; accuracy: I18nText; latency: I18nText }
  limits: I18nText[]
  evidence?: MethodCardEvidence
}

interface MethodCardsFile {
  version: number
  generatedAt: string
  report?: string
  cards: MethodCardData[]
}

const data = rawCards as MethodCardsFile

export const METHOD_CARDS: MethodCardData[] = data.cards

export function methodCard(id: string): MethodCardData | undefined {
  return METHOD_CARDS.find((card) => card.id === id)
}

const MODEL_TONE: Record<string, PillTone> = {
  cnn: "info",
  transformer: "info",
  llm: "pending",
  tabular: "ready",
  yolo: "info",
  pointnet: "idle",
}

export function MethodCardPanel() {
  const { t, locale } = useI18n()
  const [open, setOpen] = useState(false)
  const [group, setGroup] = useState<string>("all")

  const groups = useMemo(
    () => [
      "all",
      ...Array.from(new Set(METHOD_CARDS.map((card) => card.group))),
    ],
    [],
  )
  const visible = useMemo(
    () =>
      group === "all"
        ? METHOD_CARDS
        : METHOD_CARDS.filter((card) => card.group === group),
    [group],
  )

  return (
    <Card className="space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <Layers size={16} className="text-[var(--lb-primary)]" />
          <div>
            <p className="text-[13.5px] font-semibold text-[var(--lb-text)]">
              {t("压缩方法说明卡", "Method cards")}
            </p>
            <p className="lb-muted text-[12px]">
              {t(
                `${METHOD_CARDS.length} 个方法 · 校准数据 / 输入 / 适用模型 / 效果 / 限制`,
                `${METHOD_CARDS.length} methods · calibration / inputs / models / effect / limits`,
              )}
            </p>
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {open &&
            groups.map((item) => (
              <button
                key={item}
                type="button"
                onClick={() => setGroup(item)}
                className={cx(
                  "rounded-full px-3 py-1 text-[12px] font-medium transition",
                  item === group
                    ? "border-2 border-[var(--lb-primary)] bg-[var(--lb-primary-soft)] text-[var(--lb-primary)]"
                    : "border border-[var(--lb-border-strong)] text-[var(--lb-text-2)] hover:border-[var(--lb-primary)]",
                )}
              >
                {item === "all" ? t("全部", "All") : item}
              </button>
            ))}
          <Button
            variant="outline"
            size="sm"
            onClick={() => setOpen((value) => !value)}
          >
            {open ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
            {open ? t("收起", "Hide") : t("展开方法卡", "Show cards")}
          </Button>
        </div>
      </div>

      {open ? (
        <div className="grid grid-cols-1 gap-3 lg:grid-cols-2 xl:grid-cols-3">
          {visible.map((card) => (
            <MethodCard key={card.id} card={card} />
          ))}
        </div>
      ) : (
        <p className="lb-muted flex items-center gap-1.5 text-[12px]">
          <Info size={13} />
          {locale === "en"
            ? "Expand to see what each method needs, what it fits, and what it costs (numbers come from docs/compression-eval-report.md)."
            : "展开查看每个方法需要什么输入、适合什么模型、能达到什么效果（数值来自 docs/compression-eval-report.md）。"}
        </p>
      )}
    </Card>
  )
}

export function MethodCard({
  card,
  compact = false,
}: {
  card: MethodCardData
  compact?: boolean
}) {
  const { locale } = useI18n()
  const tr = (text: I18nText) => (locale === "en" ? text.en : text.zh)

  return (
    <div className="flex h-full flex-col gap-2.5 rounded-lg border border-[var(--lb-border)] bg-[var(--lb-surface)] p-3.5">
      <div className="flex items-start justify-between gap-2">
        <div>
          <p className="text-[13.5px] font-semibold text-[var(--lb-text)]">
            {tr(card.label)}
          </p>
          <p className="lb-muted mt-0.5 text-[12px] leading-snug">
            {tr(card.summary)}
          </p>
        </div>
        <span className="lb-muted shrink-0 text-[11px]">{card.group}</span>
      </div>

      <div className="flex flex-wrap gap-1.5">
        <Pill tone={card.calibration.needed ? "pending" : "ready"}>
          {card.calibration.needed
            ? locale === "en"
              ? "Needs calibration data"
              : "需要校准数据"
            : locale === "en"
              ? "No calibration data"
              : "免校准数据"}
        </Pill>
        {card.suits.models.map((model) => (
          <Pill key={model} tone={MODEL_TONE[model] ?? "idle"}>
            {model}
          </Pill>
        ))}
        {!compact &&
          card.suits.tasks.map((task) => (
            <Pill key={task} tone="idle">
              {task}
            </Pill>
          ))}
      </div>

      <p className="lb-muted text-[12px] leading-snug">
        <span className="font-medium text-[var(--lb-text-2)]">
          {locale === "en" ? "Inputs" : "需要输入"}：
        </span>
        {card.inputs
          .map(
            (input) =>
              `${input.name}${input.required ? "" : "?"}（${tr(input.format)}）`,
          )
          .join(" · ")}
      </p>

      <dl className="grid grid-cols-1 gap-1 text-[12px] sm:grid-cols-3">
        <div>
          <dt className="lb-muted">{locale === "en" ? "Size" : "压缩率"}</dt>
          <dd className="font-medium text-[var(--lb-text)]">
            {tr(card.effect.compression)}
          </dd>
        </div>
        <div>
          <dt className="lb-muted">{locale === "en" ? "Accuracy" : "精度"}</dt>
          <dd className="font-medium text-[var(--lb-text)]">
            {tr(card.effect.accuracy)}
          </dd>
        </div>
        <div>
          <dt className="lb-muted">{locale === "en" ? "Latency" : "延迟"}</dt>
          <dd className="font-medium text-[var(--lb-text)]">
            {tr(card.effect.latency)}
          </dd>
        </div>
      </dl>

      <ul className="lb-muted list-disc space-y-0.5 pl-4 text-[12px] leading-snug">
        {card.limits.map((limit, index) => (
          <li key={index}>{tr(limit)}</li>
        ))}
      </ul>

      {card.evidence && (
        <p className="mt-0.5 rounded-md border border-[var(--lb-border)] bg-[var(--lb-primary-soft)] px-2.5 py-1.5 text-[12px] text-[var(--lb-primary)]">
          {locale === "en" ? "Measured" : "实测"}（{card.evidence.dataset}）：
          {tr(card.evidence.summary)}
        </p>
      )}
    </div>
  )
}
