/**
 * routes/lab/evaluate.tsx · Insights 评估对比（双语）
 * 仅展示真实 metrics.eval / metrics.benchmark 数据。
 */
import { createFileRoute, Link } from "@tanstack/react-router"
import { RefreshCw, Zap } from "lucide-react"
import { useEffect, useMemo, useState } from "react"

import { BenchmarkRadar, type RadarSeriesInput } from "@/components/lab/BenchmarkRadar"
import { methodLabel } from "@/components/lab/constants"
import { useI18n } from "@/components/lab/i18n"
import { useLabTheme } from "@/components/lab/theme"
import { Card, EmptyState, Pill, SelectField } from "@/components/lab/ui"
import { apiListTasks } from "@/services/api"
import type { BenchmarkSide, EvalMetrics, TaskPublic } from "@/services/types"

export const Route = createFileRoute("/lab/evaluate")({
  component: EvaluatePage,
  validateSearch: (search: Record<string, unknown>): { task?: string } => ({
    task: typeof search.task === "string" ? search.task : undefined,
  }),
  head: () => ({ meta: [{ title: "Evaluate - AI Model Lightweight Platform" }] }),
})

interface EvaluableTask {
  task: TaskPublic
  evalLabel: string
  evalOriginal: number | null
  evalCompressed: number | null
  evalDelta: number | null
  benchmark?: { original?: BenchmarkSide; compressed?: BenchmarkSide }
  batch: number | null
}

function asSide(value: unknown): BenchmarkSide | null {
  if (!value || typeof value !== "object") return null
  return value as BenchmarkSide
}

function evalView(metrics: EvalMetrics | undefined): {
  label: string
  original: number | null
  compressed: number | null
} {
  if (!metrics) return { label: "", original: null, compressed: null }
  if (typeof metrics.error === "string") {
    return { label: "Eval failed", original: null, compressed: null }
  }
  if (typeof metrics.map50 === "number") {
    return { label: "YOLO mAP50", original: null, compressed: metrics.map50 }
  }
  if (typeof metrics.perplexity === "number") {
    return { label: "LLM Perplexity", original: null, compressed: metrics.perplexity }
  }
  if (typeof metrics.accuracy === "number") {
    return {
      label: "Accuracy (Top-1)",
      original:
        typeof metrics.baseline_accuracy === "number" ? metrics.baseline_accuracy : null,
      compressed: metrics.accuracy,
    }
  }
  return { label: "", original: null, compressed: null }
}

function batchOf(task: TaskPublic): number | null {
  const shape = (task.params?.benchmark as { input_shape?: unknown } | undefined)?.input_shape
  if (typeof shape !== "string") return null
  const first = shape.split(",")[0]?.trim()
  const n = Number.parseInt(first ?? "", 10)
  return Number.isFinite(n) ? n : null
}

function buildCandidates(tasks: TaskPublic[]): EvaluableTask[] {
  const out: EvaluableTask[] = []
  for (const task of tasks) {
    if (task.status !== "succeeded") continue
    const bench = task.metrics?.benchmark
    const original = bench ? asSide(bench.original) : null
    const compressed = bench ? asSide(bench.compressed) : null
    const benchmark =
      original || compressed
        ? { original: original ?? undefined, compressed: compressed ?? undefined }
        : undefined
    const ev = evalView(task.metrics?.eval)
    if (!benchmark && !ev.label) continue
    out.push({
      task,
      evalLabel: ev.label,
      evalOriginal: ev.original,
      evalCompressed: ev.compressed,
      evalDelta:
        typeof ev.original === "number" && typeof ev.compressed === "number"
          ? Number((ev.compressed - ev.original).toFixed(3))
          : null,
      benchmark,
      batch: batchOf(task),
    })
  }
  return out
}

const fmtNum = (value: number | null | undefined, digits = 1) =>
  value === null || value === undefined ? "—" : value.toFixed(digits)

function radarIndicators(
  candidate: EvaluableTask,
): { labels: string[]; original: number[]; compressed: number[] } {
  if (!candidate.benchmark) return { labels: [], original: [], compressed: [] }
  const keys: { key: keyof BenchmarkSide; label: string }[] = [
    { key: "latency_ms_p50", label: "P50 latency (ms)" },
    { key: "latency_ms_p99", label: "P99 latency (ms)" },
    { key: "latency_ms_mean", label: "Mean latency (ms)" },
    { key: "peak_memory_mb", label: "Peak memory (MB)" },
    { key: "throughput_ips", label: "Throughput (ips)" },
  ]
  const labels: string[] = []
  const original: number[] = []
  const compressed: number[] = []
  for (const item of keys) {
    const sideOriginal = candidate.benchmark.original ?? {}
    const sideCompressed = candidate.benchmark.compressed ?? {}
    const a = sideOriginal[item.key]
    const b = sideCompressed[item.key]
    if (typeof a === "number" && typeof b === "number") {
      labels.push(item.label)
      original.push(a)
      compressed.push(b)
    }
  }
  return { labels, original, compressed }
}

function MetricCompare({
  label,
  original,
  compressed,
  format,
  lowerBetter = false,
}: {
  label: string
  original: number | null
  compressed: number | null
  format?: (value: number) => string
  lowerBetter?: boolean
}) {
  const { t } = useI18n()
  const origText =
    original === null ? "—" : format ? format(original) : String(original)
  const compText =
    compressed === null ? "—" : format ? format(compressed) : String(compressed)
  const ratio =
    typeof original === "number" &&
    typeof compressed === "number" &&
    original !== 0 &&
    compressed !== 0
      ? compressed / original
      : null
  const improved = ratio !== null && (lowerBetter ? ratio < 1 : ratio > 1)
  return (
    <div className="rounded-lg border border-[var(--lb-border)] bg-[var(--lb-surface)] p-4">
      <p className="lb-muted text-xs">{label}</p>
      <div className="mt-2 grid grid-cols-2 gap-2">
        <div>
          <p className="text-[11px] text-[var(--lb-muted)]">{t("原始", "Original")}</p>
          <p className="lb-mono mt-0.5 text-base font-semibold text-[var(--lb-text-2)]">{origText}</p>
        </div>
        <div>
          <p className="text-[11px] text-[var(--lb-muted)]">{t("压缩后", "Compressed")}</p>
          <p className="lb-mono mt-0.5 text-base font-semibold text-[var(--lb-primary)]">{compText}</p>
        </div>
      </div>
      {ratio !== null && (
        <p
          className={`mt-2 text-[11px] font-semibold ${improved ? "text-[var(--lb-success)]" : "text-[var(--lb-warn)]"}`}
        >
          {lowerBetter ? t("延迟/内存 越低越好 · ", "Lower is better · ") : ""}
          {((ratio - 1) * 100).toFixed(1)}% vs {t("原始", "original")}
        </p>
      )}
    </div>
  )
}

function EvaluatePage() {
  const { task: taskParam } = Route.useSearch()
  const { t } = useI18n()
  const { theme } = useLabTheme()
  const [tasks, setTasks] = useState<TaskPublic[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState("")
  const [batch, setBatch] = useState("")
  const [selectedId, setSelectedId] = useState<string>("")

  const load = async () => {
    setLoading(true)
    setError("")
    try {
      const page = await apiListTasks({ limit: 300 })
      setTasks(page.data)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    void load()
  }, [])

  const candidates = useMemo(() => {
    const all = buildCandidates(tasks)
    if (!batch) return all
    const target = Number.parseInt(batch, 10)
    return all.filter((c) => c.batch === target || c.batch === null)
  }, [tasks, batch])

  useEffect(() => {
    if (candidates.length === 0) {
      setSelectedId("")
      return
    }
    if (taskParam && candidates.some((c) => c.task.id === taskParam)) {
      setSelectedId(taskParam)
    } else if (!selectedId || !candidates.some((c) => c.task.id === selectedId)) {
      setSelectedId(candidates[0].task.id)
    }
  }, [candidates, taskParam, selectedId])

  const selected = candidates.find((c) => c.task.id === selectedId) ?? null
  const batchOptions = useMemo(() => {
    const set = new Set<string>()
    for (const c of buildCandidates(tasks)) {
      if (c.batch !== null) set.add(String(c.batch))
    }
    return Array.from(set).sort((a, b) => Number(a) - Number(b))
  }, [tasks])

  const radar = selected ? radarIndicators(selected) : { labels: [], original: [], compressed: [] }
  const radarSeries: RadarSeriesInput[] =
    radar.labels.length >= 3
      ? [
          { name: t("原始模型", "Original"), values: radar.original, color: theme === "dark" ? "#8f8fa3" : "#94a3b8" },
          { name: t("压缩后", "Compressed"), values: radar.compressed, color: theme === "dark" ? "#a855f7" : "#7c3aed" },
        ]
      : []

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="lb-section">{t("评估对比", "Insights")}</p>
          <h1 className="mt-1 text-2xl font-bold tracking-tight text-[var(--lb-text)]">
            {t("评估对比", "Evaluation Comparison")}
          </h1>
          <p className="lb-muted mt-1 text-sm">
            {t(
              "数据来自真实任务的 metrics.eval / metrics.benchmark · 无 mock",
              "Real data from task metrics.eval / metrics.benchmark · no mock",
            )}
          </p>
        </div>
        <button type="button" onClick={() => void load()} className="lb-btn lb-btn--outline" disabled={loading}>
          <RefreshCw size={14} className={loading ? "animate-spin" : ""} /> {t("刷新", "Refresh")}
        </button>
      </div>

      {error && (
        <div className="rounded-md border border-[var(--lb-danger)]/30 bg-[var(--lb-danger-soft)] px-4 py-2.5 text-sm text-[var(--lb-danger)]">
          {error}
        </div>
      )}
      {!loading && candidates.length === 0 && (
        <Card className="p-6">
          <EmptyState
            title={t("还没有可对比的评估结果", "No comparable evaluation results yet")}
            description={t(
              "生成方法：在「压缩任务 → 新建压缩」勾选 ④ 精度评估 或 ⑤ 压缩前后性能基准，任务完成后回到本页即可看到左侧任务卡片与右侧紫色雷达。",
              "Create a compression task with ④ accuracy eval or ⑤ CPU benchmark enabled; once it completes you will see the task list and radar here.",
            )}
            action={
              <Link to="/lab/compressions" search={{}} className="lb-btn lb-btn--primary">
                {t("去创建压缩任务", "Create a compression task")}
              </Link>
            }
          />
        </Card>
      )}

      {candidates.length > 0 && (
        <div className="grid gap-6 lg:grid-cols-[320px_1fr]">
          <Card className="flex max-h-[720px] flex-col p-0">
            <div className="border-b border-[var(--lb-border)] p-4">
              <SelectField label={t("Batch Size 筛选（真实任务）", "Filter by batch size (real tasks)")} value={batch} onChange={(e) => setBatch(e.target.value)}>
                <option value="">{t("全部", "All")}</option>
                {batchOptions.map((b) => (
                  <option key={b} value={b}>batch = {b}</option>
                ))}
              </SelectField>
            </div>
            <ul className="flex-1 overflow-y-auto">
              {candidates.map((c) => {
                const active = selected?.task.id === c.task.id
                const speed = c.task.metrics?.benchmark?.speedup_x
                return (
                  <li key={c.task.id}>
                    <button
                      type="button"
                      onClick={() => setSelectedId(c.task.id)}
                      className={
                        active
                          ? "w-full border-l-4 border-[var(--lb-primary)] bg-[var(--lb-primary-soft)] px-4 py-3 text-left"
                          : "w-full border-l-4 border-transparent px-4 py-3 text-left transition hover:bg-[var(--lb-surface-2)]"
                      }
                    >
                      <p className="truncate text-sm font-semibold text-[var(--lb-text)]">
                        {c.task.model_name ?? c.task.id.slice(0, 8)}
                      </p>
                      <p className="mt-0.5 text-xs text-[var(--lb-muted)]">{methodLabel(c.task.method)}</p>
                      <div className="mt-1.5 flex items-center gap-2">
                        {c.evalLabel && <Pill tone="info">{c.evalLabel}</Pill>}
                        {typeof speed === "number" && (
                          <span className="lb-mono text-xs font-semibold text-[var(--lb-purple)]">
                            {speed.toFixed(2)}x
                          </span>
                        )}
                      </div>
                    </button>
                  </li>
                )
              })}
            </ul>
          </Card>

          {selected && (
            <div className="space-y-6">
              <Card className="p-6">
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div>
                    <h2 className="text-[15px] font-semibold text-[var(--lb-text)]">
                      {selected.task.model_name ?? selected.task.id.slice(0, 8)}
                      <span className="lb-mono ml-2 text-xs font-normal text-[var(--lb-muted)]">
                        {methodLabel(selected.task.method)}
                      </span>
                    </h2>
                    <p className="lb-muted mt-0.5 text-xs">{t("真实指标面板", "Real metric panel")}</p>
                  </div>
                  {typeof selected.task.metrics?.benchmark?.speedup_x === "number" && (
                    <div className="flex items-center gap-1.5 rounded-full bg-[var(--lb-purple-soft)] px-3 py-1.5">
                      <Zap size={13} className="text-[var(--lb-purple)]" />
                      <span className="lb-mono text-sm font-bold text-[var(--lb-purple)]">
                        {selected.task.metrics.benchmark.speedup_x?.toFixed(2)}x {t("推理加速", "speedup")}
                      </span>
                    </div>
                  )}
                </div>

                <div className="mt-5 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
                  {selected.evalLabel && (
                    <MetricCompare
                      label={selected.evalLabel}
                      original={selected.evalOriginal}
                      compressed={selected.evalCompressed}
                      format={(value) => value.toFixed(2)}
                    />
                  )}
                  {selected.benchmark && (
                    <>
                      <MetricCompare label={t("CPU P50 延迟", "CPU P50 latency")} original={selected.benchmark.original?.latency_ms_p50 ?? null} compressed={selected.benchmark.compressed?.latency_ms_p50 ?? null} format={(value) => `${value.toFixed(2)} ms`} lowerBetter />
                      <MetricCompare label={t("CPU P99 延迟", "CPU P99 latency")} original={selected.benchmark.original?.latency_ms_p99 ?? null} compressed={selected.benchmark.compressed?.latency_ms_p99 ?? null} format={(value) => `${value.toFixed(2)} ms`} lowerBetter />
                      <MetricCompare label={t("峰值内存", "Peak memory")} original={selected.benchmark.original?.peak_memory_mb ?? null} compressed={selected.benchmark.compressed?.peak_memory_mb ?? null} format={(value) => `${value.toFixed(1)} MB`} lowerBetter />
                    </>
                  )}
                </div>

                {selected.evalDelta !== null && (
                  <p className="lb-muted mt-3 text-xs">
                    {t("精度变化：", "Accuracy change: ")}
                    {(selected.evalDelta >= 0 ? "+" : "") + selected.evalDelta.toFixed(2)}（{t("原始", "original")} {fmtNum(selected.evalOriginal)} → {t("压缩后", "compressed")} {fmtNum(selected.evalCompressed)}）
                  </p>
                )}
              </Card>

              <Card className="p-6">
                <h2 className="text-[15px] font-semibold text-[var(--lb-text)]">
                  {t("性能对比雷达（真实数据）", "Performance radar (real data)")}
                </h2>
                <p className="lb-muted mt-0.5 text-xs">
                  {t(
                    "延迟 / 内存为原始数值（越低越好），吞吐为数值（越高越好）；同一维度两侧均已由真实基准测得。",
                    "Latency / memory are raw values (lower better), throughput higher better; both sides measured by the real benchmark.",
                  )}
                </p>
                {radarSeries.length >= 2 ? (
                  <div className="mt-4">
                    <BenchmarkRadar indicators={radar.labels} series={radarSeries} height={330} />
                  </div>
                ) : (
                  <p className="lb-muted mt-4 text-sm">
                    {t(
                      "该任务缺少足够的原始/压缩双端基准数值（至少需要 3 个维度），暂无法绘制雷达。",
                      "This task lacks enough original/compressed values (at least 3 axes) to draw a radar.",
                    )}
                  </p>
                )}
              </Card>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
