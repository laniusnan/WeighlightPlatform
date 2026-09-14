/**
 * routes/lab/task.$taskId.tsx · 任务详情页（进度历史 / 指标 / 产物 / 操作）
 */
import { createFileRoute, Link, useNavigate } from "@tanstack/react-router"
import { useCallback, useEffect, useState } from "react"
import { ArrowLeft, Download, MessageSquare, RefreshCw, RotateCcw, Trash2 } from "lucide-react"
import {
  apiDeleteTask,
  apiGetTask,
  apiGetTaskProgress,
  apiReevaluateTask,
  apiRetryTask,
  taskDownloadPath,
} from "@/services/api"
import { apiErrorMessage, downloadBlob } from "@/services/http"
import { formatBytes, formatDateTime } from "@/services/format"
import { ARTIFACT_INFO } from "@/routes/lab/compressions"
import { methodLabel, TASK_STATUS_META, taskStatusLabel } from "@/components/lab/constants"
import { Button, Card, EmptyState, Pill, ProgressBar } from "@/components/lab/ui"
import { useI18n } from "@/components/lab/i18n"
import type { ProgressHistoryResponse, TaskPublic } from "@/services/types"

export const Route = createFileRoute("/lab/task/$taskId")({
  component: TaskDetailPage,
})

function TaskDetailPage() {
  const { t } = useI18n()
  const navigate = useNavigate()
  const { taskId } = Route.useParams()
  const [task, setTask] = useState<TaskPublic | null>(null)
  const [history, setHistory] = useState<ProgressHistoryResponse | null>(null)
  const [error, setError] = useState("")

  const load = useCallback(async () => {
    try {
      const [detail, prog] = await Promise.all([apiGetTask(taskId), apiGetTaskProgress(taskId)])
      setTask(detail)
      setHistory(prog)
      setError("")
    } catch (e) {
      setError(apiErrorMessage(e))
    }
  }, [taskId])

  useEffect(() => {
    void load()
    const timer = window.setInterval(() => void load(), 4000)
    return () => window.clearInterval(timer)
  }, [load])

  if (error) {
    return (
      <Card>
        <EmptyState title={t("无法加载任务", "Cannot load task")} description={error} />
      </Card>
    )
  }
  if (!task) {
    return (
      <Card>
        <p className="lb-muted text-sm">{t("加载中…", "Loading…")}</p>
      </Card>
    )
  }

  const info = ARTIFACT_INFO[task.method]
  const metrics = (task.metrics ?? {}) as Record<string, unknown>
  const samples = history?.history ?? []

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <Link to="/lab/compressions" className="lb-muted inline-flex items-center gap-1 text-xs hover:underline">
            <ArrowLeft size={12} /> {t("返回任务中心", "Back to task center")}
          </Link>
          <h1 className="mt-1 text-2xl font-bold tracking-tight text-[var(--lb-text)]">
            {t("任务详情", "Task detail")}
          </h1>
          <p className="lb-muted mt-1 text-sm">
            {task.model_name ?? task.model_id.slice(0, 8)} · {methodLabel(task.method)} ·{" "}
            <span className="lb-mono">{task.id.slice(0, 8)}</span>
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <Pill tone={TASK_STATUS_META[task.status].tone}>{taskStatusLabel(task.status)}</Pill>
          <Button size="sm" variant="outline" onClick={() => void load()}>
            <RefreshCw size={14} /> {t("刷新", "Refresh")}
          </Button>
          {task.status === "failed" && (
            <Button size="sm" variant="outline" onClick={() => void apiRetryTask(task.id).then(load)}>
              <RotateCcw size={14} /> {t("重试", "Retry")}
            </Button>
          )}
          {task.status === "succeeded" && Boolean((task.params as Record<string, unknown> | undefined)?.eval) && (
            <Button
              size="sm"
              variant="outline"
              onClick={() => void apiReevaluateTask(task.id, "eval").then(() => load())}
            >
              <RefreshCw size={14} /> {t("重新评估", "Re-run eval")}
            </Button>
          )}
          {task.status === "succeeded" && Boolean((task.params as Record<string, unknown> | undefined)?.benchmark) && (
            <Button
              size="sm"
              variant="outline"
              onClick={() => void apiReevaluateTask(task.id, "benchmark").then(() => load())}
            >
              <RefreshCw size={14} /> {t("重新基准", "Re-run benchmark")}
            </Button>
          )}
          {task.status === "succeeded" && (
            <Button
              size="sm"
              variant="outline"
              onClick={() => void downloadBlob(taskDownloadPath(task.id), `${task.method}_result`)}
            >
              <Download size={14} /> {t("下载产物", "Download artifacts")}
            </Button>
          )}
          {task.status === "succeeded" && task.method === "llm_quantize" && (
            <Button size="sm" variant="outline" onClick={() => navigate({ to: "/lab/evaluate", search: { task: task.id } })}>
              <MessageSquare size={14} /> {t("对话测试", "Chat test")}
            </Button>
          )}
          <Button
            size="sm"
            variant="ghost"
            onClick={async () => {
              if (!window.confirm(`${t("删除任务", "Delete task")}？`)) return
              await apiDeleteTask(task.id)
              navigate({ to: "/lab/compressions" })
            }}
          >
            <Trash2 size={14} />
          </Button>
        </div>
      </div>
      <div className="grid gap-4 lg:grid-cols-3">
        <Card className="space-y-3 lg:col-span-2">
          <p className="text-[13px] font-semibold text-[var(--lb-text-2)]">
            {t("执行状态", "Execution")}
          </p>
          <div className="flex items-center gap-3">
            <div className="w-64">
              <ProgressBar value={task.progress} />
            </div>
            <span className="lb-mono text-xs">{task.progress.toFixed(1)}%</span>
          </div>
          <div className="grid gap-2 text-xs sm:grid-cols-2">
            <p className="lb-muted">{t("创建", "Created")}：{formatDateTime(task.created_at)}</p>
            <p className="lb-muted">
              {t("开始", "Started")}：{task.started_at ? formatDateTime(task.started_at) : "—"}
            </p>
            <p className="lb-muted">
              {t("完成", "Finished")}：{task.finished_at ? formatDateTime(task.finished_at) : "—"}
            </p>
            <p className="lb-muted">
              {t("输出大小", "Output size")}：{formatBytes(task.output_size_bytes ?? 0)}
            </p>
          </div>
          {task.error_message && (
            <p className="lb-mono rounded-md border border-[var(--lb-border-strong)] p-2 text-xs text-[var(--lb-danger,#dc2626)]">
              {task.error_message}
            </p>
          )}
          {samples.length > 0 && (
            <div>
              <p className="mb-1 text-xs font-semibold text-[var(--lb-text-2)]">
                {t("进度历史", "Progress history")}（{samples.length}）
              </p>
              <div className="flex h-16 items-end gap-0.5 overflow-hidden rounded border border-[var(--lb-border)] p-1">
                {samples.slice(-120).map((s, i) => (
                  <div
                    key={`${s[0]}-${i}`}
                    className="w-1 rounded-sm bg-[var(--lb-primary)]"
                    style={{ height: `${Math.max(4, s[1])}%` }}
                    title={`${s[0]}s → ${s[1]}%`}
                  />
                ))}
              </div>
            </div>
          )}
        </Card>

        <Card className="space-y-2">
          <p className="text-[13px] font-semibold text-[var(--lb-text-2)]">
            {t("产物说明", "Artifacts")}
          </p>
          {info ? (
            <>
              <p className="lb-muted text-xs">{t(info.zh, info.en)}</p>
              {info.items.map(([name, zh, en]) => (
                <div key={name} className="rounded border border-[var(--lb-border)] p-2">
                  <p className="lb-mono text-xs font-semibold">{name}</p>
                  <p className="lb-muted mt-0.5 text-xs">{t(zh, en)}</p>
                </div>
              ))}
            </>
          ) : (
            <p className="lb-muted text-xs">
              {t("压缩产物打包为 zip 下载", "Artifacts are served as one zip")}
            </p>
          )}
        </Card>
      </div>

      <Card className="space-y-2">
        <p className="text-[13px] font-semibold text-[var(--lb-text-2)]">
          {t("指标（原始）", "Metrics (raw)")}
        </p>
        {Object.keys(metrics).length === 0 ? (
          <p className="lb-muted text-xs">{t("暂无指标", "No metrics yet")}</p>
        ) : (
          <div className="grid gap-1.5 sm:grid-cols-2 lg:grid-cols-3">
            {Object.entries(metrics).map(([k, v]) => (
              <div key={k} className="rounded border border-[var(--lb-border)] p-2">
                <p className="lb-mono text-[11px] text-[var(--lb-muted)]">{k}</p>
                <p className="lb-mono text-xs break-all text-[var(--lb-text)]">
                  {typeof v === "object" ? JSON.stringify(v) : String(v)}
                </p>
              </div>
            ))}
          </div>
        )}
      </Card>
    </div>
  )
}
