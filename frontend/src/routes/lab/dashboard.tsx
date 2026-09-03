/**
 * routes/lab/dashboard.tsx · 新版工作台（双语）
 * 真实指标来自 platform-status / workers / compressions。
 */
import { createFileRoute, Link } from "@tanstack/react-router"
import {
  Activity,
  Boxes,
  Database,
  FlaskConical,
  HardDrive,
  PlusCircle,
  Server,
  Timer,
} from "lucide-react"

import {
  TASK_STATUS_META,
  methodLabel,
  taskStatusLabel,
} from "@/components/lab/constants"
import { useI18n } from "@/components/lab/i18n"
import { Card, Pill, ProgressBar, StatCard } from "@/components/lab/ui"
import { usePolling } from "@/hooks/usePolling"
import { apiListTasks, apiPlatformStatus, apiWorkers } from "@/services/api"
import { formatBytes, formatUptime } from "@/services/format"

export const Route = createFileRoute("/lab/dashboard")({
  component: DashboardPage,
  head: () => ({
    meta: [{ title: "Dashboard - AI Model Lightweight Platform" }],
  }),
})

interface Snapshot {
  status: Awaited<ReturnType<typeof apiPlatformStatus>>
  workers: Awaited<ReturnType<typeof apiWorkers>>
  tasks: Awaited<ReturnType<typeof apiListTasks>>
}

function DashboardPage() {
  const { t } = useI18n()
  const { data, error } = usePolling<Snapshot>(
    async () => {
      const [status, workers, tasks] = await Promise.all([
        apiPlatformStatus(),
        apiWorkers(),
        apiListTasks({ limit: 200 }),
      ])
      return { status, workers, tasks }
    },
    [],
    { intervalMs: 5000 },
  )

  const status = data?.status
  const workers = data?.workers
  const tasks = data?.tasks

  const activeTasks = (tasks?.data ?? []).filter((task) =>
    ["pending", "running"].includes(task.status),
  )
  const succeededCount = status?.tasks.succeeded ?? 0
  const failedCount = status?.tasks.failed ?? 0

  const avgProgress =
    activeTasks.length > 0
      ? activeTasks.reduce((sum, task) => sum + (task.progress ?? 0), 0) /
        activeTasks.length
      : 0

  return (
    <div className="space-y-7">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="lb-section">{t("模型轻量化平台", "Model Lightweight Platform")}</p>
          <h1 className="mt-1 text-2xl font-bold tracking-tight text-[var(--lb-text)]">
            {t("工作台", "Dashboard")}
          </h1>
          <p className="lb-muted mt-1 text-sm">
            {t(
              "全部指标来自真实后端（platform-status / workers / compressions）。",
              "All metrics come from the real backend (platform-status / workers / compressions).",
            )}
          </p>
        </div>
        <Link to="/lab/compressions" search={{}} className="lb-btn lb-btn--primary">
          <PlusCircle size={15} /> {t("新建压缩任务", "New Compression")}
        </Link>
      </div>

      {error && (
        <div className="rounded-md border border-[var(--lb-danger)]/30 bg-[var(--lb-danger-soft)] px-4 py-3 text-sm text-[var(--lb-danger)]">
          {t("数据加载失败：", "Failed to load: ")}
          {error.message}
        </div>
      )}

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5">
        <StatCard label={t("模型总量", "Models")} value={String(status?.models ?? 0)} icon={<Boxes size={17} />} hint={t("已注册模型", "Registered models")} />
        <StatCard label={t("数据集总量", "Datasets")} value={String(status?.datasets ?? 0)} icon={<Database size={17} />} hint={t("校准 / 评估数据集", "Calibration / eval data")} />
        <StatCard label={t("存储占用", "Storage")} value={formatBytes(status?.storage_bytes ?? 0)} icon={<HardDrive size={17} />} hint={t("模型 + 数据集", "Models + datasets")} />
        <StatCard label={t("Worker 数", "Workers")} value={String(workers?.worker_count ?? 0)} icon={<Server size={17} />} hint={t("在线任务执行器", "Online workers")} />
        <StatCard label={t("队列深度", "Queue depth")} value={String(workers?.queue_depth ?? 0)} icon={<Timer size={17} />} hint={t("排队等待的任务", "Queued tasks")} />
      </div>

      <Card className="p-6">
        <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="text-[15px] font-semibold text-[var(--lb-text)]">
              {t("任务运行进度", "Task Progress")}
            </h2>
            <p className="lb-muted mt-0.5 text-xs">
              {t("每 5s 从真实任务接口刷新 · ", "Auto refresh every 5s · ")}
              {t("成功", "done")} {succeededCount} · {t("失败", "failed")} {failedCount} ·{" "}
              {t("活跃", "active")} {activeTasks.length}
            </p>
          </div>
          {activeTasks.length > 0 && (
            <div className="flex items-center gap-3">
              <div className="w-44">
                <ProgressBar value={avgProgress} />
              </div>
              <span className="lb-accent-text text-sm font-semibold">
                {avgProgress.toFixed(0)}%
              </span>
            </div>
          )}
        </div>

        {activeTasks.length === 0 ? (
          <div className="flex flex-col items-center justify-center gap-2 rounded-lg border border-dashed border-[var(--lb-border-strong)] py-12 text-center">
            <Activity size={20} className="text-[var(--lb-muted)]" />
            <p className="lb-muted text-sm">{t("当前没有运行中的任务", "No running tasks")}</p>
            <Link to="/lab/compressions" search={{}} className="mt-1 text-sm font-medium text-[var(--lb-primary)]">
              {t("去创建第一个压缩任务 →", "Create your first compression task →")}
            </Link>
          </div>
        ) : (
          <ul className="space-y-3">
            {activeTasks.map((task) => {
              const tone = TASK_STATUS_META[task.status].tone
              return (
                <li
                  key={task.id}
                  className="flex flex-wrap items-center gap-4 rounded-lg border border-[var(--lb-border)] bg-[var(--lb-surface-2)] px-4 py-3"
                >
                  <div className="min-w-0 flex-1">
                    <p className="truncate text-sm font-semibold text-[var(--lb-text)]">
                      {task.model_name ?? task.model_id.slice(0, 8)}
                    </p>
                    <p className="lb-muted mt-0.5 flex items-center gap-2 text-xs">
                      <FlaskConical size={12} /> {methodLabel(task.method)}
                    </p>
                  </div>
                  <Pill tone={tone}>{taskStatusLabel(task.status)}</Pill>
                  <div className="w-40">
                    <ProgressBar value={task.progress} />
                  </div>
                  <span className="lb-mono w-11 text-right text-xs font-medium text-[var(--lb-text-2)]">
                    {task.progress.toFixed(0)}%
                  </span>
                </li>
              )
            })}
          </ul>
        )}
      </Card>

      <Card className="p-6">
        <h2 className="mb-3 text-[15px] font-semibold text-[var(--lb-text)]">
          {t("系统运行状态", "System Status")}
        </h2>
        <div className="grid gap-3 sm:grid-cols-3">
          <div className="rounded-lg border border-[var(--lb-border)] p-4">
            <p className="lb-muted text-xs">Redis</p>
            <p
              className={`mt-1 text-sm font-semibold ${
                status?.redis === "up" ? "text-[var(--lb-success)]" : "text-[var(--lb-danger)]"
              }`}
            >
              {status?.redis === "up" ? t("运行中", "Running") : t("异常", "Down")}
            </p>
          </div>
          <div className="rounded-lg border border-[var(--lb-border)] p-4">
            <p className="lb-muted text-xs">{t("运行中任务（Worker）", "Running jobs (worker)")}</p>
            <p className="mt-1 text-sm font-semibold text-[var(--lb-text-2)]">
              {workers?.running_jobs ?? 0}
            </p>
          </div>
          <div className="rounded-lg border border-[var(--lb-border)] p-4">
            <p className="lb-muted text-xs">{t("Worker 详情", "Worker details")}</p>
            {(workers?.workers ?? []).length > 0 ? (
              <ul className="mt-1 space-y-1">
                {workers?.workers.map((worker) => (
                  <li key={String(worker.id)} className="lb-mono text-xs text-[var(--lb-text-2)]">
                    pid {worker.pid} · {t("已运行", "up")} {formatUptime(Number(worker.uptime_s) || 0)}
                  </li>
                ))}
              </ul>
            ) : (
              <p className="mt-1 text-sm text-[var(--lb-muted)]">
                {t("无活跃 worker", "No active workers")}
              </p>
            )}
          </div>
        </div>
      </Card>
    </div>
  )
}
