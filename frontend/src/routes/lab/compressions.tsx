/**
 * routes/lab/compressions.tsx · Lab 压缩任务中心（双语）
 * 真实任务列表 + 新建压缩向导。
 */
import { createFileRoute, useNavigate } from "@tanstack/react-router"
import {
  Download,
  FlaskConical,
  PlusCircle,
  RefreshCw,
  RotateCcw,
  Trash2,
} from "lucide-react"
import { useCallback, useEffect, useMemo, useState } from "react"

import {
  METHOD_META,
  TASK_STATUS_META,
  methodDescription,
  methodLabel,
  taskStatusLabel,
} from "@/components/lab/constants"
import { useI18n } from "@/components/lab/i18n"
import {
  Button,
  Card,
  EmptyState,
  Modal,
  Pill,
  ProgressBar,
  SelectField,
  TextField,
} from "@/components/lab/ui"
import { usePolling } from "@/hooks/usePolling"
import {
  apiCreateTask,
  apiDeleteTask,
  apiListDatasets,
  apiListModels,
  apiListTasks,
  apiRetryTask,
  taskDownloadPath,
} from "@/services/api"
import { apiErrorMessage, downloadBlob } from "@/services/http"
import { formatBytes, formatDateTime } from "@/services/format"
import type {
  CompressionMethod,
  DatasetPublic,
  ModelPublic,
  TaskPublic,
  TaskStatus,
} from "@/services/types"

export const Route = createFileRoute("/lab/compressions")({
  component: CompressionsPage,
  validateSearch: (search: Record<string, unknown>): { model?: string } => ({
    model: typeof search.model === "string" ? search.model : undefined,
  }),
  head: () => ({ meta: [{ title: "Compressions - AI Model Lightweight Platform" }] }),
})

interface ParamDef {
  key: string
  labelZh: string
  labelEn: string
  placeholder?: string
  default?: string
}

const METHOD_PARAMS: Record<CompressionMethod, ParamDef[]> = {
  export_onnx: [
    { key: "model_kind", labelZh: "模型类型", labelEn: "Model kind", placeholder: "cnn / yolo / pointnet", default: "cnn" },
    { key: "input_shape", labelZh: "输入形状", labelEn: "Input shape", placeholder: "1,3,224,224", default: "1,3,224,224" },
    { key: "opset", labelZh: "ONNX opset", labelEn: "ONNX opset", placeholder: "12", default: "12" },
  ],
  quantize_onnx: [],
  quantize_onnx_static: [
    { key: "calib_data_url", labelZh: "校准数据 URL（可选，zip/tar 图片包）", labelEn: "Calibration URL (optional, zip/tar images)", placeholder: "https://example.com/calib.zip" },
    { key: "calibration_samples", labelZh: "校准样本数", labelEn: "Calibration samples", placeholder: "16", default: "16" },
    { key: "input_shape", labelZh: "输入形状", labelEn: "Input shape", placeholder: "1,3,224,224", default: "1,3,224,224" },
  ],
  openvino_int8: [
    { key: "precision", labelZh: "精度", labelEn: "Precision", placeholder: "int8 / fp16 / fp32", default: "int8" },
    { key: "model_kind", labelZh: "模型类型", labelEn: "Model kind", placeholder: "cnn / yolo", default: "cnn" },
    { key: "input_shape", labelZh: "输入形状", labelEn: "Input shape", placeholder: "1,3,224,224", default: "1,3,224,224" },
  ],
  prune: [
    { key: "rate", labelZh: "剪枝比例 (0~1)", labelEn: "Prune ratio (0~1)", placeholder: "0.3", default: "0.3" },
    { key: "input_shape", labelZh: "输入形状", labelEn: "Input shape", placeholder: "1,3,224,224", default: "1,3,224,224" },
  ],
  sparsify: [
    { key: "rate", labelZh: "稀疏比例 (0~1)", labelEn: "Sparsity ratio (0~1)", placeholder: "0.3", default: "0.3" },
    { key: "input_shape", labelZh: "输入形状", labelEn: "Input shape", placeholder: "1,3,224,224", default: "1,3,224,224" },
  ],
  qat: [
    { key: "epochs", labelZh: "训练轮数", labelEn: "Epochs", placeholder: "3", default: "3" },
    { key: "lr", labelZh: "学习率", labelEn: "Learning rate", placeholder: "1e-4", default: "1e-4" },
    { key: "patience", labelZh: "早停耐心", labelEn: "Early stop patience", placeholder: "2", default: "2" },
    { key: "lr_schedule", labelZh: "学习率调度", labelEn: "LR schedule", placeholder: "cosine", default: "cosine" },
    { key: "input_shape", labelZh: "输入形状", labelEn: "Input shape", placeholder: "1,3,224,224", default: "1,3,224,224" },
  ],
  export_gguf: [{ key: "outtype", labelZh: "输出类型", labelEn: "Output type", placeholder: "f16 / bf16 / q8_0", default: "f16" }],
  llm_quantize: [{ key: "bits", labelZh: "量化位宽", labelEn: "Bits", placeholder: "4 / 8", default: "4" }],
}

const EVAL_TYPES = [
  { value: "", labelZh: "不评估", labelEn: "No evaluation" },
  { value: "classification", labelZh: "图像分类 Top-1", labelEn: "Image classification Top-1" },
  { value: "yolo", labelZh: "YOLO mAP", labelEn: "YOLO mAP" },
  { value: "perplexity", labelZh: "LLM Perplexity", labelEn: "LLM Perplexity" },
]

/* ---------------------------- 新建任务向导 ---------------------------- */
function TaskWizard({
  open,
  presetModelId,
  models,
  datasets,
  onClose,
  onCreated,
}: {
  open: boolean
  presetModelId?: string
  models: ModelPublic[]
  datasets: DatasetPublic[]
  onClose: () => void
  onCreated: () => void
}) {
  const { t } = useI18n()
  const readyModels = models.filter((m) => m.status === "ready")
  const [modelId, setModelId] = useState<string>(() =>
    readyModels.some((m) => m.id === presetModelId)
      ? (presetModelId as string)
      : (readyModels[0]?.id ?? ""),
  )
  const [method, setMethod] = useState<CompressionMethod>("export_onnx")
  const [paramValues, setParamValues] = useState<Record<string, string>>({})
  const [calibDatasetId, setCalibDatasetId] = useState("")
  const [qatDatasetId, setQatDatasetId] = useState("")
  const [evalType, setEvalType] = useState("")
  const [evalDatasetId, setEvalDatasetId] = useState("")
  const [evalDataUrl, setEvalDataUrl] = useState("")
  const [evalImageSize, setEvalImageSize] = useState("224")
  const [benchmarkOn, setBenchmarkOn] = useState(false)
  const [benchmarkShape, setBenchmarkShape] = useState("1,3,224,224")
  const [benchmarkIters, setBenchmarkIters] = useState("20")
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState("")

  const setParam = (key: string, value: string) =>
    setParamValues((prev) => ({ ...prev, [key]: value }))

  const submit = async () => {
    if (!modelId) {
      setError(t("请先选择一个 ready 状态的模型", "Choose a model with Ready status first"))
      return
    }
    setBusy(true)
    setError("")
    try {
      const params: Record<string, unknown> = {}
      for (const param of METHOD_PARAMS[method]) {
        const value = (paramValues[param.key] ?? param.default ?? "").trim()
        if (value) params[param.key] = value
      }
      if (method === "quantize_onnx_static" && calibDatasetId) {
        params.calib_dataset_id = calibDatasetId
      }
      if (method === "qat") {
        if (!qatDatasetId) {
          setBusy(false)
          setError(t("QAT 需要选择一个微调数据集（按类别文件夹结构）", "QAT requires a fine-tune dataset (class folders)"))
          return
        }
        params.dataset_id = qatDatasetId
      }
      if (evalType) {
        const evalCfg: Record<string, unknown> = { type: evalType }
        if (evalDatasetId) evalCfg.dataset_id = evalDatasetId
        else if (evalDataUrl.trim()) evalCfg.data_url = evalDataUrl.trim()
        if (evalType === "classification") {
          evalCfg.image_size = Number.parseInt(evalImageSize, 10) || 224
        }
        params.eval = evalCfg
      }
      if (benchmarkOn) {
        params.benchmark = {
          input_shape: benchmarkShape.trim() || "1,3,224,224",
          iterations: Number.parseInt(benchmarkIters, 10) || 20,
        }
      }
      await apiCreateTask({ model_id: modelId, method, params })
      onCreated()
      onClose()
    } catch (e) {
      setError(apiErrorMessage(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <Modal
      open={open}
      onClose={onClose}
      title={t("新建压缩任务", "New Compression Task")}
      wide
      footer={
        <>
          <Button variant="outline" onClick={onClose}>{t("取消", "Cancel")}</Button>
          <Button onClick={() => void submit()} disabled={busy || !readyModels.length}>
            {busy ? t("提交中…", "Submitting…") : t("创建任务", "Create Task")}
          </Button>
        </>
      }
    >
      <div className="grid gap-5">
        {error && (
          <div className="rounded-md border border-[var(--lb-danger)]/30 bg-[var(--lb-danger-soft)] px-4 py-2.5 text-sm text-[var(--lb-danger)]">{error}</div>
        )}
        {readyModels.length === 0 && (
          <div className="rounded-md border border-[var(--lb-warn)]/40 bg-[var(--lb-warn-soft)] px-4 py-2.5 text-sm text-[var(--lb-warn)]">
            {t("当前没有 ready 状态的模型，请先到「模型管理」上传或拉取。", "No Ready models yet. Upload or pull one in Models first.")}
          </div>
        )}

        <SelectField label={t("① 选择模型", "1. Model")} value={modelId} onChange={(e) => setModelId(e.target.value)}>
          <option value="">{t("选择要压缩的模型", "Choose a model")}</option>
          {readyModels.map((m) => (
            <option key={m.id} value={m.id}>{m.name} · {m.framework}</option>
          ))}
        </SelectField>

        <div>
          <span className="lb-label">{t("② 压缩方法", "2. Method")}</span>
          <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
            {METHOD_META.map((m) => {
              const selected = m.value === method
              return (
                <button
                  key={m.value}
                  type="button"
                  onClick={() => setMethod(m.value)}
                  className={
                    selected
                      ? "rounded-md border-2 border-[var(--lb-primary)] bg-[var(--lb-primary-soft)] p-3 text-left transition"
                      : "rounded-md border border-[var(--lb-border-strong)] bg-[var(--lb-surface)] p-3 text-left transition hover:border-[var(--lb-primary)]"
                  }
                >
                  <p className={`text-[13px] font-semibold ${selected ? "text-[var(--lb-primary)]" : "text-[var(--lb-text)]"}`}>
                    {methodLabel(m.value)}
                  </p>
                  <p className="mt-0.5 text-xs leading-relaxed text-[var(--lb-muted)]">{methodDescription(m.value)}</p>
                </button>
              )
            })}
          </div>
        </div>

        {METHOD_PARAMS[method].length > 0 && (
          <div className="grid gap-3 rounded-lg border border-[var(--lb-border)] bg-[var(--lb-surface-2)] p-4 sm:grid-cols-2">
            <p className="lb-muted col-span-full text-xs font-semibold">{t("参数", "Parameters")}</p>
            {METHOD_PARAMS[method].map((param) => (
              <TextField
                key={param.key}
                label={t(param.labelZh, param.labelEn)}
                placeholder={param.placeholder ?? param.default ?? ""}
                value={paramValues[param.key] ?? param.default ?? ""}
                onChange={(e) => setParam(param.key, e.target.value)}
              />
            ))}
          </div>
        )}

        {method === "quantize_onnx_static" && (
          <SelectField
            label={t("校准数据集（平台数据集，可选）", "Calibration dataset (optional)")}
            value={calibDatasetId}
            onChange={(e) => setCalibDatasetId(e.target.value)}
            hint={t("需为 zip/tar 图片包（Excel 等其它类型无法参与量化校准）", "Needs zip/tar images; other file types cannot calibrate" )}
          >
            <option value="">{t("不使用", "None")}</option>
            {datasets.map((d) => (
              <option key={d.id} value={d.id}>{d.name}（{formatBytes(d.size_bytes)}）</option>
            ))}
          </SelectField>
        )}
        {method === "qat" && (
          <SelectField
            label={t("微调数据集（必选，按类别文件夹结构）", "Fine-tune dataset (required, class folders)")}
            value={qatDatasetId}
            onChange={(e) => setQatDatasetId(e.target.value)}
            hint={t("需为按类别组织的 zip/tar 图片包；Excel 无法作为 QAT 训练集", "Needs class-organised zip/tar images" )}
          >
            <option value="">{t("选择微调数据集", "Choose dataset")}</option>
            {datasets.map((d) => (
              <option key={d.id} value={d.id}>{d.name}（{formatBytes(d.size_bytes)}）</option>
            ))}
          </SelectField>
        )}

        <div className="rounded-lg border border-[var(--lb-border)] p-4">
          <p className="mb-3 text-[13px] font-semibold text-[var(--lb-text-2)]">
            {t("④ 精度评估（可选）", "4. Accuracy eval (optional)")}
          </p>
          <div className="grid gap-3 sm:grid-cols-2">
            <SelectField label={t("评估类型", "Eval type")} value={evalType} onChange={(e) => setEvalType(e.target.value)}>
              {EVAL_TYPES.map((item) => (
                <option key={item.value} value={item.value}>{t(item.labelZh, item.labelEn)}</option>
              ))}
            </SelectField>
            {evalType === "classification" && (
              <TextField label={t("图像尺寸", "Image size")} value={evalImageSize} onChange={(e) => setEvalImageSize(e.target.value)} />
            )}
          </div>
          {evalType && (
            <div className="mt-3 grid gap-3 sm:grid-cols-2">
              <SelectField label={t("或使用平台数据集", "Or platform dataset")} value={evalDatasetId} onChange={(e) => setEvalDatasetId(e.target.value)}>
                <option value="">{t("不使用", "None")}</option>
                {datasets.map((d) => (
                  <option key={d.id} value={d.id}>{d.name}（{formatBytes(d.size_bytes)}）</option>
                ))}
              </SelectField>
              <TextField
                label={t("或数据集 URL（zip/tar）", "Or dataset URL (zip/tar)")}
                value={evalDataUrl}
                onChange={(e) => setEvalDataUrl(e.target.value)}
                placeholder="https://…"
              />
            </div>
          )}
        </div>

        <div className="rounded-lg border border-[var(--lb-border)] p-4">
          <label className="flex cursor-pointer items-center gap-2 text-[13px] font-semibold text-[var(--lb-text-2)]">
            <input
              type="checkbox"
              checked={benchmarkOn}
              onChange={(e) => setBenchmarkOn(e.target.checked)}
              className="size-4 accent-[var(--lb-primary)]"
            />
            {t("⑤ 压缩前后性能基准（CPU 延迟 / 吞吐 / 峰值内存）", "5. CPU benchmark (latency / throughput / memory)")}
          </label>
          {benchmarkOn && (
            <div className="mt-3 grid gap-3 sm:grid-cols-2">
              <TextField label={t("输入形状", "Input shape")} value={benchmarkShape} onChange={(e) => setBenchmarkShape(e.target.value)} />
              <TextField label={t("迭代次数", "Iterations")} value={benchmarkIters} onChange={(e) => setBenchmarkIters(e.target.value)} />
            </div>
          )}
        </div>
      </div>
    </Modal>
  )
}

/* ------------------------------- 页面 -------------------------------- */
function CompressionsPage() {
  const { model: presetModelId } = Route.useSearch()
  const navigate = useNavigate()
  const { t } = useI18n()
  const [models, setModels] = useState<ModelPublic[]>([])
  const [datasets, setDatasets] = useState<DatasetPublic[]>([])
  const [statusFilter, setStatusFilter] = useState<"all" | TaskStatus>("all")
  const [wizardOpen, setWizardOpen] = useState(false)
  const [notice, setNotice] = useState("")
  const [loadError, setLoadError] = useState("")

  const { data: tasksPage, loading, refresh } = usePolling(
    () => apiListTasks({ limit: 200 }),
    [],
    { intervalMs: 3000 },
  )

  const loadMeta = useCallback(async () => {
    try {
      const [modelsPage, datasetsPage] = await Promise.all([
        apiListModels({ limit: 100 }),
        apiListDatasets(100),
      ])
      setModels(modelsPage.data)
      setDatasets(datasetsPage.data)
    } catch (e) {
      setLoadError(e instanceof Error ? e.message : String(e))
    }
  }, [])

  useEffect(() => {
    void loadMeta()
  }, [loadMeta])

  const tasks = tasksPage?.data ?? []
  const filtered =
    statusFilter === "all" ? tasks : tasks.filter((task) => task.status === statusFilter)
  const counts = useMemo(() => {
    const map: Record<string, number> = { all: tasks.length }
    for (const task of tasks) map[task.status] = (map[task.status] ?? 0) + 1
    return map
  }, [tasks])

  const handleError = (e: unknown) =>
    setLoadError(e instanceof Error ? e.message : String(e))

  const onRetry = async (task: TaskPublic) => {
    try {
      await apiRetryTask(task.id)
      setNotice(t("任务已重新入队", "Task re-queued"))
      refresh()
    } catch (e) {
      handleError(e)
    }
  }

  const onDelete = async (task: TaskPublic) => {
    if (!window.confirm(`${t("删除压缩任务", "Delete task")}？`)) return
    try {
      await apiDeleteTask(task.id)
      setNotice(t("任务已删除", "Task deleted"))
      refresh()
    } catch (e) {
      handleError(e)
    }
  }

  const filters: { key: "all" | TaskStatus; zh: string; en: string }[] = [
    { key: "all", zh: "全部", en: "All" },
    { key: "pending", zh: "排队", en: "Queued" },
    { key: "running", zh: "运行中", en: "Running" },
    { key: "succeeded", zh: "已完成", en: "Completed" },
    { key: "failed", zh: "失败", en: "Failed" },
    { key: "canceled", zh: "已取消", en: "Canceled" },
  ]

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="lb-section">{t("压缩任务", "Compression Lab")}</p>
          <h1 className="mt-1 text-2xl font-bold tracking-tight text-[var(--lb-text)]">
            {t("压缩任务", "Compressions")}
          </h1>
          <p className="lb-muted mt-1 text-sm">
            {t("数据来自真实 GET /api/v1/compressions · 每 3s 自动刷新", "Real GET /api/v1/compressions · auto refresh 3s")}
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" onClick={() => void loadMeta()} disabled={loading}>
            <RefreshCw size={14} className={loading ? "animate-spin" : ""} /> {t("刷新", "Refresh")}
          </Button>
          <Button onClick={() => setWizardOpen(true)}>
            <PlusCircle size={15} /> {t("新建压缩", "New Task")}
          </Button>
        </div>
      </div>

      {notice && (
        <div className="rounded-md border border-[var(--lb-success)]/40 bg-[var(--lb-success-soft)] px-4 py-2.5 text-sm text-[var(--lb-success)]">{notice}</div>
      )}
      {loadError && (
        <div className="rounded-md border border-[var(--lb-danger)]/30 bg-[var(--lb-danger-soft)] px-4 py-2.5 text-sm text-[var(--lb-danger)]">{loadError}</div>
      )}

      <div className="flex flex-wrap gap-2">
        {filters.map((f) => {
          const active = statusFilter === f.key
          return (
            <button
              key={f.key}
              type="button"
              onClick={() => setStatusFilter(f.key)}
              className={
                active
                  ? "rounded-full border-2 border-[var(--lb-primary)] bg-[var(--lb-primary-soft)] px-3.5 py-1.5 text-[13px] font-semibold text-[var(--lb-primary)]"
                  : "rounded-full border border-[var(--lb-border-strong)] bg-[var(--lb-surface)] px-3.5 py-1.5 text-[13px] font-medium text-[var(--lb-text-2)] transition hover:border-[var(--lb-primary)]"
              }
            >
              {t(f.zh, f.en)} {counts[f.key] ?? 0}
            </button>
          )
        })}
      </div>

      <Card className="overflow-hidden p-0">
        <div className="overflow-x-auto">
          <table className="lb-table">
            <thead>
              <tr>
                <th className="lb-th">{t("模型 / 方法", "Model / Method")}</th>
                <th className="lb-th">{t("状态", "Status")}</th>
                <th className="lb-th w-[220px]">{t("进度", "Progress")}</th>
                <th className="lb-th">{t("压缩比", "Ratio")}</th>
                <th className="lb-th">{t("大小", "Size")}</th>
                <th className="lb-th">{t("创建时间", "Created")}</th>
                <th className="lb-th text-right">{t("操作", "Actions")}</th>
              </tr>
            </thead>
            <tbody>
              {filtered.length === 0 && (
                <tr>
                  <td colSpan={7}>
                    <EmptyState
                      title={t("暂无任务", "No tasks")}
                      description={t("点击右上角「新建压缩」开始第一个任务", "Click New Task to create the first one")}
                    />
                  </td>
                </tr>
              )}
              {filtered.map((task) => {
                const ratio =
                  task.status === "succeeded" && typeof task.metrics?.ratio === "number"
                    ? `${task.metrics.ratio.toFixed(2)}x`
                    : null
                return (
                  <tr key={task.id} className="lb-tr">
                    <td className="lb-td">
                      <p className="font-semibold text-[var(--lb-text)]">
                        {task.model_name ?? task.model_id.slice(0, 8)}
                      </p>
                      <p className="mt-0.5 flex items-center gap-1.5 text-xs text-[var(--lb-muted)]">
                        <FlaskConical size={12} /> {methodLabel(task.method)}
                      </p>
                    </td>
                    <td className="lb-td">
                      <Pill tone={TASK_STATUS_META[task.status].tone}>{taskStatusLabel(task.status)}</Pill>
                    </td>
                    <td className="lb-td">
                      {task.status === "running" || task.status === "pending" ? (
                        <div className="flex items-center gap-2.5">
                          <div className="w-32"><ProgressBar value={task.progress} /></div>
                          <span className="lb-mono text-xs font-medium text-[var(--lb-text-2)]">
                            {task.progress.toFixed(0)}%
                          </span>
                        </div>
                      ) : task.status === "succeeded" ? (
                        <span className="lb-mono text-xs text-[var(--lb-success)]">100%</span>
                      ) : (
                        <span className="lb-mono text-xs text-[var(--lb-muted)]">—</span>
                      )}
                    </td>
                    <td className="lb-td">
                      <span className="lb-mono text-xs font-medium">{ratio ?? "—"}</span>
                    </td>
                    <td className="lb-td">
                      <span className="lb-mono text-xs">{formatBytes(task.output_size_bytes ?? 0)}</span>
                    </td>
                    <td className="lb-td">
                      <span className="text-xs text-[var(--lb-muted)]">{formatDateTime(task.created_at)}</span>
                    </td>
                    <td className="lb-td">
                      <div className="flex items-center justify-end gap-1.5">
                        {task.status === "succeeded" && (task.metrics?.benchmark || task.metrics?.eval) && (
                          <Button
                            size="sm"
                            variant="ghost"
                            onClick={() => navigate({ to: "/lab/evaluate", search: { task: task.id } })}
                          >
                            {t("对比", "Compare")}
                          </Button>
                        )}
                        {task.status === "succeeded" && (
                          <Button
                            size="sm"
                            variant="ghost"
                            aria-label={t("下载压缩产物", "Download artifact")}
                            onClick={() => void downloadBlob(taskDownloadPath(task.id), `${task.method}_result`)}
                          >
                            <Download size={14} />
                          </Button>
                        )}
                        {(task.status === "failed" || task.status === "canceled") && (
                          <Button size="sm" variant="ghost" onClick={() => void onRetry(task)}>
                            <RotateCcw size={14} /> {t("重试", "Retry")}
                          </Button>
                        )}
                        <Button size="sm" variant="danger" aria-label={t("删除任务", "Delete task")} onClick={() => void onDelete(task)}>
                          <Trash2 size={14} />
                        </Button>
                      </div>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </Card>

      {wizardOpen && models.length > 0 && (
        <TaskWizard
          open
          presetModelId={presetModelId}
          models={models}
          datasets={datasets}
          onClose={() => setWizardOpen(false)}
          onCreated={() => {
            setNotice(t("压缩任务已创建，worker 正在后台处理", "Task created. Worker is processing it."))
            refresh()
          }}
        />
      )}
    </div>
  )
}
