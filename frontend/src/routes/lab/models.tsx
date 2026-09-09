/**
 * routes/lab/models.tsx · 模型管理 Model Zoo（双语）
 * 真实搜索 + 表格 + 上传/拉取（真实 API）。
 */
import { createFileRoute, useNavigate } from "@tanstack/react-router"
import {
  Boxes,
  CloudDownload,
  Database,
  Download,
  FolderInput,
  Link2,
  RefreshCw,
  Search,
  Trash2,
  UploadCloud,
} from "lucide-react"
import { useCallback, useEffect, useMemo, useState } from "react"

import { FRAMEWORK_LABEL } from "@/components/lab/constants"
import { useI18n } from "@/components/lab/i18n"
import {
  Button,
  Card,
  EmptyState,
  Modal,
  Pill,
  SelectField,
  TextField,
  TextAreaField,
} from "@/components/lab/ui"
import {
  apiDeleteDataset,
  apiDeleteModel,
  apiFetchAllModels,
  apiListDatasets,
  apiImportLocalModel,
  apiListLocalModelDirs,
  apiListTasks,
  apiPullModel,
  apiUploadDataset,
  apiUploadModel,
  modelDownloadPath,
} from "@/services/api"
import { downloadBlob } from "@/services/http"
import { formatBytes, formatDateTime } from "@/services/format"
import type {
  DatasetPublic,
  ModelFramework,
  ModelPublic,
  ModelStatus,
  TaskPublic,
} from "@/services/types"

export const Route = createFileRoute("/lab/models")({
  component: ModelsPage,
  head: () => ({ meta: [{ title: "Models - AI Model Lightweight Platform" }] }),
})

const FRAMEWORKS: { value: ModelFramework; label: string }[] = [
  { value: "pytorch", label: "PyTorch (.pt/.pth)" },
  { value: "onnx", label: "ONNX (.onnx)" },
  { value: "huggingface", label: "HuggingFace" },
  { value: "tensorflow", label: "TensorFlow (.h5)" },
]

const MODEL_STATUS_META: Record<
  ModelStatus,
  { label: string; en: string; tone: "pending" | "succeeded" | "failed" }
> = {
  pending: { label: "处理中", en: "Processing", tone: "pending" },
  ready: { label: "就绪", en: "Ready", tone: "succeeded" },
  failed: { label: "失败", en: "Failed", tone: "failed" },
}

function resolveInputShape(modelId: string, tasks: TaskPublic[]): string | null {
  const related = tasks.filter((t) => t.model_id === modelId && t.status === "succeeded")
  if (related.length === 0) return null
  const shapes = related
    .map((task) => {
      const raw =
        (task.params?.benchmark as { input_shape?: unknown } | undefined)?.input_shape ??
        task.params?.input_shape
      return raw ? String(raw) : null
    })
    .filter((s): s is string => Boolean(s))
  return shapes[0] ?? null
}

/* ---------------------------- 上传模型 ------------------------------ */
function UploadModelModal({
  open,
  onClose,
  onSuccess,
}: {
  open: boolean
  onClose: () => void
  onSuccess: () => void
}) {
  const { t } = useI18n()
  const [name, setName] = useState("")
  const [file, setFile] = useState<File | null>(null)
  const [archFile, setArchFile] = useState<File | null>(null)
  const [framework, setFramework] = useState<ModelFramework>("pytorch")
  const [description, setDescription] = useState("")
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState("")

  const submit = async () => {
    if (!file) {
      setError(t("请选择模型文件", "Please choose a model file"))
      return
    }
    setBusy(true)
    setError("")
    try {
      await apiUploadModel({
        file,
        name: name.trim() || file.name,
        framework,
        description: description.trim() || undefined,
        archFile: archFile ?? undefined,
      })
      setName("")
      setFile(null)
      setArchFile(null)
      setDescription("")
      onSuccess()
      onClose()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <Modal
      open={open}
      onClose={onClose}
      title={t("上传模型", "Upload Model")}
      footer={
        <>
          <Button variant="outline" onClick={onClose}>
            {t("取消", "Cancel")}
          </Button>
          <Button onClick={() => void submit()} disabled={busy}>
            {busy ? t("上传中…", "Uploading…") : t("上传", "Upload")}
          </Button>
        </>
      }
    >
      <div className="grid gap-4">
        <TextField
          label={t("模型名称", "Model name")}
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder={t("默认使用文件名", "Defaults to file name")}
        />
        <div>
          <span className="lb-label">{t("模型文件", "Model file")}</span>
          <label className="flex cursor-pointer flex-col items-center justify-center gap-1 rounded-md border border-dashed border-[var(--lb-border-strong)] bg-[var(--lb-surface-2)] px-4 py-6 text-center transition hover:border-[var(--lb-primary)]">
            <input
              type="file"
              className="hidden"
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            />
            <span className="text-sm font-medium text-[var(--lb-muted)]">
              {file ? file.name : t("点击选择模型文件", "Click to choose a model file")}
            </span>
            <span className="text-xs text-[var(--lb-muted)]">
              .pt / .pth / .onnx / .h5
            </span>
          </label>
        </div>
        {framework === "pytorch" && (
          <div>
            <span className="lb-label">{t("结构文件 arch.py（可选）", "Architecture file arch.py (optional)")}</span>
            <label className="flex cursor-pointer flex-col items-center justify-center gap-1 rounded-md border border-dashed border-[var(--lb-border-strong)] bg-[var(--lb-surface-2)] px-4 py-6 text-center transition hover:border-[var(--lb-primary)]">
              <input
                type="file"
                className="hidden"
                onChange={(e) => setArchFile(e.target.files?.[0] ?? null)}
              />
              <span className="text-sm text-[var(--lb-muted)]">
                {archFile ? archFile.name : t("点击选择 arch.py", "Click to choose arch.py")}
              </span>
            </label>
          </div>
        )}
        <SelectField label={t("框架", "Framework")} value={framework} onChange={(e) => setFramework(e.target.value as ModelFramework)}>
          {FRAMEWORKS.map((f) => (
            <option key={f.value} value={f.value}>{f.label}</option>
          ))}
        </SelectField>
        <TextAreaField label={t("描述（可选）", "Description (optional)")} value={description} onChange={(e) => setDescription(e.target.value)} />
        {error && <p className="text-sm text-[var(--lb-danger)]">{error}</p>}
      </div>
    </Modal>
  )
}

/* ---------------------------- 拉取模型 ------------------------------ */
function PullModelModal({
  open,
  onClose,
  onSuccess,
}: {
  open: boolean
  onClose: () => void
  onSuccess: () => void
}) {
  const { t } = useI18n()
  const [name, setName] = useState("")
  const [framework, setFramework] = useState<ModelFramework>("pytorch")
  const [sourceUrl, setSourceUrl] = useState("")
  const [description, setDescription] = useState("")
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState("")

  const submit = async () => {
    if (!sourceUrl.trim()) {
      setError(t("请填写下载 URL 或 HuggingFace repo id", "Please provide a URL or HuggingFace repo id"))
      return
    }
    setBusy(true)
    setError("")
    try {
      await apiPullModel({
        name: name.trim() || sourceUrl.trim().split("/").pop() || "model",
        framework,
        source_url: sourceUrl.trim(),
        description: description.trim() || undefined,
      })
      setName("")
      setSourceUrl("")
      setDescription("")
      onSuccess()
      onClose()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <Modal
      open={open}
      onClose={onClose}
      title={t("从 URL 拉取模型", "Pull from URL")}
      footer={
        <>
          <Button variant="outline" onClick={onClose}>
            {t("取消", "Cancel")}
          </Button>
          <Button onClick={() => void submit()} disabled={busy}>
            {busy ? t("提交中…", "Submitting…") : t("开始拉取", "Pull")}
          </Button>
        </>
      }
    >
      <div className="grid gap-4">
        <TextField label={t("模型名称", "Model name")} value={name} onChange={(e) => setName(e.target.value)} placeholder={t("默认取 URL 最后一段", "Defaults to URL tail")} />
        <SelectField label={t("框架", "Framework")} value={framework} onChange={(e) => setFramework(e.target.value as ModelFramework)}>
          {FRAMEWORKS.map((f) => (
            <option key={f.value} value={f.value}>{f.label}</option>
          ))}
        </SelectField>
        <TextField label={t("下载链接 / HuggingFace repo id", "URL / HuggingFace repo id")} value={sourceUrl} onChange={(e) => setSourceUrl(e.target.value)} placeholder="https://…" />
        <TextAreaField label={t("描述（可选）", "Description (optional)")} value={description} onChange={(e) => setDescription(e.target.value)} />
        {error && <p className="text-sm text-[var(--lb-danger)]">{error}</p>}
      </div>
    </Modal>
  )
}


function ImportLocalModal({
  open,
  onClose,
  onSuccess,
}: {
  open: boolean
  onClose: () => void
  onSuccess: () => void
}) {
  const { t } = useI18n()
  const [name, setName] = useState("")
  const [framework, setFramework] = useState<ModelFramework>("huggingface")
  const [serverPath, setServerPath] = useState("")
  const [description, setDescription] = useState("")
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState("")
  const [localDirs, setLocalDirs] = useState<{ name: string; path: string; has_config: boolean }[]>([])
  const [presetPath, setPresetPath] = useState("")

  useEffect(() => {
    if (!open) return
    setLocalDirs([])
    apiListLocalModelDirs()
      .then((res) => setLocalDirs(res.dirs))
      .catch(() => {
        /* server may have LOCAL_MODEL_DIRS empty; manual path still works */
      })
  }, [open])

  const submit = async () => {
    if (!serverPath.trim()) {
      setError(t("请填写服务器上的模型目录路径", "Please provide the model directory path on the server"))
      return
    }
    setBusy(true)
    setError("")
    try {
      await apiImportLocalModel({
        name: name.trim() || serverPath.trim().split("/").pop() || "model",
        framework,
        server_path: serverPath.trim(),
        description: description.trim() || undefined,
      })
      setName("")
      setServerPath("")
      setDescription("")
      onSuccess()
      onClose()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <Modal
      open={open}
      onClose={onClose}
      title={t("导入服务器本地模型", "Import server-local model")}
      footer={
        <>
          <Button variant="outline" onClick={onClose}>
            {t("取消", "Cancel")}
          </Button>
          <Button onClick={() => void submit()} disabled={busy}>
            {busy ? t("提交中…", "Submitting…") : t("开始导入", "Import")}
          </Button>
        </>
      }
    >
      <div className="grid gap-4">
        <TextField label={t("模型名称", "Model name")} value={name} onChange={(e) => setName(e.target.value)} placeholder={t("默认取目录名", "Defaults to directory name")} />
        <SelectField
          label={t("或从服务器已有目录选择", "…or pick an existing server directory")}
          value={presetPath}
          onChange={(e) => {
            const path = e.target.value
            setPresetPath(path)
            if (path) setServerPath(path)
          }}
        >
          <option value="">{localDirs.length ? t("选择目录…（也可以在下框手填）", "Choose… (or type below)") : t("（服务器没有可用目录，请手填）", "(no local dirs found - type below)")}</option>
          {localDirs.map((d) => (
            <option key={d.path} value={d.path}>
              {d.name}{" "}
              {d.has_config ? t("（HF 就绪）", " (HF ready)") : ""}
            </option>
          ))}
        </SelectField>
        <TextField
          label={t("服务器目录（挂载为 /models-local，可手填）", "Server directory (mounted at /models-local, editable)")}
          value={serverPath}
          onChange={(e) => {
            setServerPath(e.target.value)
            if (presetPath && e.target.value !== presetPath) setPresetPath("")
          }}
          placeholder="/models-local/Qwen2.5-0.5B-Instruct"
        />
        <SelectField label={t("框架", "Framework")} value={framework} onChange={(e) => setFramework(e.target.value as ModelFramework)}>
          {FRAMEWORKS.map((f) => (
            <option key={f.value} value={f.value}>{f.label}</option>
          ))}
        </SelectField>
        <TextAreaField label={t("描述（可选）", "Description (optional)")} value={description} onChange={(e) => setDescription(e.target.value)} />
        {error && <p className="text-sm text-[var(--lb-danger)]">{error}</p>}
      </div>
    </Modal>
  )
}

/* ------------------------------------------------------------------ */

function ModelsPage() {
  const navigate = useNavigate()
  const { t } = useI18n()
  const [models, setModels] = useState<ModelPublic[]>([])
  const [tasks, setTasks] = useState<TaskPublic[]>([])
  const [search, setSearch] = useState("")
  const [statusFilter, setStatusFilter] = useState<"all" | ModelStatus>("all")
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState("")
  const [notice, setNotice] = useState("")
  const [uploadOpen, setUploadOpen] = useState(false)
  const [pullOpen, setPullOpen] = useState(false)
  const [importOpen, setImportOpen] = useState(false)
  const [datasets, setDatasets] = useState<DatasetPublic[]>([])

  const load = useCallback(async () => {
    setLoading(true)
    setError("")
    try {
      const [modelsPage, tasksPage, datasetsPage] = await Promise.all([
        apiFetchAllModels(),
        apiListTasks({ limit: 300 }),
        apiListDatasets(500),
      ])
      setModels(modelsPage.data)
      setTasks(tasksPage.data)
      setDatasets(datasetsPage.data)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void load()
  }, [load])

  const uploadDataset = async (file: File) => {
    try {
      const base = (file.name || "dataset").replace(/\.(zip|tar|gz)$/i, "")
      await apiUploadDataset(file, base || "dataset")
      setNotice(t("数据集已上传", "Dataset uploaded"))
      void load()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }

  const deleteDataset = async (dataset: DatasetPublic) => {
    if (!window.confirm(`${t("删除数据集", "Delete dataset")}：${dataset.name}？`)) return
    try {
      await apiDeleteDataset(dataset.id)
      setNotice(t("数据集已删除", "Dataset deleted"))
      void load()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }

  const filtered = useMemo(() => {
    const kw = search.trim().toLowerCase()
    return models
      .filter((m) => statusFilter === "all" || m.status === statusFilter)
      .filter((m) => {
        if (!kw) return true
        const haystack = [m.name, m.framework, m.tags ?? "", m.description ?? ""]
          .join(" ")
          .toLowerCase()
        return haystack.includes(kw)
      })
  }, [models, search, statusFilter])

  const handleDelete = async (model: ModelPublic) => {
    if (!window.confirm(`${t("确认删除模型", "Delete model")}「${model.name}」？`)) return
    try {
      await apiDeleteModel(model.id)
      setNotice(`${t("已删除模型", "Deleted")} ${model.name}`)
      await load()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="lb-section">{t("模型管理", "Model Zoo")}</p>
          <h1 className="mt-1 text-2xl font-bold tracking-tight text-[var(--lb-text)]">
            {t("模型管理", "Model Management")}
          </h1>
          <p className="lb-muted mt-1 text-sm">
            {t("真实数据来自 GET /api/v1/models · 共 ", "Real data from GET /api/v1/models · ")}
            {models.length} {t("个模型", "models")}
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" onClick={() => void load()} disabled={loading}>
            <RefreshCw size={14} className={loading ? "animate-spin" : ""} /> {t("刷新", "Refresh")}
          </Button>
          <Button variant="outline" onClick={() => setImportOpen(true)}>
            <FolderInput size={14} /> {t("本地导入", "Import local")}
          </Button>
          <Button variant="outline" onClick={() => setPullOpen(true)}>
            <CloudDownload size={14} /> {t("拉取模型", "Pull model")}
          </Button>
          <Button onClick={() => setUploadOpen(true)}>
            <UploadCloud size={14} /> {t("上传模型", "Upload model")}
          </Button>
        </div>
      </div>

      {notice && (
        <div className="rounded-md border border-[var(--lb-success)]/40 bg-[var(--lb-success-soft)] px-4 py-2.5 text-sm text-[var(--lb-success)]">
          {notice}
        </div>
      )}
      {error && (
        <div className="rounded-md border border-[var(--lb-danger)]/30 bg-[var(--lb-danger-soft)] px-4 py-2.5 text-sm text-[var(--lb-danger)]">
          {error}
        </div>
      )}

      <Card className="overflow-hidden p-0">
        <div className="flex flex-wrap items-center gap-3 border-b border-[var(--lb-border)] p-4">
          <div className="relative min-w-[240px] flex-1">
            <Search size={15} className="absolute left-3 top-1/2 -translate-y-1/2 text-[var(--lb-muted)]" />
            <input
              className="lb-input lb-input--icon"
              placeholder={t("搜索模型名称 / 框架 / 标签…", "Search name / framework / tags…")}
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />
          </div>
          <select className="lb-select w-44" value={statusFilter} onChange={(e) => setStatusFilter(e.target.value as "all" | ModelStatus)}>
            <option value="all">{t("全部状态", "All status")}</option>
            <option value="ready">{t("就绪", "Ready")}</option>
            <option value="pending">{t("处理中", "Processing")}</option>
            <option value="failed">{t("失败", "Failed")}</option>
          </select>
          <span className="lb-muted text-xs">{filtered.length} / {models.length}</span>
        </div>

        <div className="overflow-x-auto">
          <table className="lb-table">
            <thead>
              <tr>
                <th className="lb-th">Name</th>
                <th className="lb-th">{t("框架", "Framework")}</th>
                <th className="lb-th">{t("状态", "Status")}</th>
                <th className="lb-th">{t("输入维度", "Input dims")}</th>
                <th className="lb-th">{t("大小", "Size")}</th>
                <th className="lb-th">{t("来源", "Source")}</th>
                <th className="lb-th">{t("创建时间", "Created")}</th>
                <th className="lb-th text-right">{t("操作", "Actions")}</th>
              </tr>
            </thead>
            <tbody>
              {filtered.length === 0 && (
                <tr>
                  <td colSpan={8}>
                    <EmptyState
                      title={models.length === 0 ? t("还没有模型", "No models yet") : t("没有匹配的模型", "No matching models")}
                      description={models.length === 0 ? t("先上传或从 URL 拉取一个模型", "Upload or pull a model first") : t("换个关键词试试", "Try another keyword")}
                    />
                  </td>
                </tr>
              )}
              {filtered.map((model) => {
                const status = MODEL_STATUS_META[model.status]
                const inputShape = resolveInputShape(model.id, tasks)
                return (
                  <tr key={model.id} className="lb-tr">
                    <td className="lb-td">
                      <div className="flex items-center gap-2">
                        <Boxes size={15} className="shrink-0 text-[var(--lb-primary)]" />
                        <div className="min-w-0">
                          <p className="truncate font-semibold text-[var(--lb-text)]">{model.name}</p>
                          {model.version > 1 && (
                            <p className="lb-mono text-[11px] text-[var(--lb-muted)]">v{model.version}</p>
                          )}
                        </div>
                      </div>
                    </td>
                    <td className="lb-td">
                      <Pill tone="info">{FRAMEWORK_LABEL[model.framework]}</Pill>
                    </td>
                    <td className="lb-td">
                      <Pill tone={status.tone}>{t(status.label, status.en)}</Pill>
                    </td>
                    <td className="lb-td">
                      <span className="lb-mono text-xs text-[var(--lb-text-2)]">{inputShape ?? "—"}</span>
                    </td>
                    <td className="lb-td">
                      <span className="lb-mono text-xs">{formatBytes(model.size_bytes)}</span>
                    </td>
                    <td className="lb-td">
                      {model.source_type === "url" ? (
                        <span className="flex items-center gap-1 text-xs text-[var(--lb-text-2)]">
                          <Link2 size={12} /> URL
                        </span>
                      ) : (
                        <span className="text-xs text-[var(--lb-text-2)]">{t("上传", "Upload")}</span>
                      )}
                    </td>
                    <td className="lb-td">
                      <span className="text-xs text-[var(--lb-muted)]">{formatDateTime(model.created_at)}</span>
                    </td>
                    <td className="lb-td">
                      <div className="flex items-center justify-end gap-1.5">
                        {model.status === "ready" && (
                          <Button
                            size="sm"
                            variant="ghost"
                            onClick={() => navigate({ to: "/lab/compressions", search: { model: model.id } })}
                          >
                            {t("创建任务", "Compress")}
                          </Button>
                        )}
                        {model.status === "ready" && (
                          <Button
                            size="sm"
                            variant="ghost"
                            aria-label={t("下载原模型", "Download")}
                            onClick={() =>
                              void downloadBlob(modelDownloadPath(model.id), `${model.name}.model`)
                            }
                          >
                            <Download size={14} />
                          </Button>
                        )}
                        <Button size="sm" variant="danger" aria-label={t("删除", "Delete")} onClick={() => void handleDelete(model)}>
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

      <Card className="p-0">
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[var(--lb-border)] px-5 py-4">
          <div>
            <h2 className="flex items-center gap-2 text-[15px] font-semibold text-[var(--lb-text)]">
              <Database size={16} className="text-[var(--lb-primary)]" />
              {t("数据集", "Datasets")}
            </h2>
            <p className="lb-muted mt-0.5 text-xs">
              {t("校准/蒸馏/评估/微调用：zip/tar 图片或文本包；也可直接上传 txt/csv/json/jsonl/xlsx/xls 文本或表格", "For calibration / distillation / eval / tuning. Archives (zip/tar) or plain text/table files (txt csv json jsonl xlsx xls)")}
            </p>
          </div>
          <label className="lb-btn lb-btn--outline cursor-pointer">
            <UploadCloud size={14} /> {t("上传数据集", "Upload dataset")}
            <input
              type="file"
              accept=".zip,.tar,.tar.gz,.tgz,.gz,.txt,.csv,.json,.jsonl,.xlsx,.xls"
              className="hidden"
              onChange={(e) => {
                const file = e.target.files?.[0]
                if (file) void uploadDataset(file)
                e.target.value = ""
              }}
            />
          </label>
        </div>
        {datasets.length === 0 ? (
          <p className="px-5 py-6 text-sm text-[var(--lb-muted)]">
            {t("暂无数据集，可上传 zip/tar 图片包或文本包。", "No datasets yet. Upload a zip/tar archive.")}
          </p>
        ) : (
          <div className="overflow-x-auto">
            <table className="lb-table">
              <thead>
                <tr>
                  <th className="lb-th">{t("名称", "Name")}</th>
                  <th className="lb-th">{t("大小", "Size")}</th>
                  <th className="lb-th">{t("创建时间", "Created")}</th>
                  <th className="lb-th text-right">{t("操作", "Actions")}</th>
                </tr>
              </thead>
              <tbody>
                {datasets.map((dataset) => (
                  <tr key={dataset.id} className="lb-tr">
                    <td className="lb-td"><p className="font-medium text-[var(--lb-text)]">{dataset.name}</p></td>
                    <td className="lb-td"><span className="lb-mono text-xs">{formatBytes(dataset.size_bytes)}</span></td>
                    <td className="lb-td"><span className="text-xs text-[var(--lb-muted)]">{formatDateTime(dataset.created_at)}</span></td>
                    <td className="lb-td">
                      <div className="flex justify-end">
                        <Button size="sm" variant="danger" aria-label={t("删除数据集", "Delete dataset")} onClick={() => void deleteDataset(dataset)}>
                          <Trash2 size={13} />
                        </Button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>

      <UploadModelModal
        open={uploadOpen}
        onClose={() => setUploadOpen(false)}
        onSuccess={() => {
          setNotice(t("模型已上传，worker 正在校验，稍后自动变为就绪", "Uploaded. Worker is validating, it will turn Ready soon."))
          void load()
        }}
      />
      <PullModelModal
        open={pullOpen}
        onClose={() => setPullOpen(false)}
        onSuccess={() => {
          setNotice(t("已提交拉取任务，模型就绪后可创建压缩任务", "Pull started. Create a task once the model is Ready."))
          void load()
        }}
      />
      <ImportLocalModal
        open={importOpen}
        onClose={() => setImportOpen(false)}
        onSuccess={() => {
          setNotice(t("本地模型已导入并注册为就绪", "Local model imported and registered as Ready."))
          void load()
        }}
      />
    </div>
  )
}
