/**
 * Typed client for the model compression platform endpoints.
 *
 * Uses plain `fetch` (with the same Bearer token as the generated client) so it
 * works without regenerating the OpenAPI client.
 */

export type ModelFramework = "pytorch" | "onnx" | "huggingface" | "tensorflow"
export type ModelSourceType = "upload" | "url"
export type ModelStatus = "pending" | "ready" | "failed"
export type CompressionMethod =
  | "export_onnx"
  | "quantize_onnx"
  | "quantize_onnx_static"
  | "openvino_int8"
  | "prune"
  | "sparsify"
  | "qat"
  | "export_gguf"
  | "llm_quantize"
export type TaskStatus =
  | "pending"
  | "running"
  | "succeeded"
  | "failed"
  | "canceled"

export interface ModelPublic {
  id: string
  name: string
  framework: ModelFramework
  description: string | null
  source_type: ModelSourceType
  source_url: string | null
  status: ModelStatus
  size_bytes: number
  version: number
  tags: string | null
  error_message: string | null
  created_at: string | null
}

export interface ModelsPublic {
  data: ModelPublic[]
  count: number
}

export interface CompressionTaskPublic {
  id: string
  model_id: string
  model_name: string | null
  method: CompressionMethod
  params: Record<string, unknown>
  status: TaskStatus
  progress: number
  error_message: string | null
  output_size_bytes: number | null
  input_size_bytes: number | null
  metrics: Record<string, unknown>
  created_at: string | null
  started_at: string | null
  finished_at: string | null
}

export interface CompressionTasksPublic {
  data: CompressionTaskPublic[]
  count: number
}

const API_BASE = (import.meta.env.VITE_API_URL ?? "").replace(/\/+$/, "")
const PREFIX = "/api/v1"

function authHeaders(): Record<string, string> {
  const token = localStorage.getItem("access_token")
  return token ? { Authorization: `Bearer ${token}` } : {}
}

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const res = await fetch(`${API_BASE}${PREFIX}${path}`, {
    ...options,
    headers: { ...authHeaders(), ...(options.headers ?? {}) },
  })
  if (!res.ok) {
    let detail: unknown = res.statusText
    try {
      const body = await res.json()
      detail = (body as { detail?: unknown }).detail ?? detail
    } catch {
      // keep statusText
    }
    const message = typeof detail === "string" ? detail : JSON.stringify(detail)
    throw new Error(message || `Request failed with status ${res.status}`)
  }
  if (res.status === 204) return undefined as T
  return (await res.json()) as T
}

function withQuery(path: string, params: Record<string, string | undefined>) {
  const qs = new URLSearchParams()
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== "") qs.set(key, value)
  }
  const suffix = qs.toString()
  return suffix ? `${path}?${suffix}` : path
}

// ---- Models ---------------------------------------------------------------

export function listModels(
  params: { skip?: number; limit?: number; status?: ModelStatus } = {},
): Promise<ModelsPublic> {
  return request<ModelsPublic>(
    withQuery("/models/", {
      skip: params.skip != null ? String(params.skip) : undefined,
      limit: params.limit != null ? String(params.limit) : undefined,
      status: params.status,
    }),
  )
}

export function uploadModel(
  file: File,
  name: string,
  framework: ModelFramework,
  description?: string,
  archFile?: File,
): Promise<ModelPublic> {
  const form = new FormData()
  form.append("file", file)
  form.append("name", name)
  form.append("framework", framework)
  if (description) form.append("description", description)
  if (archFile) form.append("arch_file", archFile)
  return request<ModelPublic>("/models/upload", {
    method: "POST",
    body: form,
  })
}

export function pullModel(data: {
  name: string
  framework: ModelFramework
  source_url: string
  description?: string
}): Promise<ModelPublic> {
  return request<ModelPublic>("/models/pull", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  })
}

export function deleteModel(id: string): Promise<{ message: string }> {
  return request(`/models/${id}`, { method: "DELETE" })
}

export function modelDownloadPath(id: string): string {
  return `${API_BASE}${PREFIX}/models/${id}/download`
}

// ---- Compression tasks ----------------------------------------------------

export function createCompression(data: {
  model_id: string
  method: CompressionMethod
  params: Record<string, unknown>
}): Promise<CompressionTaskPublic> {
  return request<CompressionTaskPublic>("/compressions/", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  })
}

export function listCompressions(
  params: { skip?: number; limit?: number; status?: TaskStatus } = {},
): Promise<CompressionTasksPublic> {
  return request<CompressionTasksPublic>(
    withQuery("/compressions/", {
      skip: params.skip != null ? String(params.skip) : undefined,
      limit: params.limit != null ? String(params.limit) : undefined,
      status: params.status,
    }),
  )
}

export function getCompression(id: string): Promise<CompressionTaskPublic> {
  return request(`/compressions/${id}`)
}

export function retryCompression(id: string): Promise<CompressionTaskPublic> {
  return request(`/compressions/${id}/retry`, { method: "POST" })
}

export function deleteCompression(id: string): Promise<{ message: string }> {
  return request(`/compressions/${id}`, { method: "DELETE" })
}

export function compressionDownloadPath(id: string): string {
  return `${API_BASE}${PREFIX}/compressions/${id}/download`
}

export function getTaskProgress(id: string): Promise<{
  task_id: string
  status: string
  progress: number
  history: number[][]
}> {
  return request(`/compressions/${id}/progress-history`)
}

export function compressionDeployPath(id: string): string {
  return `${API_BASE}${PREFIX}/compressions/${id}/deploy-export`
}

// ---- Downloads ------------------------------------------------------------

export async function downloadBlob(
  url: string,
  filename: string,
): Promise<void> {
  const res = await fetch(url, { headers: authHeaders() })
  if (!res.ok) throw new Error(`Download failed with status ${res.status}`)
  const blob = await res.blob()
  const objectUrl = URL.createObjectURL(blob)
  const anchor = document.createElement("a")
  anchor.href = objectUrl
  anchor.download = filename
  document.body.appendChild(anchor)
  anchor.click()
  anchor.remove()
  URL.revokeObjectURL(objectUrl)
}

// ---- Datasets -------------------------------------------------------------

export interface DatasetPublic {
  id: string
  name: string
  description: string | null
  source_type: string
  source_url: string | null
  status: string
  size_bytes: number
  created_at: string | null
}

export interface DatasetsPublic {
  data: DatasetPublic[]
  count: number
}

export function listDatasets(
  params: { skip?: number; limit?: number } = {},
): Promise<DatasetsPublic> {
  return request<DatasetsPublic>(
    withQuery("/datasets/", {
      skip: params.skip != null ? String(params.skip) : undefined,
      limit: params.limit != null ? String(params.limit) : undefined,
    }),
  )
}

export function uploadDataset(
  file: File,
  name: string,
  description?: string,
): Promise<DatasetPublic> {
  const form = new FormData()
  form.append("file", file)
  form.append("name", name)
  if (description) form.append("description", description)
  return request<DatasetPublic>("/datasets/upload", {
    method: "POST",
    body: form,
  })
}

export function deleteDataset(id: string): Promise<{ message: string }> {
  return request(`/datasets/${id}`, { method: "DELETE" })
}

// ---- Formatting helpers ---------------------------------------------------

export interface PlatformStatus {
  redis: string
  tasks: Record<string, number>
  models: number
  datasets: number
  storage_bytes: number
}

export function getPlatformStatus(): Promise<PlatformStatus> {
  return request<PlatformStatus>("/utils/platform-status/")
}

export interface WorkerInfo {
  id: string
  pid: number
  started_at: number | null
  queue: string
  uptime_s: number
}

export interface WorkersStatus {
  redis: string
  workers: WorkerInfo[]
  worker_count: number
  queue_depth: number
  running_jobs: number
}

export function getWorkers(): Promise<WorkersStatus> {
  return request<WorkersStatus>("/utils/workers/")
}

export function formatBytes(bytes: number | null | undefined): string {
  if (bytes == null) return "-"
  if (bytes === 0) return "0 B"
  const units = ["B", "KB", "MB", "GB", "TB"]
  const i = Math.min(
    Math.floor(Math.log(bytes) / Math.log(1024)),
    units.length - 1,
  )
  return `${(bytes / 1024 ** i).toFixed(i === 0 ? 0 : 2)} ${units[i]}`
}

export function formatDateTime(value: string | null | undefined): string {
  if (!value) return "-"
  return new Date(value).toLocaleString()
}
