/**
 * services/types.ts
 * 与后端真实 schema 一一对应的 TS 类型（无任何 mock）。
 */

export type ID = string

export type ModelFramework = "pytorch" | "onnx" | "huggingface" | "tensorflow"
export type ModelStatus = "pending" | "ready" | "failed"
export type ModelSourceType = "upload" | "url"

export interface ModelPublic {
  id: ID
  name: string
  description?: string | null
  version: number
  tags?: string | null
  framework: ModelFramework
  source_type: ModelSourceType
  source_url?: string | null
  status: ModelStatus
  size_bytes: number
  error_message?: string | null
  created_at?: string | null
}

export interface ModelsPage {
  data: ModelPublic[]
  count: number
}

export type TaskStatus =
  | "pending"
  | "running"
  | "succeeded"
  | "failed"
  | "canceled"

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

export interface BenchmarkSide {
  latency_ms_mean?: number
  latency_ms_p50?: number
  latency_ms_p95?: number
  latency_ms_p99?: number
  latency_ms_min?: number
  latency_ms_max?: number
  latency_ms_std?: number
  throughput_ips?: number
  peak_memory_mb?: number
  memory_delta_mb?: number
  iterations?: number
}

export interface Benchmark {
  original?: BenchmarkSide
  compressed?: BenchmarkSide
  speedup_x?: number
  original_error?: string
  compressed_error?: string
  error?: string
}

export interface EvalMetrics {
  error?: string
  // 图像分类
  accuracy?: number
  baseline_accuracy?: number
  accuracy_delta?: number
  top1?: number
  // YOLO
  map50?: number
  map50_95?: number
  precision?: number
  recall?: number
  // LLM
  perplexity?: number
  [key: string]: unknown
}

export interface TaskMetrics {
  ratio?: number
  size_reduction_pct?: number
  input_bytes?: number
  output_bytes?: number
  eval?: EvalMetrics
  benchmark?: Benchmark
  progress_history?: number[][]
  [key: string]: unknown
}

export interface TaskPublic {
  id: ID
  model_id: ID
  model_name?: string | null
  method: CompressionMethod
  params: Record<string, unknown>
  status: TaskStatus
  progress: number
  error_message?: string | null
  output_size_bytes?: number | null
  input_size_bytes?: number | null
  metrics: TaskMetrics
  created_at?: string | null
  started_at?: string | null
  finished_at?: string | null
}

export interface TasksPage {
  data: TaskPublic[]
  count: number
}

export interface ProgressHistoryResponse {
  task_id: ID
  status: TaskStatus
  progress: number
  history: number[][]
}

export interface DatasetPublic {
  id: ID
  name: string
  description?: string | null
  source_type: string
  source_url?: string | null
  status: string
  size_bytes: number
  created_at?: string | null
}

export interface DatasetsPage {
  data: DatasetPublic[]
  count: number
}

export interface UserPublic {
  id: ID
  email: string
  full_name?: string | null
  is_active: boolean
  is_superuser: boolean
}

export interface WorkerInfo {
  id?: string
  pid?: number
  uptime_s?: number
  [key: string]: unknown
}

export interface WorkersStatus {
  redis: "up" | "down"
  workers: WorkerInfo[]
  worker_count: number
  queue_depth: number
  running_jobs: number
}

export interface PlatformStatus {
  redis: "up" | "down"
  tasks: Record<string, number>
  models: number
  datasets: number
  storage_bytes: number
}

/** 新建压缩任务的请求体（与 POST /api/v1/compressions 一致） */
export interface CompressionTaskCreate {
  model_id: ID
  method: CompressionMethod
  params: Record<string, unknown>
}
