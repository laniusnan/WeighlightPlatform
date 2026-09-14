/**
 * services/api.ts
 * 平台真实 API 封装（axios）。所有函数对应 backend 实际路由，无 mock。
 */
import { clearToken, http, setToken } from "./http"
import type {
  CompressionTaskCreate,
  DatasetPublic,
  DatasetsPage,
  ModelFramework,
  ModelPublic,
  ModelsPage,
  ModelStatus,
  PlatformStatus,
  ProgressHistoryResponse,
  TaskPublic,
  TasksPage,
  TaskStatus,
  UserPublic,
  WorkersStatus,
} from "./types"

/* ----------------------------- 用户 / 认证 ----------------------------- */
export const apiMe = () =>
  http.get<UserPublic>("/api/v1/users/me").then((r) => r.data)

export const apiLogin = async (
  email: string,
  password: string,
  remember = true,
) => {
  const form = new URLSearchParams()
  form.append("username", email)
  form.append("password", password)
  const res = await http.post<{ access_token: string; token_type: string }>(
    "/api/v1/login/access-token",
    form,
    { headers: { "Content-Type": "application/x-www-form-urlencoded" } },
  )
  setToken(res.data.access_token, remember)
  return res.data
}

export const apiSignUp = (input: {
  email: string
  full_name: string
  password: string
}) =>
  http
    .post<UserPublic>("/api/v1/users/signup", {
      email: input.email.toLowerCase(),
      full_name: input.full_name,
      password: input.password,
    })
    .then((r) => r.data)

export const apiLogout = () => clearToken()

/* ----------------------------- 状态 / Worker --------------------------- */
export const apiPlatformStatus = () =>
  http.get<PlatformStatus>("/api/v1/utils/platform-status/").then((r) => r.data)

export const apiWorkers = () =>
  http.get<WorkersStatus>("/api/v1/utils/workers/").then((r) => r.data)
export const apiSelfcheckFitNets = () =>
  http
    .post<Record<string, unknown>>("/api/v1/utils/selfcheck/fitnets")
    .then((r) => r.data)

export const apiSelfcheckRun = (check: string) =>
  http
    .post<Record<string, unknown>>("/api/v1/utils/selfcheck/run", { check })
    .then((r) => r.data)


/* ----------------------------- 模型管理 ------------------------------- */
export interface ListModelsQuery {
  limit?: number
  offset?: number
  status?: ModelStatus
}

export const apiListModels = (query: ListModelsQuery = {}) =>
  http
    .get<ModelsPage>("/api/v1/models/", { params: { limit: 100, ...query } })
    .then((r) => r.data)

/** 拉取当前用户全部模型（分页循环，供前端真实搜索使用） */
export const apiFetchAllModels = async (): Promise<ModelsPage> => {
  const all: ModelPublic[] = []
  const pageSize = 100
  for (let offset = 0; offset < 2000; offset += pageSize) {
    const page = await apiListModels({ limit: pageSize, offset })
    all.push(...page.data)
    if (all.length >= page.count || page.data.length < pageSize) break
  }
  return { data: all, count: all.length }
}

export interface UploadModelInput {
  file: File
  name: string
  framework: ModelFramework
  description?: string
  archFile?: File
}

export const apiUploadModel = async (input: UploadModelInput) => {
  const form = new FormData()
  form.append("file", input.file)
  form.append("name", input.name)
  form.append("framework", input.framework)
  if (input.description) form.append("description", input.description)
  if (input.archFile) form.append("arch_file", input.archFile)
  return http
    .post<ModelPublic>("/api/v1/models/upload", form)
    .then((r) => r.data)
}

export interface PullModelInput {
  name: string
  framework: ModelFramework
  source_url: string
  description?: string
}

export const apiPullModel = (input: PullModelInput) =>
  http
    .post<ModelPublic>("/api/v1/models/pull", {
      name: input.name,
      framework: input.framework,
      source_url: input.source_url,
      description: input.description ?? null,
    })
    .then((r) => r.data)

export const apiDeleteModel = (id: string) =>
  http.delete(`/api/v1/models/${id}`).then((r) => r.data)


export interface ChatTurn {
  role: "user" | "assistant"
  content: string
}

export interface ChatSampling {
  max_new_tokens: number
  temperature: number
  top_p: number
  min_p: number
}

export const apiChatCompletion = async (
  taskId: string,
  messages: ChatTurn[],
  sampling: ChatSampling,
): Promise<{ content: string }> => {
  const data = await http
    .post<{
      choices?: { message?: { content?: string } }[]
    }>("/api/v1/inference/chat/completions", {
      task_id: taskId,
      messages,
      stream: false,
      ...sampling,
    })
    .then((r) => r.data)
  return { content: data.choices?.[0]?.message?.content ?? "" }
}

export const modelDownloadPath = (id: string) => `/api/v1/models/${id}/download`

export interface ImportLocalModelInput {
  name?: string
  /** Path inside the container under a read-only local root, e.g. /models-local/Qwen2.5-0.5B-Instruct */
  server_path: string
  framework: ModelFramework
  description?: string
}

export interface LocalDirInfo {
  name: string
  path: string
  has_config: boolean
  weight_files: number
}

export const apiListLocalModelDirs = () =>
  http
    .get<{ roots: string[]; dirs: LocalDirInfo[] }>("/api/v1/models/local-dirs")
    .then((r) => r.data)

export const apiImportLocalModel = (input: ImportLocalModelInput) =>
  http
    .post<ModelPublic>("/api/v1/models/import-local", {
      name: input.name ?? null,
      server_path: input.server_path,
      framework: input.framework,
      description: input.description ?? null,
    })
    .then((r) => r.data)

/* ----------------------------- 数据集 -------------------------------- */
export const apiListDatasets = (limit = 100) =>
  http
    .get<DatasetsPage>("/api/v1/datasets/", { params: { limit } })
    .then((r) => r.data)

export const apiDeleteDataset = (id: string) =>
  http.delete(`/api/v1/datasets/${id}`).then((r) => r.data)

export const apiUploadDataset = async (file: File, name: string) => {
  const form = new FormData()
  form.append("file", file)
  form.append("name", name)
  return http
    .post<DatasetPublic>("/api/v1/datasets/upload", form)
    .then((r) => r.data)
}

/* ----------------------------- 压缩任务 ------------------------------- */
export interface ListTasksQuery {
  limit?: number
  offset?: number
  status?: TaskStatus
}

export const apiListTasks = (query: ListTasksQuery = {}) =>
  http
    .get<TasksPage>("/api/v1/compressions/", {
      params: { limit: 100, ...query },
    })
    .then((r) => r.data)

export const apiCreateTask = (body: CompressionTaskCreate) =>
  http
    .post<TaskPublic>("/api/v1/compressions/", body)
    .then((r) => r.data)

export const apiGetTask = (id: string) =>
  http.get<TaskPublic>(`/api/v1/compressions/${id}`).then((r) => r.data)

export const apiGetTaskProgress = (id: string) =>
  http
    .get<ProgressHistoryResponse>(`/api/v1/compressions/${id}/progress-history`)
    .then((r) => r.data)

export const apiReevaluateTask = (id: string, kind: "eval" | "benchmark") =>
  http
    .post<{ queued: boolean; kind: string }>(`/api/v1/compressions/${id}/reevaluate`, { kind })
    .then((r) => r.data)

export const apiRetryTask = (id: string) =>
  http.post<TaskPublic>(`/api/v1/compressions/${id}/retry`).then((r) => r.data)

export const apiDeleteTask = (id: string) =>
  http.delete(`/api/v1/compressions/${id}`).then((r) => r.data)

export const taskDownloadPath = (id: string) =>
  `/api/v1/compressions/${id}/download`

export const taskDeployPath = (id: string) =>
  `/api/v1/compressions/${id}/deploy-export`
