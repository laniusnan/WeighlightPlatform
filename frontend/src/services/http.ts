/**
 * services/http.ts
 * Axios 单例：统一 baseURL + Bearer Token 注入 + 401 兜底。
 */
import axios, { AxiosError } from "axios"

export const API_BASE = (import.meta.env.VITE_API_URL as string | undefined) ?? ""

const TOKEN_KEY = "access_token"

/** “记住我”用 localStorage（持久）；否则 sessionStorage（关标签即失效） */
export function getToken(): string | null {
  return localStorage.getItem(TOKEN_KEY) ?? sessionStorage.getItem(TOKEN_KEY)
}

export function setToken(token: string, remember: boolean): void {
  clearToken()
  ;(remember ? localStorage : sessionStorage).setItem(TOKEN_KEY, token)
}

export function clearToken(): void {
  localStorage.removeItem(TOKEN_KEY)
  sessionStorage.removeItem(TOKEN_KEY)
  _sessionCacheValid = false
  _sessionCacheAt = 0
}

let _sessionCacheValid = false
let _sessionCacheAt = 0

/**
 * True only when a token exists AND the backend accepts it (GET /users/me).
 * Guards use this instead of mere token presence so stale/expired tokens do
 * not bounce users away from the login page. Result is cached for 3s.
 */
export async function isSessionValid(): Promise<boolean> {
  const now = Date.now()
  if (now - _sessionCacheAt < 3000) return _sessionCacheValid
  _sessionCacheAt = now
  if (!getToken()) {
    _sessionCacheValid = false
    return false
  }
  try {
    await http.get("/api/v1/users/me", { timeout: 8000 })
    _sessionCacheValid = true
  } catch {
    // The 401 interceptor already cleared the token for us.
    _sessionCacheValid = false
  }
  return _sessionCacheValid
}

export const http = axios.create({
  baseURL: API_BASE,
  timeout: 60_000,
})

http.interceptors.request.use((config) => {
  const token = getToken()
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

http.interceptors.response.use(
  (response) => response,
  (error: AxiosError) => {
    if (error.response?.status === 401) {
      clearToken()
      if (!window.location.pathname.startsWith("/lab/login")) {
        window.location.href = "/lab/login"
      }
    }
    return Promise.reject(error)
  },
)

/** 把后端 detail / 网络错误转为可展示文案 */
export function apiErrorMessage(error: unknown): string {
  if (axios.isAxiosError(error)) {
    const detail = error.response?.data as
      | { detail?: unknown }
      | undefined
    if (detail && typeof detail.detail === "string") return detail.detail
    if (typeof detail?.detail === "object" && detail.detail) {
      try {
        return JSON.stringify(detail.detail)
      } catch {
        /* ignore */
      }
    }
    if (error.code === "ECONNABORTED") return "请求超时，请稍后重试"
    if (!error.response) return "无法连接服务器，请确认后端已启动"
    return `请求失败（HTTP ${error.response.status}）`
  }
  return error instanceof Error ? error.message : "发生未知错误"
}

/** 服务端返回的相对文件下载地址 → 可点击的完整 URL */
export function absoluteUrl(path: string): string {
  if (!path) return path
  if (path.startsWith("http")) return path
  return `${API_BASE}${path}`
}

/** 带认证下载（backend 需要 Bearer），落盘为文件 */
export async function downloadBlob(
  path: string,
  fallbackName: string,
): Promise<void> {
  const response = await http.get(path, { responseType: "blob" })
  const disposition = response.headers["content-disposition"] as
    | string
    | undefined
  const match = disposition?.match(/filename="?([^";]+)"?/i)
  const name = match?.[1] ?? fallbackName

  const url = URL.createObjectURL(response.data as Blob)
  const anchor = document.createElement("a")
  anchor.href = url
  anchor.download = name
  document.body.appendChild(anchor)
  anchor.click()
  anchor.remove()
  URL.revokeObjectURL(url)
}
