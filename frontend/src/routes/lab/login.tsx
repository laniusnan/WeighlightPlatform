/**
 * routes/lab/login.tsx · 新版登录（双语）
 * 真实 POST /api/v1/login/access-token；支持“记住我”。
 */
import { createFileRoute, Link, redirect, useNavigate } from "@tanstack/react-router"
import { AlertCircle, FlaskConical, Loader2, Lock, Mail } from "lucide-react"
import { useState } from "react"

import { useI18n } from "@/components/lab/i18n"
import { AuthBar } from "@/components/lab/AuthBar"
import { isLoggedIn } from "@/hooks/useAuth"
import { apiLogin } from "@/services/api"
import { apiErrorMessage } from "@/services/http"

export const Route = createFileRoute("/lab/login")({
  beforeLoad: async () => {
    if (isLoggedIn()) {
      throw redirect({ to: "/lab/dashboard" })
    }
  },
  component: LoginPage,
  head: () => ({ meta: [{ title: "Sign in - AI Model Lightweight Platform" }] }),
})

function LoginPage() {
  const navigate = useNavigate()
  const { t } = useI18n()
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [remember, setRemember] = useState(true)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState("")

  const submit = async (event: React.FormEvent) => {
    event.preventDefault()
    if (!email.trim() || !password) {
      setError(t("请输入邮箱与密码", "Please enter email and password"))
      return
    }
    setBusy(true)
    setError("")
    try {
      await apiLogin(email.trim(), password, remember)
      navigate({ to: "/lab/dashboard" })
    } catch (e) {
      setError(apiErrorMessage(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="relative w-full max-w-[400px]">
      <AuthBar />
      <div className="mb-8 flex flex-col items-center gap-3 text-center">
        <span className="flex size-12 items-center justify-center rounded-xl bg-[var(--lb-primary)] text-[var(--lb-primary-foreground)]">
          <FlaskConical size={24} />
        </span>
        <div>
          <p className="text-xl font-bold tracking-tight text-[var(--lb-text)]">
            AI Model <span className="text-[var(--lb-primary)]">{t("轻量化平台", "Lightweight")}</span>
          </p>
          <p className="lb-muted mt-1 text-sm">
            {t("模型压缩 · 量化 · 评估 · 部署", "Compression · Quantization · Eval · Deploy")}
          </p>
        </div>
      </div>

      <div className="lb-card lb-card-white rounded-[10px] p-7 shadow-[var(--lb-shadow-md)]">
        <h1 className="text-lg font-bold tracking-tight text-[var(--lb-text)]">
          {t("登录工作台", "Sign in")}
        </h1>
        <p className="lb-muted mt-1 text-[13px]">
          {t("使用平台账号登录（真实后端认证）", "Sign in with your platform account")}
        </p>

        <form onSubmit={(e) => void submit(e)} className="mt-6 grid gap-4">
          <div>
            <label className="lb-label" htmlFor="lb-email">
              {t("邮箱", "Email")}
            </label>
            <div className="relative">
              <Mail size={15} className="absolute left-3 top-1/2 -translate-y-1/2 text-[var(--lb-muted)]" />
              <input
                id="lb-email"
                type="email"
                autoComplete="email"
                autoFocus
                className="lb-input lb-input--icon"
                placeholder="user@example.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
              />
            </div>
          </div>

          <div>
            <label className="lb-label" htmlFor="lb-password">
              {t("密码", "Password")}
            </label>
            <div className="relative">
              <Lock size={15} className="absolute left-3 top-1/2 -translate-y-1/2 text-[var(--lb-muted)]" />
              <input
                id="lb-password"
                type="password"
                autoComplete="current-password"
                className="lb-input lb-input--icon"
                placeholder={t("请输入密码", "Your password")}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
              />
            </div>
          </div>

          <label className="flex cursor-pointer items-center gap-2 text-[13px] text-[var(--lb-text-2)]">
            <input
              type="checkbox"
              checked={remember}
              onChange={(e) => setRemember(e.target.checked)}
              className="size-4 accent-[var(--lb-primary)]"
            />
            {t("记住我（保持登录状态）", "Remember me")}
          </label>

          {error && (
            <div className="flex items-start gap-2 rounded-md border border-[var(--lb-danger)]/30 bg-[var(--lb-danger-soft)] px-3 py-2.5 text-[13px] text-[var(--lb-danger)]">
              <AlertCircle size={15} className="mt-0.5 shrink-0" />
              <span>{error}</span>
            </div>
          )}

          <button type="submit" disabled={busy} className="lb-btn lb-btn--primary !h-10 w-full">
            {busy ? (
              <>
                <Loader2 size={15} className="animate-spin" /> {t("登录中…", "Signing in…")}
              </>
            ) : (
              t("登录", "Sign in")
            )}
          </button>
        </form>
      </div>

      <p className="lb-muted mt-5 text-center text-[13px]">
        {t("还没有账号？", "No account?")}{" "}
        <Link to="/lab/signup" className="font-medium text-[var(--lb-primary)] hover:underline">
          {t("注册账号", "Create account")}
        </Link>
      </p>
    </div>
  )
}
