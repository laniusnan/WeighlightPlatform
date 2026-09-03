/**
 * routes/lab/signup.tsx · 新版注册（双语）
 * 真实 POST /api/v1/users/signup + 自动登录。
 */
import { createFileRoute, Link, redirect, useNavigate } from "@tanstack/react-router"
import { AlertCircle, FlaskConical, Loader2 } from "lucide-react"
import { useState } from "react"

import { useI18n } from "@/components/lab/i18n"
import { AuthBar } from "@/components/lab/AuthBar"
import { isLoggedIn } from "@/hooks/useAuth"
import { apiLogin, apiSignUp } from "@/services/api"
import { apiErrorMessage } from "@/services/http"

export const Route = createFileRoute("/lab/signup")({
  beforeLoad: async () => {
    if (isLoggedIn()) {
      throw redirect({ to: "/lab/dashboard" })
    }
  },
  component: SignUpPage,
  head: () => ({ meta: [{ title: "Create account - AI Model Lightweight Platform" }] }),
})

function SignUpPage() {
  const navigate = useNavigate()
  const { t } = useI18n()
  const [fullName, setFullName] = useState("")
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [confirm, setConfirm] = useState("")
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState("")

  const submit = async (event: React.FormEvent) => {
    event.preventDefault()
    if (!fullName.trim() || !email.trim() || !password) {
      setError(t("请填写姓名、邮箱与密码", "Please fill in name, email and password"))
      return
    }
    if (password.length < 8) {
      setError(t("密码至少 8 个字符", "Password must be at least 8 characters"))
      return
    }
    if (password !== confirm) {
      setError(t("两次输入的密码不一致", "Passwords do not match"))
      return
    }
    setBusy(true)
    setError("")
    try {
      await apiSignUp({ email: email.trim(), full_name: fullName.trim(), password })
      await apiLogin(email.trim(), password, true)
      navigate({ to: "/lab/dashboard" })
    } catch (e) {
      setError(apiErrorMessage(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="relative w-full max-w-[420px]">
      <AuthBar />
      <div className="mb-8 flex flex-col items-center gap-3 text-center">
        <span className="flex size-12 items-center justify-center rounded-xl bg-[var(--lb-primary)] text-[var(--lb-primary-foreground)]">
          <FlaskConical size={24} />
        </span>
        <div>
          <p className="text-xl font-bold tracking-tight text-[var(--lb-text)]">
            {t("创建 AI Model Lightweight 账号", "Create an AI Model Lightweight account")}
          </p>
          <p className="lb-muted mt-1 text-sm">
            {t("注册后自动登录 · 平台默认不发送邮件", "Auto sign-in after signup · email disabled by default")}
          </p>
        </div>
      </div>

      <div className="lb-card lb-card-white rounded-[10px] p-7 shadow-[var(--lb-shadow-md)]">
        <form onSubmit={(e) => void submit(e)} className="grid gap-4">
          <div>
            <label className="lb-label" htmlFor="lb-s-full">{t("姓名", "Full name")}</label>
            <input id="lb-s-full" className="lb-input" type="text" autoComplete="name" autoFocus
              placeholder={t("你的名字", "Your name")}
              value={fullName} onChange={(e) => setFullName(e.target.value)} />
          </div>
          <div>
            <label className="lb-label" htmlFor="lb-s-email">{t("邮箱", "Email")}</label>
            <input id="lb-s-email" className="lb-input" type="email" autoComplete="email"
              placeholder="user@example.com"
              value={email} onChange={(e) => setEmail(e.target.value)} />
          </div>
          <div>
            <label className="lb-label" htmlFor="lb-s-password">{t("密码", "Password")}</label>
            <input id="lb-s-password" className="lb-input" type="password" autoComplete="new-password"
              placeholder={t("至少 8 个字符", "At least 8 characters")}
              value={password} onChange={(e) => setPassword(e.target.value)} />
          </div>
          <div>
            <label className="lb-label" htmlFor="lb-s-confirm">{t("确认密码", "Confirm password")}</label>
            <input id="lb-s-confirm" className="lb-input" type="password" autoComplete="new-password"
              placeholder={t("再次输入密码", "Re-enter password")}
              value={confirm} onChange={(e) => setConfirm(e.target.value)} />
          </div>

          {error && (
            <div className="flex items-start gap-2 rounded-md border border-[var(--lb-danger)]/30 bg-[var(--lb-danger-soft)] px-3 py-2.5 text-[13px] text-[var(--lb-danger)]">
              <AlertCircle size={15} className="mt-0.5 shrink-0" />
              <span>{error}</span>
            </div>
          )}

          <button type="submit" disabled={busy} className="lb-btn lb-btn--primary !h-10 w-full">
            {busy ? (
              <>
                <Loader2 size={15} className="animate-spin" /> {t("注册中…", "Creating…")}
              </>
            ) : (
              t("注册并进入工作台", "Create account")
            )}
          </button>
        </form>
      </div>

      <p className="lb-muted mt-5 text-center text-[13px]">
        {t("已有账号？", "Already have an account?")}{" "}
        <Link to="/lab/login" className="font-medium text-[var(--lb-primary)] hover:underline">
          {t("返回登录", "Back to sign in")}
        </Link>
      </p>
    </div>
  )
}
