/**
 * components/lab/Navbar.tsx
 * 浅色顶部导航（当前选中项底部蓝色高亮）。
 */
import { useEffect, useState } from "react"

import { apiLogout, apiMe } from "@/services/api"
import type { UserPublic } from "@/services/types"
import { useI18n } from "./i18n"
import { useLabTheme } from "./theme"
import {
  BarChart3,
  Boxes,
  FlaskConical,
  Languages,
  LayoutDashboard,
  LogOut,
  Moon,
  Rocket,
  Sun,
} from "lucide-react"
import { Link, useNavigate, useRouterState } from "@tanstack/react-router"

export function Navbar() {
  const navigate = useNavigate()
  const { location } = useRouterState()
  const { theme, toggle } = useLabTheme()
  const { locale, t, toggle: toggleLocale } = useI18n()
  const [user, setUser] = useState<UserPublic | null>(null)

  useEffect(() => {
    apiMe()
      .then(setUser)
      .catch(() => setUser(null))
  }, [locale])

  const logout = () => {
    apiLogout()
    navigate({ to: "/lab/login" })
  }

  const NAV_ITEMS = [
    { to: "/lab/dashboard", label: t("工作台", "Dashboard"), icon: LayoutDashboard },
    { to: "/lab/models", label: t("模型管理", "Models"), icon: Boxes },
    { to: "/lab/compressions", label: t("压缩任务", "Compressions"), icon: FlaskConical },
    { to: "/lab/evaluate", label: t("评估对比", "Evaluate"), icon: BarChart3 },
    { to: "/lab/export", label: t("部署导出", "Deploy"), icon: Rocket },
  ]

  return (
    <header className="lb-nav">
      <Link
        to="/lab/dashboard"
        className="flex items-center gap-2 text-[17px] font-bold tracking-tight text-[var(--lb-text)]"
      >
        <span className="flex size-8 items-center justify-center rounded-md bg-[var(--lb-primary)] text-[var(--lb-primary-foreground)]">
          <FlaskConical size={17} />
        </span>
        <span>
          AI Model{" "}
          <span className="text-[var(--lb-primary)]">
            {t("轻量化平台", "Lightweight")}
          </span>
        </span>
      </Link>

      <nav className="flex flex-1 items-center gap-6">
        {NAV_ITEMS.map((item) => {
          const active = location.pathname.startsWith(item.to)
          return (
            <Link
              key={item.to}
              to={item.to}
              className={active ? "lb-nav__link active" : "lb-nav__link"}
            >
              <item.icon size={16} />
              {item.label}
            </Link>
          )
        })}
      </nav>

      <div className="flex items-center gap-3">
        <button
          type="button"
          aria-label={t("切换语言", "Switch language")}
          onClick={toggleLocale}
          className="lb-btn lb-btn--ghost p-2"
          data-testid="lab-locale-toggle"
        >
          <Languages size={16} />
          <span className="text-xs font-semibold">{locale === "zh" ? "EN" : "中"}</span>
        </button>
        <button
          type="button"
          aria-label={theme === "dark" ? t("切换到浅色模式", "Light mode") : t("切换到暗色模式", "Dark mode")}
          onClick={toggle}
          className="lb-btn lb-btn--ghost p-2"
          data-testid="lab-theme-toggle"
        >
          {theme === "dark" ? <Sun size={16} /> : <Moon size={16} />}
        </button>
        <span className="h-5 w-px bg-[var(--lb-border)]" />
        <span className="max-w-40 truncate text-[13px] font-medium text-[var(--lb-text-2)]">
          {user?.full_name || user?.email || "…"}
        </span>
        <button
          type="button"
          aria-label={t("退出登录", "Sign out")}
          onClick={logout}
          className="lb-btn lb-btn--ghost p-2"
        >
          <LogOut size={15} />
        </button>
      </div>
    </header>
  )
}

