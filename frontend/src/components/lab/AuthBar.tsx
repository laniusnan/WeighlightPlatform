/**
 * components/lab/AuthBar.tsx · 登录/注册页顶部的 暗/亮 + 中/英 切换
 */
import { Languages, Moon, Sun } from "lucide-react"

import { useI18n } from "./i18n"
import { useLabTheme } from "./theme"

export function AuthBar() {
  const { theme, toggle } = useLabTheme()
  const { locale, t, toggle: toggleLocale } = useI18n()

  return (
    <div className="absolute right-5 top-5 flex items-center gap-1.5">
      <button
        type="button"
        aria-label={t("切换语言", "Switch language")}
        onClick={toggleLocale}
        className="lb-btn lb-btn--ghost p-2"
        data-testid="auth-locale-toggle"
      >
        <Languages size={16} />
        <span className="text-xs font-semibold">{locale === "zh" ? "EN" : "中"}</span>
      </button>
      <button
        type="button"
        aria-label={theme === "dark" ? t("切换到浅色模式", "Light mode") : t("切换到暗色模式", "Dark mode")}
        onClick={toggle}
        className="lb-btn lb-btn--ghost p-2"
        data-testid="auth-theme-toggle"
      >
        {theme === "dark" ? <Sun size={16} /> : <Moon size={16} />}
      </button>
    </div>
  )
}
