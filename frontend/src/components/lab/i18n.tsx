/**
 * components/lab/i18n.tsx
 * 中 / 英 双语：全界面文案完全分离，Navbar 顶部切换并持久化。
 * t(zh, en)：组件内随 locale 变化自动重渲染。
 */
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
  type ReactNode,
} from "react"

export type Locale = "zh" | "en"

const STORAGE_KEY = "lab-locale"

/** 模块级同步：供非组件内常量/函数读取，切换语言时同步更新 */
let currentLocale: Locale = "zh"
try {
  const stored = localStorage.getItem(STORAGE_KEY)
  currentLocale = stored === "en" ? "en" : "zh"
} catch {
  /* ignore */
}
export const getCurrentLocale = (): Locale => currentLocale

interface I18nState {
  locale: Locale
  t: (zh: string, en: string) => string
  setLocale: (locale: Locale) => void
  toggle: () => void
}

const I18nContext = createContext<I18nState | null>(null)

export function LabI18nProvider({ children }: { children: ReactNode }) {
  const [locale, setLocaleState] = useState<Locale>(() => {
    const stored = localStorage.getItem(STORAGE_KEY)
    return stored === "en" ? "en" : "zh"
  })

  const setLocale = useCallback((next: Locale) => {
    currentLocale = next
    setLocaleState(next)
  }, [])

  useEffect(() => {
    currentLocale = locale
    localStorage.setItem(STORAGE_KEY, locale)
    document.documentElement.lang = locale === "en" ? "en" : "zh-CN"
  }, [locale])

  const t = useCallback(
    (zh: string, en: string) => (locale === "en" ? en : zh),
    [locale],
  )

  const toggle = useCallback(
    () => setLocale(currentLocale === "zh" ? "en" : "zh"),
    [setLocale],
  )

  return (
    <I18nContext.Provider value={{ locale, t, setLocale, toggle }}>
      {children}
    </I18nContext.Provider>
  )
}

export function useI18n(): I18nState {
  const ctx = useContext(I18nContext)
  if (!ctx) throw new Error("useI18n must be used inside LabI18nProvider")
  return ctx
}

/** 非 React（纯函数/常量）场景使用的语言判断：组件须随 locale 重渲染后重新调用 */
export function localeOf(locale: Locale): "zh" | "en" {
  return locale
}
