/**
 * components/lab/theme.tsx
 * /lab 浅色/暗色主题上下文（新版界面统一使用；localStorage 持久化）。
 */
import {
  createContext,
  useContext,
  useEffect,
  useState,
  type ReactNode,
} from "react"

export type LabTheme = "light" | "dark"

const STORAGE_KEY = "lab-ui-theme"

interface LabThemeState {
  theme: LabTheme
  toggle: () => void
  setTheme: (t: LabTheme) => void
}

const LabThemeContext = createContext<LabThemeState | null>(null)

export function LabThemeProvider({ children }: { children: ReactNode }) {
  const [theme, setTheme] = useState<LabTheme>(() => {
    const stored = localStorage.getItem(STORAGE_KEY)
    return stored === "dark" ? "dark" : "light"
  })

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, theme)
  }, [theme])

  const toggle = () => setTheme((t) => (t === "dark" ? "light" : "dark"))

  return (
    <LabThemeContext.Provider value={{ theme, toggle, setTheme }}>
      {children}
    </LabThemeContext.Provider>
  )
}

export function useLabTheme(): LabThemeState {
  const ctx = useContext(LabThemeContext)
  if (!ctx) throw new Error("useLabTheme must be used inside LabThemeProvider")
  return ctx
}
