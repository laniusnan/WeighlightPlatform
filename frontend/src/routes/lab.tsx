/**
 * routes/lab.tsx
 * /lab 新版界面的父布局：认证守卫 + 暗/亮主题隔离 + 顶部导航。
 * /lab/login 与 /lab/signup 无需登录，也不显示顶栏。
 */
import { createFileRoute, Outlet, redirect, useRouterState } from "@tanstack/react-router"

import { Navbar } from "@/components/lab/Navbar"
import { LabI18nProvider } from "@/components/lab/i18n"
import { LabThemeProvider, useLabTheme } from "@/components/lab/theme"
import { isSessionValid } from "@/services/http"

import "../styles/light-theme.css"
import "../styles/dark-theme.css"

const PUBLIC_LAB_PAGES = ["/lab/login", "/lab/signup"]

export const Route = createFileRoute("/lab")({
  beforeLoad: async ({ location }) => {
    const isPublic = PUBLIC_LAB_PAGES.includes(location.pathname)
    const loggedIn = await isSessionValid()
    if (!loggedIn && !isPublic) {
      throw redirect({ to: "/lab/login" })
    }
    if (loggedIn && isPublic) {
      throw redirect({ to: "/lab/dashboard" })
    }
  },
  component: LabLayout,
  head: () => ({
    meta: [{ title: "AI Model Lightweight Platform" }],
  }),
})

function LabLayout() {
  return (
    <LabI18nProvider>
      <LabThemeProvider>
        <ThemedLabLayout />
      </LabThemeProvider>
    </LabI18nProvider>
  )
}

function ThemedLabLayout() {
  const { location } = useRouterState()
  const { theme } = useLabTheme()
  const showNavbar = !PUBLIC_LAB_PAGES.includes(location.pathname)

  return (
    <div className={theme === "dark" ? "theme-dark" : "theme-light"}>
      {showNavbar && <Navbar />}
      <main
        className={
          showNavbar
            ? "mx-auto w-full max-w-[1320px] px-6 py-7 md:px-8"
            : "flex min-h-screen items-center justify-center px-6 py-10"
        }
      >
        <Outlet />
      </main>
    </div>
  )
}


