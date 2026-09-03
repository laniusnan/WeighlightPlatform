import { expect, test } from "@playwright/test"
import { firstSuperuser, firstSuperuserPassword } from "./config.ts"

test.use({ storageState: { cookies: [], origins: [] } })

test("登录后可从侧边栏点进「平台使用报告」阅读", async ({ page }) => {
  await page.goto("/login")
  await page.getByTestId("email-input").fill(firstSuperuser)
  await page.getByTestId("password-input").fill(firstSuperuserPassword)
  await page.getByRole("button", { name: /登录/ }).click()
  // 等登录后布局稳定（Dashboard 懒加载期间点击可能被重渲染取消）
  await page.waitForTimeout(1500)

  // 侧边栏入口（页脚也有同名链接，取侧边栏第一个）
  const navLink = page.getByRole("link", { name: "平台使用报告" }).first()
  await expect(navLink).toBeVisible({ timeout: 15000 })
  await navLink.click()

  await expect(page).toHaveURL(/\/report$/, { timeout: 10000 })

  // 报告内容可读
  await expect(
    page.getByRole("heading", { name: /平台使用报告/ }),
  ).toBeVisible()
  await expect(
    page.getByText("五分钟快速上手", { exact: false }),
  ).toBeVisible()
  await expect(page.getByText("压缩与量化方法总览")).toBeVisible()
  await expect(page.getByText("GGUF 导出专项说明")).toBeVisible()

  // 实时状态（来自后端接口）
  await expect(page.getByText("Redis", { exact: true })).toBeVisible()
  await expect(page.getByText("Worker 数", { exact: true })).toBeVisible()

  // 页脚也有入口
  await expect(page.getByText("平台使用报告").last()).toBeVisible()
})

