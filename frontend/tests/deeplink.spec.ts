import { expect, test } from "@playwright/test"
import { firstSuperuser, firstSuperuserPassword } from "./config.ts"

test.use({ storageState: { cookies: [], origins: [] } })

test("深度链接直达 /report + 刷新后仍可阅读报告", async ({ page }) => {
  // 未登录直达 /report → 应跳登录页（而不是 404）
  await page.goto("/report")
  await expect(page).toHaveURL(/\/login/, { timeout: 10000 })
  await expect(page.getByTestId("email-input")).toBeVisible()

  await page.getByTestId("email-input").fill(firstSuperuser)
  await page.getByTestId("password-input").fill(firstSuperuserPassword)
  await page.getByRole("button", { name: /登录/ }).click()
  await expect(page).toHaveURL(/\/$/, { timeout: 10000 })

  // 登录后直接访问深链 /report
  await page.goto("/report")
  await expect(
    page.getByRole("heading", { name: /平台使用报告/ }),
  ).toBeVisible({ timeout: 15000 })
  await expect(page.getByText("压缩与量化方法总览")).toBeVisible()

  // 刷新后仍停留在 /report 且内容可读
  await page.reload()
  await expect(page).toHaveURL(/\/report$/)
  await expect(
    page.getByRole("heading", { name: /平台使用报告/ }),
  ).toBeVisible({ timeout: 15000 })

  // 其它 SPA 页面深链也生效（/models）
  await page.goto("/models")
  await expect(
    page.getByRole("heading", { name: /模型管理/ }),
  ).toBeVisible({ timeout: 15000 })
})
