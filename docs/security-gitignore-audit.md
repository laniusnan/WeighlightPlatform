# .gitignore 审查清单（2026-09-09）

目的：确认仓库不会把真实密钥/口令/本地环境提交到 gitee 远程。

## 1. 规则现状（`.gitignore`）
```
.venv/  .venv311*  __pycache__/  *.pyc
.env  .env.cpu  frontend/.env  .env.gpu  .env.*  （!.env.*.example 放行模板）
*.local  models-storage/  backend/models-storage/
*.log  cpu-build.log  ui-preview/  dist/  *.tgz  .DS_Store
node_modules/  backend/app/frontend/  test 结果与 playwright 目录
```

## 2. 已跟踪文件的敏感扫描结果（`git ls-files` 过滤 env/secret/key/password）
| 命中文件 | 是否敏感 |
|---|---|
| `.env.cpu.example` | 否（占位模板）|
| `.env.gpu.example` | 否（占位模板）|
| `email-templates/reset_password.html` | 否（邮件模板源码）|
| `frontend/.../ChangePassword.tsx` 等 | 否（前端功能源码，非凭据）|

结论：**无真实密钥/口令入库**。

## 3. 服务器侧确认
- CPU 服务器 `/opt/app` 为 rsync 部署，非 git 仓库；其中的 `.env.cpu` 在 `.gitignore` 规则内
- 服务器代码状态将以独立分支 `cpu-server` 推送到同一远程（见仓库分支列表）

## 4. 例行检查命令
```bash
git ls-files | grep -iE '(^|/)\\.env|secret|\\.pem$|\\.key$|password'   # 应只剩 example/模板
git ls-files | grep -E '\\.env($|\\.)'                                   # 仅 .env.*.example
```

## 5. 建议（可选）
- 首次推送后到 gitee 页面复核一次文件树
- 真实服务器 `.env.cpu` 若曾用弱口令，建议在 gitee 后台关掉“示例明文展示”相关疑虑前轮换一次口令
