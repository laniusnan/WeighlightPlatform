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

## 6. 补记（2026-10-16）：漏判的本地 SSH 密钥已清理

**发现**：`lanius`（OpenSSH ed25519 私钥，带口令加密）与其公钥 `lanius.pub` 一直处于已跟踪状态，自首次快照提交（2026-09-03）起存在于全部历史中；gitee 上的 `cpu-server` 分支（独立历史）同样携带这两个文件。第 2 节的过滤条件按**文件名关键字**匹配（`\.env|secret|\.pem$|\.key$|password`），`lanius` 既无扩展名也无关键字，导致漏判——**原结论“无真实密钥/口令入库”作废**。

**性质（取证结论）**：该密钥对在项目内无任何引用（无脚本/CI/Dockerfile/文档以路径引用它），未注册到 GitHub（账号下 3 把公钥指纹均不含 `SHA256:WRaGW34naTL3jqODJq0z5fBcRwlNM30FKWRi5j1Vy1o`），部署免密依赖系统默认密钥 `~/.ssh/id_ed25519`。判定为 2026-09-03 搭建 SSH 时误留在项目根目录、被 `git add -A` 快照扫入的“孤儿”密钥。

**处理**：
- 从索引移除，并加入 `.gitignore`、`.dockerignore` 与 `scripts/deploy-to-server.sh` 的 rsync 排除列表
- 用 `git filter-branch --index-filter 'git rm --cached --ignore-unmatch lanius lanius.pub' --prune-empty -- --branches` 重写 `master` 与 `cpu-server` 全历史，并强推 gitee / GitHub

**改进后的例行检查（按内容而非文件名，应无输出）**：
```bash
git grep -lE 'BEGIN [A-Z ]*PRIVATE KEY' $(git rev-list --all)
git rev-list --all --objects | grep -iE 'lanius(\.pub)?$'
git ls-files | grep -iE '(^|/)\.env|secret|\.pem$|\.key$|password|id_(rsa|ed25519)'
```
> 注：第一行刻意写成 `BEGIN [A-Z ]*PRIVATE KEY` 正则形式，避免本文件转述该字符串时产生自匹配。
