# CPU-only 单机部署（Docker Compose 直连版）

适用：一台只有 CPU 的 Linux 服务器（x86_64/amd64），通过 `http://<服务器IP>:8001`
直接访问，无域名、无 HTTPS 反向代理。需要 HTTPS/域名的请看同目录
[`deployment-docker-compose.md`](deployment-docker-compose.md)（Traefik + Let's Encrypt）。

## 部署速查（先看这里）

> 本部署使用文件：
> - `compose.cpu.yml` —— 单机编排（db / redis / backend / worker）
> - `backend/Dockerfile.cpu` —— CPU 镜像（免 BuildKit，老 Docker 也能构建）
> - `.env.cpu.example` —— 环境变量模板
> - `scripts/cpu-deploy.sh` —— 服务器侧：构建/启动/日志/状态
> - `scripts/deploy-to-server.sh` —— **一键部署**：同步代码→检查配置→远程后台构建→启动
>
> 依赖已默认 PyTorch **CPU 索引**：`torch==2.13.0+cpu`，锁文件里没有 nvidia-*/CUDA 轮子。

### 方式一：一键脚本（推荐，需已配置 SSH 免密）

```bash
# 在【本机仓库根目录】执行一次：
./scripts/deploy-to-server.sh root@你的服务器IP
# 首次运行会提示：登录服务器填好 /opt/app/.env.cpu 的三个密钥后，再跑一次即可
```

脚本自动完成：`rsync` 同步代码 → 检查/初始化 `.env.cpu` → 服务器**后台构建**
（`setsid+nohup`，本机断开 SSH 也不中断）→ 轮询到“构建完成” → `up -d` 启动，
最后打印访问地址。

### 方式二：手动分步（见下文各节）

```bash
# 1) 本机：同步代码
rsync -az --exclude=.git --exclude=.venv --exclude=node_modules \
  --exclude=backend/models-storage --exclude=backend/app/frontend \
  --exclude=.env ./ root@服务器IP:/opt/app/

# 2) 服务器：填写 .env.cpu
ssh root@服务器IP
cd /opt/app && cp .env.cpu.example .env.cpu
openssl rand -hex 32     # 执行 3 次，分别填入 SECRET_KEY / POSTGRES_PASSWORD / FIRST_SUPERUSER_PASSWORD
sed -i 's#http://localhost:8001#http://服务器IP:8001#' .env.cpu

# 3) 服务器：后台构建 + 启动
bash scripts/cpu-deploy.sh background    # 日志 cpu-build.log，断 SSH 不中断
bash scripts/cpu-deploy.sh status        # 轮询直到“构建完成 ✔”
bash scripts/cpu-deploy.sh up

# 4) 访问
#    http://服务器IP:8001          平台前端（超级管理员登录）
#    http://服务器IP:8001/docs    Swagger API 文档
```

### 方式三：本机构建好镜像 → 直接传服务器（省服务器构建时间）

```bash
# 本机（已 build 过 backend:latest 时）
docker save backend:latest postgres:18 redis:7-alpine | gzip > cpu-deploy-images.tgz
scp cpu-deploy-images.tgz root@服务器IP:/opt/app/

# 服务器
cd /opt/app && gunzip -c cpu-deploy-images.tgz | docker load
docker compose -f compose.cpu.yml --env-file .env.cpu up -d    # 不加 --build，用本地镜像
```

---

## 一、服务器准备

```bash
# Docker Engine + Compose v2（`docker compose version`；v1 老环境用 `docker-compose --version` 亦可）
docker --version && docker compose version

# 放行端口（云服务器还需在安全组/防火墙里开放）
sudo ufw allow 8001/tcp    # 可选：Mailpit 需要再加 8025/tcp
```

## 二、把代码拷到服务器

```bash
# 任选其一：rsync 或 git
rsync -av --exclude=".git/" --filter=":- .gitignore" ./ root@SERVER:/opt/app/
# 或
git clone <你的仓库地址> /opt/app && cd /opt/app
```

> 镜像构建时前端（bun/vite）与后端（uv/pip）都会在服务器上编译，构建机需要有
> 外网访问 pypi/npm。构建产物不依赖 GPU。

## 三、配置环境变量

```bash
cd /opt/app
cp .env.cpu.example .env.cpu

# 生成三个随机 secret 并填入 .env.cpu 的 SECRET_KEY / POSTGRES_PASSWORD / FIRST_SUPERUSER_PASSWORD
openssl rand -hex 32

# 把 FRONTEND_HOST 改成你的地址：
#   sed -i 's#http://localhost:8001#http://服务器IP:8001#' .env.cpu
```

> 重要：`.env.cpu` 中 `FASTAPI_ENV` 必须留空（生产模式）。留空后，后端启动会
> 校验密钥——若仍是模板默认值 `changethis` 会直接报错拒启，这是安全保护。

## 四、生成 Docker 镜像

```bash
cd /opt/app
docker-compose -f compose.cpu.yml --env-file .env.cpu build
```

`backend` 与 `worker` 共用同一镜像 `backend:latest`（多阶段：前端 vite 构建 →
python:3.14 + uv 安装全部 CPU 依赖：torch / onnxruntime / openvino / optimum-intel 等）。
`compose.cpu.yml` 使用的是 **`backend/Dockerfile.cpu`**——不依赖 BuildKit 的兼容写法
（老版本 Docker 的 legacy builder 也能构建）；新版 Docker/Compose v2 亦完全兼容。
首次构建需下载数百 MB 的 CPU 依赖（**不含 CUDA**），耐心等待即可，建议在服务器本机构建。

> 若使用模板自带的 `backend/Dockerfile`（BuildKit 版）走官方 compose，则要求
> Docker Engine + compose v2（BuildKit 默认开启），见 `compose.yml`。

> 体积提示：本仓库 **已默认使用 PyTorch CPU 索引**（根 `pyproject.toml` 里
> `tool.uv.sources` 把 `torch` 指向 `download.pytorch.org/whl/cpu`，并锁定
> `torch==2.13.0+cpu`）——`uv.lock` 中**不再包含 nvidia-* CUDA 运行库与 triton**，
> 镜像因此小约 2GB，纯 CPU 服务器不会下载任何 CUDA 轮子。之后若改动依赖，请用
> `uv lock` 重新生成（保持该 CPU 索引配置）。

## 四之二、避免卡住 / 拖垮 VS Code：在独立终端构建（推荐）

镜像构建里 `uv` 要并行下载/解压 torch、onnxruntime、openvino 等（合计约 2GB+），
**不要**在 VS Code 集成终端里跑重型构建——容易把 WSL/编辑器内存吃满甚至退出。

### 方案 A：独立系统终端（最省心）
在 **Windows Terminal → WSL**、或 **SSH 登录服务器** 的普通终端里执行：

```bash
cd /opt/app
docker-compose -f compose.cpu.yml --env-file .env.cpu build   # 前台，看到完整进度
```

### 方案 B：后台构建（关 VS Code / 断 SSH 都不中断）
仓库内置脚本 `scripts/cpu-deploy.sh`：

```bash
cd /opt/app
cp .env.cpu.example .env.cpu   # 先填好密钥
bash scripts/cpu-deploy.sh background   # 后台启动，日志写入 ./cpu-build.log
bash scripts/cpu-deploy.sh logs         # 随时看进度（Ctrl+C 只退出查看）
bash scripts/cpu-deploy.sh status       # 完成/失败状态
bash scripts/cpu-deploy.sh up           # 构建完成后再启动服务
```

> 脚本用 `setsid + nohup` 把构建进程脱离 VS Code 会话；即使 VS Code 整个退出，
> 构建仍会继续，日志都落在 `cpu-build.log`。

### 它"卡住"其实是正常现象
镜像构建需要下载/解压 torch(CPU≈183MB)、openvino、onnxruntime 等数十个包，
uv 输出进度时可能短暂停顿，并非死机。`backend/Dockerfile.cpu` 已内置低并行设置
（`UV_CONCURRENT_DOWNLOADS=8`、`UV_CONCURRENT_BUILDS=1`、`UV_CONCURRENT_INSTALLS=1`）
控制内存；等待即可，或用 `bash scripts/cpu-deploy.sh status` 看是否仍在跑。

## 五、启动

```bash
docker-compose -f compose.cpu.yml --env-file .env.cpu up -d
```

启动顺序由 compose 保证：

1. `db`（Postgres）健康后 → `redis` 健康后
2. `backend` 容器先执行 `./scripts/prestart.sh`：**自动 alembic 迁移 +
   创建首个超级管理员**（幂等），随后启动 FastAPI（默认 2 个 worker 进程）
3. `worker` 在 backend 健康后才启动，运行 `arq app.worker.WorkerSettings`，
   执行压缩/拉取任务（默认同一时间只跑 1 个，避免 CPU 被打满）

## 六、访问地址

| 入口 | 地址 |
| --- | --- |
| 平台前端（登录/工作台/压缩/报告…） | http://SERVER_IP:8001 |
| Swagger API 文档 | http://SERVER_IP:8001/docs |
| 超级管理员 | `.env.cpu` 里 FIRST_SUPERUSER 邮箱 + FIRST_SUPERUSER_PASSWORD |
| Mailpit 收件箱（启用后） | http://SERVER_IP:8025 |

> 站点所有 SPA 页面（如 /report）都支持浏览器直达与刷新。

## 七、日常运维

```bash
# 日志
docker-compose -f compose.cpu.yml --env-file .env.cpu logs -f backend worker
# 进入后端容器手动操作（如手动跑迁移）
docker-compose -f compose.cpu.yml exec backend sh
# 重启
docker-compose -f compose.cpu.yml --env-file .env.cpu restart
# 停止/删除（保留数据卷）
docker-compose -f compose.cpu.yml --env-file .env.cpu down
```

**数据持久化**（命名卷，删容器不丢）：

- `app-db-data` → Postgres 数据
- `redis-data` → Redis AOF
- `models-data` → 模型/数据集/任务产物 + llama.cpp 工具缓存（`/app/backend/models-storage`）

```bash
# 备份数据库
docker-compose -f compose.cpu.yml exec -T db pg_dump -U postgres app > backup.sql
# 备份模型文件
docker run --rm -v full-stack-fastapi_models-data:/data -v "$PWD":/backup alpine tar czf /backup/models.tgz -C /data .
```

**更新到新版本**：重新 `rsync`/`git pull` 代码 → `docker-compose ... build` →
`docker-compose ... up -d`（backend 启动时自动执行新迁移）。

## 八、CPU 服务器调优建议

| 项 | 建议 |
| --- | --- |
| `WEB_CONCURRENCY` | 2（默认）；1 核小机器可设 1 |
| `WORKER_MAX_JOBS` | 1（默认）；4 核以上可试 2（LLM/GGUF 任务本身吃满多核） |
| `USER_STORAGE_QUOTA_BYTES` | 按磁盘规划，默认 50GB/用户 |
| 磁盘 | 镜像+数据建议 ≥ 30GB；压缩 LLM（GGUF/OpenVINO）需额外空间 |
| 内存 | 建议 ≥ 8GB；QAT/LLM 转换峰值明显 |

## 九、网络说明（GGUF / 模型下载）

- 镜像内不装 `git`：llama.cpp 转换工具获取会自动跳过 git 源，改走
  codeload tar / jsDelivr CDN（仅需 HTTPS 出网）；也可在 models-data 卷内
  预置 llama.cpp 并用环境变量 `LLAMA_CPP_PATH` 指向它。
- 模型下载按 HuggingFace → hf-mirror.com → 魔搭 ModelScope 自动兜底；
  国内服务器可直接把模型源填成魔搭页面 URL。

## 十、可选：HTTPS + 域名

本方案直连 IP 明文 HTTP。对外生产建议域名 + Traefik（模板自带）：
`docker-compose -f compose.yml -f compose.deploy.yml up -d --build`，
见 [`deployment-docker-compose.md`](deployment-docker-compose.md)。
