# Full Stack FastAPI Template

[![Test Docker Compose](../../actions/workflows/test-docker-compose.yml/badge.svg)](../../actions/workflows/test-docker-compose.yml)
[![Test Backend](../../actions/workflows/test-backend.yml/badge.svg)](../../actions/workflows/test-backend.yml)

## 模型轻量化平台 (Model Compression Platform)

基于本模板构建的模型压缩平台，专为 **CPU-only 服务器** 设计：

- **模型管理**：文件上传、URL / HuggingFace repo 拉取（后台下载）
- **压缩方法**（全部 CPU 可跑）：
  - `export_onnx`：PyTorch → ONNX + onnxsim 简化（支持 YOLO）
  - `quantize_onnx`：ONNX 动态 INT8 量化（onnxruntime）
  - `openvino_int8`：OpenVINO IR + NNCF INT8 后训练量化（CPU 推荐）
  - `prune`：结构化滤波器剪枝，6 种准则 —— L1/L2 范数、BN-Slim(γ)、AutoSlim(稀疏训练+γ)、
  Group Fisher(数据驱动 Fisher 信息)、DMCP(Taylor 一阶)（后三需数据集）；剪枝后自动对齐分类头
  （`head_layers_adjusted`）并做前向自检（`forward_check` / `topology_warning`）
- `qat`：量化感知训练 —— FX QAT 或 LSQ 可学习步长，落成全 INT8 QDQ ONNX（权重+激活校准）
- `distill`：知识蒸馏（Hinton logits KD / FitNets 特征对齐）
- `llm_finetune`：LoRA SFT —— CPU 内置微型字符 LM 原型；HuggingFace 模型走真实
  transformers 路径（需 CUDA，见下方 GPU 说明）
- 数据集：支持 Excel/CSV 指令集自动解析（.xlsx/.csv/.tsv 前两列 → 问题=>回答）供微调
- **本地开放数据集（服务器 / GPU 本地）**：`scripts/fetch-open-datasets.sh` 一键拉取
  CIFAR-10/100、MNIST、Fashion-MNIST（默认走 HF 镜像 parquet，由镜像内 polars 读取，无需新增依赖；
  `--source archive` 可回退原始 tar/idx 归档）并转成「按类图片目录」；平台侧
  `GET /datasets/local-dirs` + `POST /datasets/import-local`（reference 模式不复制、不占配额，
  删除仅注销记录、宿主数据不动；copy 模式快照进平台存储并计配额）→ 直接供数据驱动剪枝
  （group_fisher/DMCP/AutoSlim）、QAT、蒸馏与分类评测消费。实测：CIFAR-10（10 类，5 万训练 +
  1 万测试）→ prune(group_fisher) + ONNX 导出 + 1 万张分类评测全链路通过
- **本地文本数据集（同一条 `/datasets-local` 通道）**：同一脚本支持 WikiText-2 raw（真实语料
  perplexity）、Alpaca / GSM8K（LoRA SFT 指令集）、HH-RLHF（prompt/chosen/rejected 偏好对，DPO）；
  消费方式：`eval={"type":"perplexity","dataset_id":wikitext2-raw-test}`、
  `llm_finetune{dataset_id: alpaca-train}`、`llm_finetune{stage:"dpo", dataset_id: hh-rlhf-train}`。
  CPU 实测：PPL 15.835（wikitext2，512 token 窗口）· LoRA 256 样本 · 偏好对 256/dpo_ready=true；
  GPU 实测：LoRA loss 1.2769→0.8792 · DPO 初始 loss 0.693147(=ln2)→0.6446
- `llm_quantize`：LLM（Qwen 等）INT4/INT8 量化导出为 OpenVINO IR（optimum-intel）
- **精度评估**：图像分类 Top-1、YOLO mAP、LLM Perplexity（验证集 zip 自动下载）
- **架构**：FastAPI + Redis + Arq 任务队列，压缩任务在独立 `worker` 容器中串行执行，
  模型文件存放在共享磁盘卷，元数据存 PostgreSQL
- **任务中心**：状态筛选 + 关键字搜索；「产物说明」弹窗展示文件清单/生成参数/评估与基准摘要，一键打包下载
- **GPU 本地链路（已验证）**：Qwen2.5-0.5B —— LoRA SFT（adapter/merged）→ INT4 OpenVINO IR（954MB→339MB，2.95×）→ GGUF 导出；DPO 偏好对以 preference_pairs.jsonl 预留
- **GGUF 量化矩阵（Qwen2.5-0.5B 实测）**：f16 994MB（转换器直出）· q8_0 531MB/1.88×（直出）· q4_0 283.8MB/3.52×（Python 回退）· q4_K_M 693MB（Python Q4_K 回退，最大误差 0.046）——缺失 `llama-quantize` 时自动走内置纯 numpy/gguf 量化，无需编译；`LLAMA_CPP_PATH` 可预置转换器目录（CPU compose 已默认挂载 /opt/app/llama.cpp）
- **自检**：实验页含 9+ 张冒烟卡（FitNets/蒸馏/6 种剪枝/QAT-LSQ/LoRA/指令集），可一键验证全链路
- **部署（CPU-only 服务器）**：`compose.cpu.yml` + `scripts/cpu-deploy.sh`（8001 端口）；
  网络受限的服务器推荐 `scripts/cpu-image-ship.sh`：**本地构建镜像 → 压缩传输（zstd/gzip）→
  服务器 `docker load` + `compose up -d --no-build`**（保留 `backend:latest-prev` 回滚标签），
  彻底避开服务器上 `uv sync` 拉取 2GB 依赖时的网络抖动

## Technology Stack and Features

- ⚡ [**FastAPI**](https://fastapi.tiangolo.com) for the Python backend API.
  - 🧰 [SQLModel](https://sqlmodel.tiangolo.com) for the Python SQL database interactions (ORM).
  - 🔍 [Pydantic](https://docs.pydantic.dev), used by FastAPI, for the data validation and settings management.
  - 💾 [PostgreSQL](https://www.postgresql.org) as the SQL database.
- 🚀 [React](https://react.dev) for the frontend.
  - 🧩 Built into the backend application and served by FastAPI on the same domain as the API.
  - 💃 Using TypeScript, hooks, [Vite](https://vitejs.dev), and other parts of a modern frontend stack.
  - 🎨 [Tailwind CSS](https://tailwindcss.com) and [shadcn/ui](https://ui.shadcn.com) for the frontend components.
  - 🤖 An automatically generated frontend client.
  - 🧪 [Playwright](https://playwright.dev) for end-to-end testing.
  - 🦇 Dark mode support.
- ☁️ [FastAPI Cloud](https://fastapicloud.com) for deployment.
- 🐋 [Docker Compose](https://www.docker.com) for local services and self-hosted deployment.
  - 📞 [Traefik](https://traefik.io) as a reverse proxy with automatic HTTPS.
- 🔒 Secure password hashing by default.
- 🔑 JWT (JSON Web Token) authentication.
- 📫 Email-based password recovery.
- ✉️ [React Email](https://react.email) for email templates.
- 📬 [Mailpit](https://mailpit.axllent.org) for local email testing during development.
- ✅ Tests with [Pytest](https://pytest.org).
- 🏭 CI (continuous integration) and CD (continuous deployment) based on GitHub Actions.

### Dashboard Login

![Dashboard login screenshot](img/login.png)

### Dashboard - Admin

![Admin dashboard screenshot](img/dashboard.png)

### Dashboard - Items

![Items dashboard screenshot](img/dashboard-items.png)

### Dashboard - Dark Mode

![Dark mode dashboard screenshot](img/dashboard-dark.png)

### React Email Templates

![Email templates screenshot](img/react-email.png)

### Mailpit - Local Email Testing

![Mailpit screenshot](img/mailpit.png)

### Interactive API Documentation

![API docs](img/docs.png)

## How to Use It

Click the **Use this template** button at the top of this page to create a new repository.

## Backend Development

Backend docs: [backend/README.md](./backend/README.md).

## Frontend Development

Frontend docs: [frontend/README.md](./frontend/README.md).

## Deployment

FastAPI Cloud deployment: [deployment.md](./deployment.md).

Self-hosted deployment with Docker Compose: [deployment-docker-compose.md](./deployment-docker-compose.md).

## Development

General development docs: [development.md](./development.md).

This includes the local FastAPI and Vite workflow, Docker Compose services, `.env` configuration, and more.

## Release Notes

Check the file [release-notes.md](./release-notes.md).

## License

The Full Stack FastAPI Template is licensed under the terms of the MIT license.

## 可移植特性与算法路线图

结合 workspace 内 `mmrazor`、`lmdeploy`、`xtuner` 的甄别结论（完整分析见
[`docs/portable-features.md`](docs/portable-features.md)）：

- **推荐移植（CPU 可行）**：mmrazor 的 KD 蒸馏、L1/L2 结构化剪枝与 BN-Slim、迭代幅度稀疏；
  lmdeploy 的 weight-only INT4 量化思想；已按 P0/P1/P2 排入本仓库后端路线。
          **已落地：P0-1 —— `prune` 现支持 `algorithm=l1|l2|bnslim`（结构化剪枝）。**
          **P0-2 —— `sparsify` 现支持 `strategy=uniform|global`（全局幅度稀疏）。**
          **P1 —— 新增 `distill` 方法（arch.py 学生 + width_mult + 类别数据集 的知识蒸馏）。**
          **P1 —— `llm_quantize` 支持 weight-only RTN group-wise（`group_size`/`sym`，INT4/INT8）。**
- **概念对位已存在**：OpenVINO 量化导出、ONNX 静态/动态 INT8、OpenAI 兼容 Chat/SSE 推理、
  压缩前后 CPU 基准（p50/p99/吞吐/内存）。
- **不移植（GPU/C++ 绑定）**：xtuner LoRA/DPO 微调、mmrazor NAS、lmdeploy Turbomind/TRT/KV 优化——
  作为后续 GPU Worker 扩展的候选。

> 实现某一项（例如 `distill` 蒸馏、`prune` 的 L1/BN-Slim）时，会同步更新：
> `backend/app/models.py`（CompressionMethod）、压缩任务后端实现、前端方法卡片与向导、以及本文档。

---

## GPU 支持（本地开发，与 CPU-only 服务器分离）

本地 NVIDIA（RTX 4060 8GB）通过两个途径使用 GPU：

- **原生 venv**：`.venv311gpu`（torch CUDA + transformers），直接跑真实 HF LoRA SFT：
  `scripts/local-gpu-qwen-validation.py`（Qwen2.5-0.5B，loss 8.7→6.15，13s）。
  大模型权重缓存于 `~/.cache/huggingface`（HF_ENDPOINT=https://hf-mirror.com）。
- **容器栈**：`compose.gpu.yml`（name: mlplatform-gpu-local，8002 端口，worker 分配 1×GPU），
  依赖 docker compose v2 + nvidia-container-toolkit（一键准备：`sudo bash scripts/setup-local-gpu-docker.sh`）。

HuggingFace 模型的 `llm_finetune` 在非 CUDA 环境会显式报错，避免 CPU-only 服务器误训大模型。
服务抽象骨架见 `docs/gpu-worker-architecture.md`、`docs/gpu-migration-plan.md`。

## 代码分支与远程

- `master`：本地主线（CPU 全量方法 + GPU 本地代码）
- `cpu-server`：CPU-only 服务器（10.110.1.36）实际部署态快照
- 远程：`git@gitee.com:laniusnan/aimodel-weighlight-platform.git`（SSH）
- 安全：`.env.*` 全部忽略（仅保留 `*.example` 模板），清单见 `docs/security-gitignore-audit.md`
