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
  - `prune`：L2 范数结构化卷积滤波器剪枝
  - `llm_quantize`：LLM（Qwen 等）INT4/INT8 量化导出为 OpenVINO IR（optimum-intel）
- **精度评估**：图像分类 Top-1、YOLO mAP、LLM Perplexity（验证集 zip 自动下载）
- **架构**：FastAPI + Redis + Arq 任务队列，压缩任务在独立 `worker` 容器中串行执行，
  模型文件存放在共享磁盘卷，元数据存 PostgreSQL
- **部署**：`docker compose -f compose.yml -f compose.deploy.yml up -d`（含 Traefik HTTPS）

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

## GPU 支持（规划阶段）

平台当前全部计算在 **CPU** 上执行。代码已为 NVIDIA GPU 迁移做了架构准备：

- 新增计算后端配置：`COMPUTE_BACKEND=cpu|gpu`、`GPU_DEVICE_IDS`、`CUDA_VISIBLE_DEVICES`；
- 计算服务抽象层位于 `backend/app/services/`：CPU 实现直接复用现有管线，
  GPU 实现为骨架（`*_gpu.py`），未来按方法逐个使能即可；
- Arq worker 启动时打印当前计算后端（`docker logs <worker>` 可查）。

**当前请保持 `COMPUTE_BACKEND=cpu`（默认）。** 详细评估与使能步骤见
[`docs/gpu-migration-plan.md`](docs/gpu-migration-plan.md) 与
[`docs/architecture.md`](docs/architecture.md)；调试可运行
`python backend/scripts/check_backend.py`。
