# 模型压缩能力 · 环境要求（compression-env）

> 适用对象：把「模型压缩」作为**推理模块的子功能**集成时，需要准备的环境。
> 本文所有版本号均来自本仓库 `uv.lock` / `backend/Dockerfile.cpu` 的**实际锁定版本**，可直接复制使用。

---

## 1. 操作系统

| 项目 | 要求 | 说明 |
|---|---|---|
| 架构 | **Linux x86_64**（amd64） | OpenVINO / onnxruntime 预编译轮子只提供 x86_64；ARM64 需自行编译 |
| 发行版 | Ubuntu 20.04+ / Debian 11+（容器内为 `python:3.11` = Debian bookworm） | 需要 `libgl1`、`libglib2.0-0`（torchvision/ultralytics 图像算子运行时依赖） |
| 内核 | ≥ 4.15 | Docker 20.10+ 前提 |
| Shell | bash ≥ 4（脚本使用数组/关联数组） | 例如 `scripts/fetch-open-datasets.sh` 使用 `declare -A` |

开发/演示也可在 WSL2（同为 x86_64）运行；**Windows 原生不支持**（依赖 Linux 容器与 Unix 权限）。

---

## 2. Python 版本

| 项目 | 版本 | 说明 |
|---|---|---|
| Python | **3.11**（`>=3.11,<4.0`） | 与 `backend/pyproject.toml` / `backend/Dockerfile.cpu`（`FROM python:3.11`）一致 |
| 包管理 | uv **0.9.26**（镜像内 `pip install uv==0.9.26`） | 生产构建使用 `uv sync --frozen`；本地也可用 venv + pip |

---

## 3. 核心依赖及锁定版本

以下为 `uv.lock` 实测锁定版本（2026-09 快照，CPU 版 torch）：

| 依赖 | 锁定版本 | 作用（压缩相关） |
|---|---|---|
| `torch` | **2.9.1+cpu** | 剪枝 / 蒸馏 / QAT / LoRA-SFT / DPO 的模型与训练 |
| `torchvision` | **0.24.1** | 图像预处理、YOLO（ultralytics）依赖 |
| `transformers` | **4.49.0** | HF 模型/分词器（LLM 量化、GGUF 转换、微调） |
| `optimum` | **1.27.0** | ONNX/OpenVINO 导出后端 |
| `optimum-intel` | **1.27.0** | **LLM INT4/INT8 OpenVINO IR**（`llm_quantize`）、perplexity 评测 |
| `nncf` | **3.3.0** | **OpenVINO IR INT8 后训练量化**（`openvino_int8`）、LLM INT4 权重压缩 |
| `openvino` | **2026.3.1** | IR 推理/保存（GPU/CPU 自动回落） |
| `onnx` | **1.22.0** | ONNX 图导出/读写 |
| `onnxruntime` | **1.29.0** | 动态/静态 INT8 量化执行器、ONNX 推理 |
| `onnxsim` | **0.7.3** | 导出后图简化 |
| `gguf` | **0.19.0** | GGUF 读写（f16/q8_0 直出，q4_0/q4_K/Q6_K 纯 Python 回退） |
| `sentencepiece` | **≥0.2.0** | LLM 分词 |
| `protobuf` | **≥4.25,<7** | ONNX / OpenVINO 序列化 |
| `ultralytics` | **≥8.3.0** | YOLO 模型支持（可选） |
| `polars` | **1.44.1** | 数据集取数（parquet → 按类图片 / 文本语料），**无需 pyarrow** |
| `pillow` | **11.3.0** | 图像读写 |
| `numpy` | **2.4.6** | 数值计算 |
| `openpyxl` | **≥3.1** | Excel 指令集解析（`.xlsx` → 指令对） |
| `fastapi[standard]` | **≥0.141** | 平台 API（压缩任务提交/查询/产物下载） |
| `sqlmodel` | **≥0.0.39** | ORM（模型/数据集/任务元数据） |
| `arq` | **0.28.0** | Redis 异步任务队列（worker 执行压缩） |
| `redis` | **≥5.0,<7** | 队列后端 |
| `psycopg[binary]` | **≥3.3.4** | PostgreSQL 驱动 |
| `alembic` | **≥1.12.1** | 数据库迁移 |
| `httpx` | **≥0.25.1** | CLI/后台 HTTP 客户端 |
| `pydantic` / `pydantic-settings` | **>2.0** / **≥2.2.1** | 配置与校验（含 `COMPRESSION_*` 等环境变量覆盖） |

> GPU 版（可选）仅替换 `torch/torchvision` 为 CUDA 轮子（如 `torch 2.9.1+cu126`），其余依赖相同；`backend/Dockerfile.gpu` 为 GPU 镜像。

### `pyproject.toml` 片段（直接可用）

```toml
[project]
name = "app"
version = "0.1.0"
requires-python = ">=3.11,<4.0"
dependencies = [
    "fastapi[standard]>=0.141.1,<1.0.0",
    "pydantic>2.0",
    "pydantic-settings<3.0.0,>=2.2.1",
    "sqlmodel>=0.0.39,<1.0.0",
    "alembic<2.0.0,>=1.12.1",
    "psycopg[binary]>=3.3.4,<4.0.0",
    "httpx<1.0.0,>=0.25.1",
    "arq>=0.28,<1.0",
    "redis>=5.0,<7.0",
    "numpy>=1.26,<3.0",
    "openpyxl>=3.1,<4.0",
    "tokenizers>=0.21,<0.22",
    "pillow>=10.0,<12.0",
    # --- compression toolchain ---
    "torch>=2.6,<3.0",
    "torchvision>=0.24,<0.26",
    "onnx>=1.17,<2.0",
    "onnxruntime>=1.20,<2.0",
    "onnxsim>=0.4.36,<1.0",
    "openvino>=2025.0,<2027.0",
    "nncf>=3.0,<4.0",
    "transformers>=4.49,<4.50",
    "optimum>=1.21,<3.0",
    "optimum-intel>=1.21,<3.0",
    "accelerate>=1.0,<2.0",
    "sentencepiece>=0.2.0,<1.0",
    "protobuf>=4.25,<7.0",
    "ultralytics>=8.3.0,<9.0",
    "tqdm>=4.66,<5.0",
    "gguf>=0.19.0",
]
```

### `requirements.txt` 片段（不使用 uv 时的等价写法）

```text
# 安装（CPU 版 torch）
pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  torch==2.9.1 torchvision==0.24.1 \
  transformers==4.49.0 optimum==1.27.0 optimum-intel==1.27.0 \
  nncf==3.3.0 openvino==2026.3.1 \
  onnx==1.22.0 onnxruntime==1.29.0 onnxsim==0.7.3 \
  gguf==0.19.0 sentencepiece protobuf polars pillow numpy openpyxl \
  fastapi 'uvicorn[standard]' sqlmodel alembic 'psycopg[binary]' \
  arq redis httpx 'pydantic>=2' pydantic-settings tqdm
```

> 精确复现实验环境请直接用仓库的 `uv.lock`：`uv sync --frozen --no-dev --package app`。

---

## 4. 系统依赖

| 组件 | 版本/要求 | 用途 | 安装命令 |
|---|---|---|---|
| **Docker Engine** | ≥ 20.10（实测 29.x） | 运行 backend/worker/db/redis；`scripts/fetch-open-datasets.sh` 也复用平台镜像做数据转换 | `curl -fsSL https://get.docker.com \| sh` |
| **Docker Compose** | v2（`docker compose`） | `compose.cpu.yml` / `compose.gpu.yml` | 随 Docker Desktop / `docker-compose-plugin` |
| **PostgreSQL** | 18（compose 内置 `postgres:18`；17-alpine 亦可） | 模型/数据集/压缩任务元数据 | compose 托管，无需宿主安装 |
| **Redis** | 7（compose 内置 `redis:7-alpine`） | Arq 任务队列（异步压缩任务） | compose 托管 |
| **nvidia-container-toolkit** | 可选（GPU 压缩/训练时必需） | 容器内可见 CUDA（`runtime: nvidia`） | `apt install nvidia-container-toolkit` |
| **git / curl / tar / gzip / zstd** | 常用版本 | 模型拉取、数据集下载、镜像搬运 | `apt install -y git curl tar gzip zstd` |
| **libgl1 / libglib2.0-0** | 系统库 | torchvision/ultralytics 导入 | `apt install -y libgl1 libglib2.0-0`（镜像内已装） |

> **另需磁盘空间**：`backend:latest` 镜像约 7.6GB（CPU）/18.5GB（GPU），模型与产物放在 `models-storage/`；
> 建议宿主 `/` 或数据盘预留 **≥ 100GB**（含 CIFAR-10/100 + MNIST + Fashion-MNIST 约 0.4GB、Qwen2.5-0.5B 约 1GB、GGUF/IR 产物各 0.3~1GB）。

---

## 5. 硬件要求

| 场景 | CPU | 内存 | 磁盘 | GPU | 实测参考 |
|---|---|---|---|---|---|
| **最低（CPU-only，小模型）** | 4 核 x86_64（支持 AVX2） | **16GB** | 60GB SSD | 不需要 | ONNX/OpenVINO INT8 量化、剪枝、蒸馏、QAT、GGUF、LoRA 微原型 |
| **推荐（CPU-only，生产）** | **32 核** | **64GB** | 200GB NVMe | 不需要 | 本项目服务器实测：32 核/62GB，`llm_quantize` Qwen2.5-0.5B INT4 ≈ 19s（999MB → 339MB） |
| **LLM 微调 / DPO（真实 transformers 训练）** | 16 核 | 32GB+ | 100GB | **≥ 8GB 显存**（实测 RTX 4060 Laptop 8GB） | LoRA SFT/DPO 32 样本 1 epoch 数分钟；7B 级建议 ≥ 24GB |
| **GGUF q4_K_M / Q6_K（纯 Python 回退）** | 8 核 | 16GB | 60GB | 不需要 | Qwen2.5-0.5B：f16 994MB → q8_0 531MB/1.88× → q4_0 283.8MB/3.52× → q4_K_M 511MB/1.95× |

* 构建镜像（`uv sync` 拉取 torch/openvino 等 ~2GB 依赖）时**峰值内存约 4GB**，`UV_CONCURRENT_*` 已在 `backend/Dockerfile.cpu` 中降并发以适配小内存/WSL。
* 无 GPU 时 `llm_finetune` 对 HuggingFace 模型会明确报错（需要 CUDA），CPU 只跑内置 micro-LM 原型；这是设计约束而非缺陷。

---

## 6. 网络要求

| 目标 | 是否必需 | 说明 / 离线替代 |
|---|---|---|
| **PyPI / uv index** | 构建镜像时必需 | 国内可用 `https://pypi.tuna.tsinghua.edu.cn/simple`；`Dockerfile.cpu` 已内置 apt 清华源、`UV_HTTP_TIMEOUT=120`。**服务器出网不稳时的推荐做法：本地构建镜像再搬运**（`scripts/cpu-image-ship.sh`，见下） |
| **HuggingFace** | 拉模型/数据集时必需 | 平台内置多源回退：`huggingface.co` → `hf-mirror.com` → ModelScope |
| **hf-mirror.com** | 推荐 | 实测 **1.15 MB/s**（同一网络直连 `cs.toronto.edu` 仅 12 KB/s）；数据集取数默认走它 |
| **llama.cpp（GitHub）** | GGUF 转换时需要 | 平台按序尝试 github → ghfast.top → gh-proxy.com → gitclone.com → codeload tar；**离线可预置目录并挂载**：`LLAMA_CPP_PATH=/opt/llama.cpp`（CPU compose 默认挂载宿主 `/opt/app/llama.cpp`）。缺失 `llama-quantize` 时自动使用**纯 Python GGUF 量化回退**（无需编译） |
| **Docker Hub** | 首次拉基础镜像 | `python:3.11`、`postgres:18`、`redis:7-alpine`、`oven/bun:1` |
| **服务器 ↔ 开发机** | 镜像搬运时 | 实测局域网 **12 MB/s**（1.7GB 镜像 ≈ 2.5 分钟） |

### 服务器不出网/出网抖动时的标准姿势

```bash
# ① 本地构建 CPU 镜像（网络好），② 压缩传输，③ 服务器 docker load + up --no-build
bash scripts/cpu-image-ship.sh --host root@10.110.1.36
#   自动：build → save|zstd/gzip → rsync → load → 打回滚标签 backend:latest-prev → compose up -d --no-build → 校验
```

---

## 7. 一键环境自检

```bash
# 平台侧（容器内）
docker compose -f compose.cpu.yml --env-file .env.cpu exec backend python - <<'PY'
import torch, onnx, onnxruntime, openvino, nncf, transformers, optimum.intel, gguf, polars
print("torch       ", torch.__version__, "| cuda:", torch.cuda.is_available())
print("onnx/ort    ", onnx.__version__, onnxruntime.__version__)
print("openvino    ", openvino.__version__, "| nncf", nncf.__version__)
print("transformers", transformers.__version__, "| optimum-intel", optimum.intel.__version__)
print("gguf/polars ", gguf.__version__, polars.__version__)
PY

# CLI 侧
cd backend && python -m app.cli.compression --version
python -m app.cli.compression exit-codes
python -m app.cli.compression models --api http://127.0.0.1:8001 --json
```
