# 模型压缩作业交付包（deploy/compress）

本目录是「模型压缩」能力的**打包交付件**，同事拿到后可以**只用命令行**完成模型压缩：

```
deploy/compress/
├── README.md                 # 本文（交付包说明 / 快速开始）
├── Dockerfile.cpu            # CPU 作业镜像（独立构建，含 torch/onnx/onnxruntime/openvino/nncf/transformers）
├── Dockerfile.gpu            # GPU 作业镜像（CUDA 12.6 wheels，供 LoRA/DPO 微调）
├── Dockerfile.fast           # 快速构建：复用已有平台镜像（秒级）
├── llama-cpp/                # 可选：离线 GGUF 转换工具（放 convert_hf_to_gguf.py）
└── examples/
    ├── welding_eval.py       # 表格分类模型精度/延迟评测脚本（复现训练切分）
    ├── welding_demo.sh       # 焊接模型压缩 demo 一键脚本
    └── results/              # 本次 demo 的原始结果（评测 JSON、result.json、失败样例、INT8 产物）
```

配套（仓库根目录）：

| 文件 | 说明 |
|------|------|
| `compress_job.py` | 单文件压缩作业 CLI（镜像内为 `/app/compress_job.py`） |
| `backend/app/jobs/compression.py` | **压缩引擎实现体**（量化/剪枝/稀疏化/蒸馏/QAT/微调/GGUF），CLI 只是它的壳 |
| `pyproject.toml` + `uv.lock` + `backend/pyproject.toml` | 依赖声明与锁定（镜像与宿主机安装共用） |
| `scripts/compress-image.sh` | 构建镜像 / `docker save` 出 tar / 冒烟自检 |
| `scripts/compress-bundle.sh` | 打包**代码包**（`--bundle-only`）或**全量包**（含镜像 tar） |
| `scripts/check_compress_package.py` | **代码包自检**（布局/引擎入口/CLI），见 §2 |
| `docs/compression-job.md` | **命令行与参数说明**（每个参数的含义、取值、作用） |
| `docs/compression-report.md` / `exp.md` | **实验报告**（压缩率/精度损失实测） |
| `examples/compression-demo-welding.md` | **demo 文档**（焊接模型逐步操作 + 结论） |

---

## 1. 快速开始（3 步）

```bash
# ① 构建镜像（一次即可）
docker build -f deploy/compress/Dockerfile.cpu -t compress-job:cpu .
#   已有平台镜像 backend:cpu-ship 时，可用秒级构建：
#   docker build -f deploy/compress/Dockerfile.fast -t compress-job:cpu --build-arg BASE_IMAGE=backend:cpu-ship .

# ② 跑一个真实压缩作业（ONNX 权重 INT8）
docker run --rm \
  -v /data/models:/models -v /data/out:/out \
  compress-job:cpu \
    --job-id=1001 --tenant-id=acme --trace-id=7f3c \
    --base-model-path=/models/model.onnx \
    --output-dir=/out/1001 \
    --method=quantize --algo=onnx_dynamic_int8

# ③ 读取结果
cat /data/out/1001/result.json      # artifact_bytes / metrics / duration_s
ls  /data/out/1001/artifact/        # model_quantized.onnx
```

调度器模板变量直接对应：

```bash
python /app/compress_job.py \
  --job-id={{job.id}} --tenant-id={{job.tenant_id}} --trace-id={{job.trace_id}} \
  --base-model-path={{inputs.base_model_path}} \
  --dataset-path={{inputs.dataset_path}} \
  --output-dir={{inputs.output_path}} \
  --method=quantize --algo=onnx_dynamic_int8 \
  --epochs=80 --batch-size=16 --learning-rate=0.001
```

---

## 2. 代码包：内容、自检、三种运行方式

代码包（`dist/compress-code-<日期>.zip`，约 0.8 MB）是**可运行的最小源码树**，不是文档合集：

```
compress-code-<日期>/
├── compress_job.py                    # 单文件 CLI（壳）
├── backend/app/jobs/compression.py    # ★ 压缩引擎实现体（185 KB）
├── backend/app/{models.py,core/,crud.py,jobs/storage.py,…}
├── backend/pyproject.toml + pyproject.toml + uv.lock   # 依赖与锁定
├── deploy/compress/Dockerfile.{cpu,gpu,fast} + examples/
├── scripts/check_compress_package.py  # 自检脚本
└── docs/ · examples/ · SHA256SUMS
```

### 2.1 先自检（拿到包的第一件事）

```bash
python scripts/check_compress_package.py                     # 布局 + 语法 + 引擎导入 + --dry-run
python scripts/check_compress_package.py --run model.onnx    # 再真跑一次量化作业
# 期望输出：结论：自检通过 —— 代码包自带压缩引擎，`python compress_job.py …` 可直接运行。
```

自检覆盖：17 个关键文件（含引擎）、63 个 `.py` 语法、17 个引擎入口
（`quantize_onnx`/`onnx_int4`/`quantize_onnx_static`/`openvino_int8`/`prune`/`sparsify`/`distill`/`qat`/
`llm_finetune`/`llm_quantize`/`export_gguf`…）、`CompressionMethod` 12 个枚举、CLI `--help`/`--dry-run`。

### 2.2 运行方式

```bash
# ① 宿主机直接跑（需自备 Python 依赖；与镜像同源、按 uv.lock 锁定版本）
cd compress-code-<日期>
pip install uv && uv sync --no-dev --package app
python compress_job.py --job-id=1001 --base-model-path=/models/model.pt \
  --output-dir=/out/1001 --method=quantize --algo=onnx_dynamic_int8

# ② 在代码包里构建镜像（Dockerfile.cpu 只用本包内容，可离线复现）
docker build -f deploy/compress/Dockerfile.cpu -t compress-job:cpu .

# ③ 用环境包镜像（推荐，零依赖安装）
scripts/compress-image.sh load dist/compress-job-cpu-<日期>.tar.gz
```

> 提示：在**仓库根目录**直接 `python compress_job.py` 时，同目录的启动器 `compression.py`
> 会遮蔽标准库 `compression`；脚本已内置 `_drop_shadowing_self_dir()` 自动规避（见
> `docs/compression-job.md` 排错表）。

---

## 3. 镜像分发（`docker save` / `docker load`）

打包分两个包，可分别传输：

| 包 | 文件 | 大小 | 内容 |
|----|------|------|------|
| **环境包** | `dist/compress-job-cpu-<日期>.tar.gz`（+`.sha256`） | 约 1.7 GB | Docker 镜像（全部 Python 依赖与运行时），`docker load` 即得 `compress-job:cpu` |
| **代码包** | `dist/compress-code-<日期>.zip` | 几百 KB | CLI 源码、Dockerfile、文档、demo、实测结果（**不含镜像**） |
| 全量包 | `dist/compress-full-<日期>.zip` | 约 1.7 GB | 代码包内容 + `image/` 下的环境包，一次拿齐 |

```bash
# 打包方
scripts/compress-image.sh build      # 或 build-fast / build-gpu
scripts/compress-image.sh save       # -> dist/compress-job-cpu-<日期>.tar.gz（环境包）
scripts/compress-bundle.sh --bundle-only   # -> dist/compress-code-<日期>.zip（代码包）
scripts/compress-bundle.sh                 # -> dist/compress-full-<日期>.zip（全量包，复用已导出的 tar）

# 使用方：导入后即可运行（环境包）
sha256sum -c compress-job-cpu-<日期>.tar.gz.sha256     # 可选校验
scripts/compress-image.sh load dist/compress-job-cpu-<日期>.tar.gz
docker run --rm compress-job:cpu --help
# 只有代码包时自行构建镜像：
docker build -f deploy/compress/Dockerfile.cpu -t compress-job:cpu .
```

镜像内已内置代码快照（`/app/compress_job.py`、`/app/examples/`）；拿新代码热更新正在运行的容器：

```bash
docker cp compress_job.py                 <container>:/app/compress_job.py
docker cp backend/app/jobs/compression.py <container>:/app/backend/app/jobs/compression.py
docker cp backend/app/models.py           <container>:/app/backend/app/models.py
```

镜像内约定：

| 项 | 值 |
|----|----|
| 入口 | `python /app/compress_job.py`（`ENTRYPOINT`，直接跟参数） |
| 平台代码 | `/app/backend/app`（`PYTHONPATH=/app/backend`） |
| **输出目录参数** | **只叫 `--output-dir`**（无 `--out/--outdir` 别名）；产物/报告/日志/进度都写在它下面 |
| 输出目录内容 | `artifact/` · `result.json` · `metrics.json` · `logs.jsonl` · `progress.json` · `progress.log` · `_stage/` |
| 日志 | 同时 `print` 到 stdout（默认 JSON Lines）**并**追加写 `<output-dir>/logs.jsonl` |
| 进度 | 每 10%（`--progress-interval` 可改）写一行 `progress.log`，`progress.json` 是原子更新的快照；失败也会落盘 |
| 离线 GGUF | `/opt/llama.cpp`（`LLAMA_CPP_PATH`），见 `llama-cpp/README.md` |
| 自检 | `docker run --rm compress-job:cpu --help` / `--dry-run …`（`scripts/compress-image.sh smoke`） |

### 版本核对（同名新旧包必看）

交付包内 `BUILD_INFO.txt` 给出构建时间、git 提交与关键文件 sha256；`SHA256SUMS` 可逐文件校验；
zip 同级有 `<zip>.sha256`。快速判断「是否含 2026-09-22 的 ONNX 自动降级修复」：

```bash
grep -c _prepare_quant_input backend/app/jobs/compression.py    # ≥1 = 有修复（只换 compress_job.py 不够）
python scripts/check_compress_package.py                        # 布局/引擎/CLI/进度文件自检
```

---

## 4. 能力速查（详见 `docs/compression-job.md`）

| 需求 | 命令 |
|------|------|
| CNN/Transformer 权重 INT8（最常用） | `--method=quantize --algo=onnx_dynamic_int8` |
| **ONNX 外部权重（`.onnx` + `.onnx.data`）** | 额外加 `--external-data-path <文件或目录>`（必填，缺失退出码 3） |
| 静态 INT8（有校准数据，精度更优） | `--method=quantize --algo=onnx_static_int8 --dataset-path=<校准图片> --max-samples=128` |
| OpenVINO IR（CPU/核显推理） | `--method=quantize --algo=openvino_int8 --precision=int8` |
| LLM INT4（Qwen/Llama） | `--method=quantize --algo=nncf_int4 --bits=4 --group-size=128` |
| GGUF（llama.cpp） | `--method=quantize --algo=gguf --outtype=q4_K_M` |
| CNN 结构化剪枝 | `--method=prune --algo=group_fisher --rate=0.3 --dataset-path=<按类图片>` |
| 知识蒸馏 | `--method=distill --algo=logits --width-mult=0.5 --temperature=4` |
| LoRA/DPO 微调 | `--method=finetune --algo=lora|dpo`（HF 需 GPU 镜像） |
| 组合流水线 | `--method=pipeline --pipeline="prune:group_fisher,quantize:onnx_dynamic_int8"` |

退出码：`0` 成功 · `2` 参数错 · `3` 输入缺失 · `4` 方法不支持 · `5` 执行失败 · `6` 超时 · `7` 缺依赖 · `130` 中断。

---

## 5. 本次交付验证（焊接模型 demo）

| 指标 | 基线 `.pt` | INT8 ONNX | 判据 | 结论 |
|------|-----------|-----------|------|------|
| 体积 | 480.5 KB | **179.7 KB** | 压缩率 > 60% | **62.60%** ✅ |
| 准确率 | 0.935641 | 0.9350 | 精度损失 < 3% | **0.06%** ✅ |
| 延迟（CPU） | 0.664 ms/样本 | **0.164 ms/样本** | — | ≈4x 提速 |

一键复现：`deploy/compress/examples/welding_demo.sh <模型包目录> <输出目录> compress-job:cpu`
（完整说明见 `examples/compression-demo-welding.md`，原始数值见 `examples/results/`）。
