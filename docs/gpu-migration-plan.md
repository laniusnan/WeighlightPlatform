# GPU 迁移准备计划（Planning, not active）

> 状态：**规划 + 代码骨架阶段**。生产仍运行 `COMPUTE_BACKEND=cpu`（默认），
> GPU 实现为骨架（`NotImplementedError`），逐方法使能后再切 `COMPUTE_BACKEND=gpu`。

## 1. 计算密集型模块盘点与 GPU 适用性评估

平台计算任务全部在 `backend/app/jobs/compression.py` 中，由 Arq worker
（`app/worker.py` → `jobs/jobs.py::run_compression`）触发，入口
`run_compression_task(task_id)` 经 `_dispatch()` 分发给具体函数。

| 功能 | 实现函数 | 计算特征 | GPU 适用性 | 原因 / GPU 策略 |
|---|---|---|---|---|
| ONNX 导出 | `export_onnx` | 一次 trace + onnxsim 化简 | 🟡 中 | 单次前向 trace 以 CPU 为主；超大模型可用 fp16 CUDA trace 省显存墙钟 |
| ONNX 动态 INT8 | `quantize_onnx` | 权重量化，计算少 | 🔴 低 | 权重级运算 CPU 足够；迁移价值低 |
| ONNX 静态 INT8 | `quantize_onnx_static` | **校准集批量前向** | 🟢 高 | 批量矩阵运算 → ORT `CUDAExecutionProvider` / PyTorch CUDA 校准 |
| OpenVINO INT8 | `openvino_int8` | NNCF 校准批量前向 | 🟢 高 | NNCF 采样/统计在 OpenVINO `GPU` 插件上执行 |
| 剪枝 | `prune` | 迭代训练 + 微调 | 🟢 高 | CUDA 训练循环（单卡或多卡数据并行） |
| 稀疏化 | `sparsify` | 前向+阈值/梯度 | 🟡 中 | 纯前向可 GPU；大模型（如 llama 权重稀疏）收益明显 |
| 蒸馏 | `distill` | teacher+student 训练 | 🟢 高 | 典型训练负载 → CUDA fp16/autocast |
| QAT | `qat` | FX QAT 微调训练 | 🟢 高 | CUDA 训练；产出静态量化 ONNX |
| GGUF 导出 | `export_gguf` | llama.cpp 转换（拉权重→f16→量化） | 🔴 低 | 转换 IO/内存密集、非矩阵主负载；可留 CPU（llama.cpp 自身有 GPU 推理但不用于本步） |
| LLM INT4/INT8 | `llm_quantize` | 权重加载+校准+NNCF 压缩 | 🟡 中 | 校准可 GPU；纯 weight-only 压缩墙钟提升有限。当前 optimum 上游 bug → engine=nncf 纯路径已验证 CPU 可用 |
| 推理 predict | `_make_runner` + `/inference/predict` | 单样本同步 | 🔴 低 | **轻量单样本 CPU 更优**（避免传输/调度开销）；批量/服务化后再考虑 GPU |
| LLM chat | `chat/completions` | 自回归 | 🟢 高 | naive（无 KV）路径可先上 GPU 提速；最终用 optimum stateful IR + KV cache |
| 评估/基准 | `run_evaluation`/`run_benchmark` | 批量推理吞吐 | 🟢 高(批量) | perplexity/分类/yolo 批量评估在 GPU 显著加速；基准需同时测 CPU 参考线 |

结论：**优先 GPU 化的三类** = ①训练类（distill/prune/qat/sparsify）②校准/批量推理类
（静态量化、评估）③LLM 推理（chat）；**维持 CPU 的三类** = 轻量单样本 predict、GGUF 转换、
ONNX 权重级动态量化。

## 2. 已落地的架构调整

```
backend/app/services/
├── __init__.py            # 公共 API（get_service_bundle / configure_runtime …）
├── base.py                # ABC + 方法族常量 + backend/device 解析 + 日志
├── compression_cpu.py     # CPU 全量 runner（委托现有 run_compression_task，零逻辑复制）
├── compression_gpu.py     # GPU 骨架（NotImplementedError + 逐方法 TODO）
├── quantize_cpu.py        # QuantizeCpuService（量化/转换族）
├── distill_cpu.py         # DistillCpuService（prune/sparsify/distill 族）
├── quantize_gpu.py        # GPU 骨架
├── distill_gpu.py         # GPU 骨架
├── inference_cpu.py       # CPU 推理（复用 _make_runner 与 _load_chat_model 同一套加载）
├── inference_gpu.py       # GPU 骨架
└── factory.py             # ServiceFactory / ServiceBundle（按 COMPUTE_BACKEND 缓存）
```

- worker 入口：`app/jobs/jobs.py::run_compression` → `get_service_bundle().runner.run(task_id)`
  - CPU：委托 `run_compression_task`（原逻辑 1:1，不经工厂也能跑同一路径）
  - GPU：骨架抛 `NotImplementedBackendError` → 任务被标记 failed 并给出明确原因
- `app/worker.py` 启动时打印 `[worker] compute backend=...`，便于 `docker logs` 排查。

## 3. 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `COMPUTE_BACKEND` | `cpu` | `cpu` / `gpu` |
| `GPU_DEVICE_IDS` | `0` | 逗号分隔设备 id；调度/日志用 |
| `CUDA_VISIBLE_DEVICES` | — | 由 compose/调度器设置；进程内只读（`resolve_devices` 以其为准） |
| `MODEL_STORAGE_PATH` | `./models-storage` | 存储根（模型/任务/数据集共用，`app/jobs/storage.py`） |
| `LOCAL_MODEL_DIRS` | `/models-local` | “本地目录导入”白名单根 |
| `WORKER_MAX_JOBS` | `1` | CPU 单任务建议 1；GPU 可调大配合 `GPU_DEVICE_IDS` |

路径已全部由配置/`storage.py` 管理，代码无硬编码仓库路径。

## 4. GPU 使能步骤（未来）

1. 在 GPU 服务器配好驱动 + CUDA（本仓库组合：py3.11 / torch2.9.1 / transformers4.49，
   cu126 轮子见 `backend/requirements-gpu.txt`）。
2. 逐方法实现 `*_gpu.py`，从 `NOT_IMPLEMENTED_YET` 移除并保留 CPU 回退注释。
3. `COMPUTE_BACKEND=gpu` 跑 `python backend/scripts/check_backend.py`（确认
   `torch.cuda.is_available()` / devices / 各服务实例）。
4. 建一条冒烟任务（如 `openvino_int8` 或 `distill`）验证产物与 CPU 基线误差一致。
5. worker 镜像可继续复用 `backend/Dockerfile.cpu`（CUDA 基镜像变体另建
   `Dockerfile.gpu`，Docker Compose 侧 profile 化，不影响 CPU 栈）。

## 5. 回归与验收清单

- [ ] `python -m py_compile` 覆盖新增/修改文件
- [ ] `python backend/scripts/check_backend.py` → 显示 `backend=cpu`
- [ ] `COMPUTE_BACKEND=gpu ... check_backend.py` → gpu 实例为骨架（不崩溃）
- [ ] CPU worker 日志出现 `[worker] compute backend={'backend': 'cpu', ...}`
- [ ] CPU 上跑通一条既有压缩任务（回归不变）
- [ ] API 接口、数据库模型、前端均未改动

## 6. 里程碑 A：GPU 推理已实测（2026-09，本机 RTX 4060）

首个 GPU 实现 = `InferenceGpuService`（`backend/app/services/inference_gpu.py`）：
- LLM HF 本地目录：`fp16 + CUDA` 朴素逐 token 生成（无 KV），已带 chat template；
- `.onnx`：onnxruntime `CUDAExecutionProvider`（无 EP 时自动回落 CPU 并告警）；
- OpenVINO IR：尝试 `GPU` 插件（Intel），NVIDIA 主机自动回落 CPU 并告警。

本地实测（Qwen2.5-0.5B-Instruct，RTX 4060 8GB，cu126 torch2.9.1）：
```
text: 我是来自阿里云的超大规模语言模型，我叫通义千问。
prompt 33 · completion 17 · load 1.65s · latency 2420ms · device cuda
```
复现：`cd backend && PYTHONPATH=. ../.venv311gpu/bin/python scripts/test_gpu_inference.py \
  /home/lanius/Model/Qwen2.5-0.5B-Instruct "用一句话介绍你自己" 48`
（`.venv311gpu` 安装：cu126 index，注意在仓库目录外执行避免 pyproject CPU 源覆盖。）

> 说明：当前 API 路由仍走 CPU（符合“不改变现有 CPU 行为”约束）；接入方式为
> `get_service_bundle("gpu").inference.chat_completion(...)`，未来可在路由层按
> `COMPUTE_BACKEND` 加一行选择（接口不变）。
