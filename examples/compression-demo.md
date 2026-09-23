# 端到端 Demo：把「模型压缩」接入推理模块

> 全部命令与输出均为**真机实测**（GPU 栈：RTX 4060 Laptop / 容器 `mlplatform-gpu-local-*`；CPU 栈：32 核服务器 `10.110.1.36`）。
> 参数表见 [`../docs/compression-cli.md`](../docs/compression-cli.md)，环境见 [`../docs/compression-env.md`](../docs/compression-env.md)。

---

## 场景 1（主线）：Qwen2.5-0.5B-Instruct → INT4 OpenVINO IR → 直接推理/部署

**目标**：把 953MB 的 fp32 模型压到 ~324MB INT4 IR，压缩后**不落盘到推理模块**，直接用 `task_id` 走推理接口对话。

### 1.1 前置准备

```bash
# ① 环境变量（网站后端可直接注入这些变量，无需拼参数）
export COMPRESSION_API=http://127.0.0.1:8002      # GPU 栈；CPU 服务器用 http://10.110.1.36:8001
export COMPRESSION_EMAIL=admin@gpu.com
export COMPRESSION_PASSWORD=******

# ② 确认模型已就绪（下载/上传/本地导入三选一）
compression models --framework huggingface
# MODEL     NAME                FRAMEWORK    STATUS  SIZE     VER
# --------  ------------------  -----------  ------  -------  ---
# 711f2a5c  Qwen2.5-0.5B-Local  huggingface  ready   953.3MB  v1

# ③（可选）准备评测用真实语料：先从本地目录导入 WikiText-2（reference 模式，不复制数据）
compression local-datasets
# NAME                 KIND         CLASSES  SAMPLES  SIZE     IMPORTED
# -------------------  -----------  -------  -------  -------  --------
# wikitext2-raw-test   text         -        1        1.2MB
# CIFAR10-train        imagefolder  10       50000    107.8MB  yes
```

### 1.2 执行：INT4 量化（异步 + 轮询）

```bash
$ compression quantize --model Qwen2.5-0.5B-Local --method nncf_int4 \
      --bits 4 --group-size 128 --sym true \
      --watch --poll-interval 5 --wait-timeout 600 --json
```

**实测输出**（节选）：

```json
{
  "task_id": "1a0a49d2-0c00-4247-958a-5f5c19d8b68e",
  "model_name": "Qwen2.5-0.5B-Local",
  "method": "llm_quantize",
  "status": "succeeded",
  "progress": 100.0,
  "output_size_bytes": 339069306,
  "input_size_bytes": 999602900,
  "artifact_available": true,
  "download_url": "/api/v1/compressions/1a0a49d2-.../download",
  "deploy_export_url": "/api/v1/compressions/1a0a49d2-.../deploy-export",
  "metrics": {
    "engine": "nncf", "bits": 4, "group_size": 128, "sym": true,
    "llm": true, "precision": "int4",
    "size_reduction_pct": 66.08, "ratio": 2.9481
  }
}
```

stderr（人工模式下的进度，`--json` 时被抑制）：

```text
submitted  task_id=1a0a49d2-...  method=llm_quantize  status=pending
  running      15.0%
  running      35.0%
  running      60.0%
  running      95.0%
  succeeded   100.0%
```

> 同步写法（不等待，立即返回给前端）：
> `compression quantize --model Qwen2.5-0.5B-Local --method nncf_int4 --json` → exit 0，拿到 `task_id`；
> 之后 `compression task status <id> --json` 或平台 `GET /api/v1/compressions/{id}` 轮询。

### 1.3 产物与使用

| 产物 | 位置 | 用途 |
|---|---|---|
| OpenVINO IR | `models-storage/tasks/<task_id>/model.xml` + `model.bin` + tokenizer + `engine_nncf.txt` | **推理**：`POST /api/v1/inference/chat/completions`（传 `task_id`） |
| 产物 zip | `task download <task_id>` → `exports/artifact_<task_id>.zip`（实测 **290 MB**） | 归档/回传 |
| 部署包 | `task deploy <task_id>` 或 `export --format deploy_zip --task <task_id>` → `deploy_<task_id>.zip`（实测 **290 MB**） | 内含 `Dockerfile` + `serve.py` + `model_card.json` + `model/`，可直接 `docker build` |

```bash
# ① 直接用压缩结果对话（推理模块无需知道压缩细节）
curl -s http://127.0.0.1:8002/api/v1/inference/chat/completions \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"task_id":"1a0a49d2-0c00-4247-958a-5f5c19d8b68e",
       "messages":[{"role":"user","content":"用一句话说明模型量化的作用"}],
       "max_tokens":64,"temperature":0.7}'

# ② 拉取部署包并本地起服务
compression task deploy 1a0a49d2-0c00-4247-958a-5f5c19d8b68e --out-dir ./exports
unzip -o exports/deploy_1a0a49d2-*.zip -d ./qwen-int4
cd qwen-int4 && docker build -t qwen-int4 . && docker run -p 8080:8000 qwen-int4
```

### 1.4 真实语料质量评估（可选，强烈推荐）

```bash
$ compression quantize --model Qwen2.5-0.5B-Local --method nncf_int4 \
      --set 'eval={"type":"perplexity","dataset_id":"2f23d4e5-8e4c-4011-83a0-7f4e8211f84d"}' \
      --watch --json
```

实测 `metrics.eval`：

```json
{
  "type": "perplexity",
  "perplexity": 15.835,
  "samples": 512,
  "text_source": "dataset_id: 2f23d4e5-8e4c-4011-83a0-7f4e8211f84d (100000 chars)",
  "note": "engine=nncf light eval (single forward pass over the eval text)"
}
```

> `--set` 可透传任意平台参数（平台加新参数无需改 CLI）；也可用 `COMPRESSION_PARAMS='{"eval":{...}}'`。

---

## 场景 2：CIFAR-10 剪枝 → ONNX → 分类评测（CPU 服务器）

**目标**：用平台内本地 CIFAR-10 数据集驱动 Group Fisher 剪枝，导出 ONNX，并在 1 万张测试图上评测。

```bash
export COMPRESSION_API=http://10.110.1.36:8001
export COMPRESSION_EMAIL=admin@example.com
export COMPRESSION_PASSWORD=******

# ① 数据驱动剪枝（实测 8 秒完成）
$ compression prune --model CifarCNN10 --method group_fisher --rate 0.3 \
      --dataset CIFAR10-train --input-shape 1x3x32x32 --watch --json
```

**实测输出**：

```json
{
  "task_id": "aecd3aa5-4d8f-40f4-8cf2-f49d4460321e",
  "method": "prune",
  "status": "succeeded",
  "output_size_bytes": 381321,
  "metrics": {
    "total_filters": 224,
    "pruned_filters": 69,
    "prune_algorithm": "group_fisher",
    "params_after": 45915,
    "forward_check": "ok",
    "head_layers_adjusted": 1,
    "onnx_exported": true,
    "size_reduction_pct": 2.78
  },
  "created_at": "2026-09-18T04:27:55Z",
  "finished_at": "2026-09-18T04:28:04Z"
}
```

* `head_layers_adjusted=1`：分类头已随通道剪枝自动对齐（否则会出现 `mat1 and mat2 shapes cannot be multiplied`）
* `forward_check=ok`：剪枝后前向自检通过；分支型拓扑会返回 `topology_warning=true` 而不是静默产出坏模型

```bash
# ② 导出 ONNX（同一模型的另一种部署格式）
$ compression export --model CifarCNN10 --format onnx --input-shape 1x3x32x32 --watch --json
# 实测：status=succeeded, output_size_bytes=380919, metrics {"model_kind":"cnn","params":94986}

# ③ 分类评测：10,000 张 CIFAR10-test
#    metrics.eval = {"samples":10000,"accuracy":0.0975,"baseline_accuracy":0.0969,
#                    "accuracy_delta":0.0006}
```

> 该模型权重为随机初始化的冒烟模型，故 accuracy≈10%（10 分类随机水平）；
> 换成本平台训练好的 checkpoint 即为真实精度，**评测链路本身已验证**（1 万张图跑通）。

---

## 场景 3：文本数据 + LoRA / DPO（本地语料）

```bash
# ① LoRA SFT（GPU 栈，真 transformers 训练；32 条 Alpaca 指令）
$ compression finetune --model Qwen2.5-0.5B-Local --method lora \
      --dataset alpaca-train-32 --epochs 1 --hf-max-len 128 --watch --json
# 实测：{"samples":32,"lora_rank":8,"trainable_params":1081344,
#        "initial_loss":1.276917,"final_loss":0.87922,"output_bytes":1980634708}

# ② DPO 偏好优化（32 条 HH-RLHF 偏好对）
$ compression finetune --model Qwen2.5-0.5B-Local --method dpo \
      --dataset hh-rlhf-train-32 --epochs 1 --dpo-beta 0.1 --watch --json
# 实测：{"beta":0.1,"initial_loss":0.693147,"final_loss":0.644608}
#   初始 loss 恰为 ln2 ≈ 0.6931 —— 证明 chosen/rejected 对被正确读取并参与训练

# ③ CPU 服务器（PyTorch micro 原型，无需 GPU）
$ compression finetune --model CifarCNN10 --method lora --dataset alpaca-train-256 --epochs 1 --json
# 实测：{"samples":256,"trainable_params":21944,"preference_pairs":0,"dpo_ready":false}
$ compression finetune --model CifarCNN10 --method dpo --dataset hh-rlhf-train-256 --json
# 实测：{"samples":256,"preference_pairs":256,"dpo_ready":true}
```

---

## 场景 4：GGUF 导出（llama.cpp / ollama 生态）

```bash
$ compression export --model Qwen2.5-0.5B-Local --format gguf --outtype q4_K_M --watch --json
```

**实测矩阵**（f16 基线 994MB）：

| outtype | 产物大小 | 压缩比 | 路径 | 备注 |
|---|---|---|---|---|
| `f16` | 994 MB | 1.00× | llama.cpp 转换器直出 | 精度基准 |
| `q8_0` | 531 MB | 1.88× | 转换器直出 | 推荐默认 |
| `q4_0` | 283.8 MB | 3.52× | **Python GGUF 回退** | 无需编译 |
| `q4_K_M` | **511 MB** | **1.95×** | Python 混合（Q4_K + Q6_K/Q4_0 回退） | `metrics.tool="python-K(q4_K_M)"` |
| `q6_K` | — | — | Python 回退 | 维度能被 256 整除时启用 |

> 缺少 `llama-quantize` 时自动走纯 numpy/gguf 回退（`metrics.tool` 标注 `python-K(...)`），
> 离线环境同样能产出 K-quant；预置 `LLAMA_CPP_PATH` 则使用官方二进制。


---

## 5. 产物位置与「如何使用」总览

```text
models-storage/                          # = MODEL_STORAGE_PATH（容器内 /app/backend/models-storage）
├── models/<model_uuid>/original*         # 原始模型（上传/URL 拉取/本地导入）
├── tasks/<task_uuid>/                    # 压缩产物统一目录
│   ├── model.onnx / model_quantized.onnx / model_qat.onnx           # ONNX 系
│   ├── model.xml + model.bin (+ engine_nncf.txt)                    # OpenVINO IR 系
│   ├── model_pruned.pt / model_pruned.onnx                          # 剪枝系
│   ├── adapter_lora.pt / model_lora_merged.pt / finetune_meta.json   # LoRA 系
│   ├── adapter_dpo.pt / dpo_meta.json / preference_pairs.jsonl       # DPO 系
│   ├── model.gguf                                                    # GGUF
│   └── deploy/ (+ deploy_package.zip)                                # 部署包
└── datasets/<dataset_uuid>/              # upload/copy 模式的数据集（reference 模式不落盘）
```

| 想做什么 | 命令 / 接口 | 说明 |
|---|---|---|
| 查看任务产物与指标 | `compression task status <id>` | 表格含 `download_url` / `deploy_export_url` |
| 下载产物 | `compression task download <id> --out-dir ./exports` | 目录产物自动打 zip |
| 生成部署包 | `compression task deploy <id>` | Dockerfile + `serve.py` + `model_card.json` + `model/` |
| 直接推理（不下载） | `POST /api/v1/inference/predict`（CNN/ONNX/OV）或 `/inference/chat/completions`（LLM IR） | 只需传 `task_id` |
| 重新评估 / 基准 | `POST /api/v1/compressions/{id}/reevaluate {"kind":"eval"}` | 复用任务内已存的 eval 配置 |

### 5.1 命令行作业（`compress_job.py`）的输出目录、日志与进度

用「单文件 CLI」跑作业（调度器旁挂 / 离线单机）时，规则固定：

* **输出目录参数只叫 `--output-dir`**（环境变量 `COMPRESS_JOB_OUTPUT_DIR`；无 `--out/--outdir` 别名），
  产物、报告、日志、进度全在它下面：

```
<output-dir>/
├── artifact/       压缩产物（model.onnx / model_quantized.onnx / IR 目录 / *.gguf …）
├── result.json     总报告（含 inputs 清单：参数名/来源 cli·env·default/解析结果）
├── metrics.json    纯指标
├── logs.jsonl      结构化日志（与 stdout 同内容，逐条 JSON Lines）
├── progress.json   进度快照（status/step/percent/message/elapsed_s/artifact）
├── progress.log    进度里程碑（每 10% 一行，可 tail -f）
└── _stage/         暂存副本（内联后的 ONNX、arch.py …），跑完可删
```

* **日志两条腿**：stdout 实时打印（默认 JSON Lines；`--no-json-logs` 人类可读），同时追加到
  `<output-dir>/logs.jsonl`；
* **进度**：`--progress-interval`（默认 10）保证每个步骤 `0/10/…/100` 每档一行 `progress.log` +
  一条 `progress_milestone` 日志；`progress.json` 原子更新；失败/中断也落盘 `status=failed/interrupted`；
  `--no-progress-file` 可关闭；
* 摘要末尾会打印「输出目录 + 已写出文件」，便于调度器核对：
  `输出目录 : /out/1001` / `  已写出 : artifact  result.json  metrics.json  logs.jsonl  progress.json  progress.log`。

详见 [`docs/compression-job.md`](../docs/compression-job.md) §5.0/§5.2/§5.3。

---

## 6. 集成到综合网站（推理模块）

### 6.1 后端：提交压缩任务

```python
"""推理模块的「压缩前置」服务:提交任务 → 轮询 → 拿到可推理的 task_id。"""
import httpx

COMPRESSION_API = "http://10.110.1.36:8001/api/v1"   # 内网服务地址
SERVICE_TOKEN = "…"                                   # 平台服务账号 token

METHOD_MAP = {                 # 推理模块对外的方法名 → 平台 method + 默认参数
    "int4":          ("llm_quantize",  {"engine": "nncf", "bits": 4, "group_size": 128, "sym": "true"}),
    "int8_openvino": ("openvino_int8", {"precision": "int8", "calibration_samples": 8}),
    "onnx_int8":     ("quantize_onnx", {"opset": 12}),
    "gguf_q4km":     ("export_gguf",   {"outtype": "q4_K_M"}),
    "prune_l2":      ("prune",         {"algorithm": "l2", "rate": 0.3, "export_onnx": True}),
    "prune_fisher":  ("prune",         {"algorithm": "group_fisher", "rate": 0.3}),
    "distill":       ("distill",       {"method": "logits", "temperature": 4.0, "alpha": 0.7}),
    "lora":          ("llm_finetune",  {"stage": "sft", "epochs": 1}),
    "dpo":           ("llm_finetune",  {"stage": "dpo", "dpo_beta": 0.1}),
}

async def submit_compression(model_id: str, ui_method: str, **overrides) -> dict:
    platform_method, params = METHOD_MAP[ui_method]
    params = {**params, **overrides}
    async with httpx.AsyncClient(base_url=COMPRESSION_API, timeout=60) as client:
        response = await client.post(
            "/compressions/",
            headers={"Authorization": f"Bearer {SERVICE_TOKEN}"},
            json={"model_id": model_id, "method": platform_method, "params": params},
        )
        response.raise_for_status()
        task = response.json()          # 立即返回（异步），含 task_id
    return {"task_id": task["id"], "method": platform_method, "status": task["status"]}
```

### 6.2 后端：查询进度 + 推理入口

```python
AUTH = {"Authorization": f"Bearer {SERVICE_TOKEN}"}

async def task_progress(task_id: str) -> dict:
    async with httpx.AsyncClient(base_url=COMPRESSION_API, timeout=30) as client:
        task = (await client.get(f"/compressions/{task_id}", headers=AUTH)).json()
    return {
        "status": task["status"], "progress": task["progress"],
        "error": task["error_message"], "metrics": task["metrics"],
        "ready": task["status"] == "succeeded" and bool(task["output_size_bytes"]),
    }

async def infer(task_id: str, payload: dict) -> dict:
    """压缩完成即可推理:CNN/ONNX/OV 用 predict,LLM IR 用 chat/completions。"""
    endpoint = "/inference/chat/completions" if payload.get("chat") else "/inference/predict"
    async with httpx.AsyncClient(base_url=COMPRESSION_API, timeout=300) as client:
        return (await client.post(endpoint, headers=AUTH, json={"task_id": task_id, **payload})).json()
```

### 6.3 前端展示位置建议

| 页面/区域 | 展示内容 | 数据来源 |
|---|---|---|
| 推理 → 模型详情 → **「优化」标签页** | 压缩方法下拉（INT4 / INT8-OV / ONNX-INT8 / GGUF / 剪枝 / 蒸馏 / LoRA / DPO）+ 参数表单 + `提交` | `POST /compressions/` |
| 同页 → **任务卡片** | 进度条 + 阶段文案 + 指标徽标（体积↓%、压缩比、`forward_check`、`dpo_ready`） | `GET /compressions/{id}`（1~3s 轮询或 SSE） |
| 同页 → **产物面板** | 产物清单、`下载 zip`、`生成部署包`、`用此模型推理` | 任务 `status` / `download` / `deploy-export` |
| 推理 → 试运行 | `predict` / `chat` 直连（自动带 `task_id`） | `/inference/predict`、`/inference/chat/completions` |
| 运维后台 | 任务列表、失败原因、重试/删除 | `GET /compressions/`、`retry`、`DELETE` |

> 核心约定：**推理模块只保存 `task_id`**。产物留在平台侧，推理接口凭 `task_id` 定位；
> 需要下发到独立环境时再用 `deploy_zip` 走容器化发布，`model_card.json` 含 `method/params/metrics` 可直接入库做版本追踪。



---

## 7. 常见问题（实测踩坑）

| 现象 | 原因 | 处理 |
|---|---|---|
| `exit 4 / no model matches 'X'` | 模型名错误或状态未 ready | `compression models` 查看；名称可用唯一前缀 |
| `exit 5 / 422` 提示需要 dataset | 数据驱动方法（group_fisher/dmcp/autoslim）缺 `--dataset` | `compression local-datasets` 选数据集后补上 |
| `llm_finetune ... requires a CUDA runtime` | CPU 服务器对 HuggingFace 模型做训练 | 切 GPU 栈（`COMPRESSION_API=…:8002`）或改用 PyTorch 模型走 micro 原型 |
| GGUF `q4_0/q4_K_M` 报缺工具 | 无 `llama-quantize` | 平台自动 Python 回退（`metrics.tool=python-K(...)`）；需要官方速度时挂载 `LLAMA_CPP_PATH` |
| `predict` 报输入维度不符 | ONNX/IR 已按 `input_shape` 固化 | 提交时统一 `--input-shape`（如 `1x3x32x32`） |
| 剪枝后推理报 `mat1 and mat2 shapes` | 旧版本未对齐分类头 | 已修复（看 `head_layers_adjusted`）；`topology_warning=true` 说明是分支拓扑，需自定义 `arch.py` |
| 下载 401 | 未携带 token | CLI 自动登录；直连 API 时带 `Authorization: Bearer` |
| `--watch` 长时间无输出 | 任务排队 / CPU 繁忙 | 调大 `--wait-timeout`，或去掉 `--watch` 用 `task status` 轮询 |

### 复现 Demo 的最短路径

```bash
# 0) 环境
cp .env.cpu.example .env.cpu && vim .env.cpu        # 填 SECRET_KEY / 密码
docker compose -f compose.cpu.yml --env-file .env.cpu up -d --build

# 1) 数据（可选：CIFAR-10/MNIST + WikiText-2 + Alpaca）
bash scripts/fetch-open-datasets.sh --only cifar10,wikitext2,alpaca

# 2) 模型（本地导入或 URL 拉取；HF 直连慢时自动走 hf-mirror）

# 3) 压缩（本 Demo 场景 1~4 任选）
export COMPRESSION_API=http://127.0.0.1:8001 COMPRESSION_EMAIL=… COMPRESSION_PASSWORD=…
compression quantize --model <模型名> --method nncf_int4 --watch --json
```
