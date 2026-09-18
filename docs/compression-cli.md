# 模型压缩 CLI 参考（compression-cli）

> 统一入口、异步任务、环境变量覆盖、表格/JSON 双输出。
> 代码位置：`backend/app/cli/compression.py`（入口）、`backend/app/cli/commands/*`（子命令）。
> 环境依赖见 [`compression-env.md`](./compression-env.md)，端到端示例见 [`../examples/compression-demo.md`](../examples/compression-demo.md)。

---

## 1. 统一入口

```bash
# ① 模块方式（任意已安装环境可用，cwd 必须是 backend/）
cd backend
python -m app.cli.compression <subcommand> [options]

# ② 仓库内便捷包装（自动挑选 venv，无需 cd）
scripts/compression <subcommand> [options]

# ③ 容器内（生产环境常用）
docker compose -f compose.cpu.yml exec backend python -m app.cli.compression <subcommand>
```

### 子命令总览

| 子命令 | 作用 | 平台方法映射 |
|---|---|---|
| `quantize` | 量化（ONNX 动态/静态 INT8、OpenVINO INT8/FP16、NNCF INT4、GGUF、QAT） | `quantize_onnx` / `quantize_onnx_static` / `openvino_int8` / `llm_quantize` / `export_gguf` / `qat` |
| `prune` | 结构化剪枝（l1/l2/bnslim/autoslim/group_fisher/dmcp）与非结构化稀疏化 | `prune` / `sparsify` |
| `distill` | 知识蒸馏（Hinton logits KD / FitNets 特征对齐） | `distill` |
| `finetune` | 微调（LoRA SFT / DPO 偏好优化） | `llm_finetune` |
| `export` | 导出 ONNX / OpenVINO IR / GGUF，或下载部署包 | `export_onnx` / `quantize_onnx` / `openvino_int8` / `export_gguf` / *deploy-export* |
| `task` | 异步任务查询：`status` `list` `wait` `download` `deploy` `delete` | *`/compressions/*`* |
| `models` / `datasets` / `local-datasets` | 资产盘点（用于解析 `--model` / `--dataset` 引用） | `/models/*` `/datasets/*` |
| `exit-codes` / `env` | 打印退出码契约 / 打印生效的 `COMPRESSION_*` 变量 | — |

### 异步任务模型

```text
压缩类子命令（quantize / prune / distill / finetune / export）
   └─ 提交后**立即返回 task_id**（exit 0）
        ├─ 加 --watch：持续轮询直到终态（succeeded → exit 0；failed → exit 6；超时 → exit 7）
        └─ 不加 --watch：稍后用 `task status <id>` / `task wait <id>` 查询

产物统一存放：<MODEL_STORAGE_PATH>/tasks/<task_id>/（默认 models-storage/tasks/<task_id>/）
下载：task download <id>        （产物 zip）
部署：task deploy <id> / export --format deploy_zip --task <id>   （Dockerfile + serve.py + model_card.json + model/）
```

---

## 2. 全局参数与环境变量覆盖

**优先级：命令行参数 > 环境变量 > 内置默认值**。
每个 `--flag` 都对应 `COMPRESSION_<FLAG大写_下划线>`，例如 `--group-size` → `COMPRESSION_GROUP_SIZE`。

| 参数 | 类型 | 默认值 | 环境变量 | 说明 |
|---|---|---|---|---|
| `--api` | URL | `http://127.0.0.1:8001` | `COMPRESSION_API` / `COMPRESSION_API_BASE` | 平台 API 地址；带或不带 `/api/v1` 均可 |
| `--token` | str | — | `COMPRESSION_TOKEN` | Bearer token（优先于邮箱密码） |
| `--email` | str | — | `COMPRESSION_EMAIL` | 登录邮箱（无 token 时自动换取 token） |
| `--password` | str | — | `COMPRESSION_PASSWORD` | 登录密码（未提供且为 TTY 时交互输入） |
| `--timeout` | float | `60` | `COMPRESSION_TIMEOUT` | 单次 HTTP 超时（秒） |
| `--poll-interval` | float | `5.0` | `COMPRESSION_POLL_INTERVAL` | `--watch` 轮询间隔（秒） |
| `--wait-timeout` | float | `1800` | `COMPRESSION_WAIT_TIMEOUT` | `--watch` 最长等待（秒），超时退出码 7 |
| `--watch` / `--no-watch` | flag | `--no-watch` | `COMPRESSION_WATCH=1` | 是否轮询到终态 |
| `--json` | flag | 关闭 | — | JSON 输出（等价于 `--output-format json`） |
| `--output-format` | `table\|json` | `table` | `COMPRESSION_OUTPUT_FORMAT` | 输出格式 |
| `--dry-run` | flag | 关闭 | `COMPRESSION_DRY_RUN=1` | 仅打印解析后的请求体（含模型 ID 解析），不提交 |
| `--set KEY=VALUE` | 可重复 | — | `COMPRESSION_PARAMS`（JSON 对象） | 透传/覆盖任意方法参数（平台新参数无需改 CLI） |

```bash
# 环境变量驱动的典型集成（网站后端不需要拼命令行参数）
export COMPRESSION_API=http://127.0.0.1:8001
export COMPRESSION_TOKEN=$SERVICE_TOKEN
export COMPRESSION_PARAMS='{"eval":{"type":"perplexity","dataset_id":"<uuid>"}}'
scripts/compression quantize --model Qwen2.5-0.5B --method nncf_int4 --watch --json
```


---

## 3. 子命令参数表

### 3.1 `quantize`

**方法选择**（`--method`，必填）：

| `--method` | 类型 | 必填 | 默认值 | 说明 | 平台方法 / 典型产物 |
|---|---|---|---|---|---|
| `onnx_dynamic_int8` | str | 是（六选一） | — | ONNX 动态 INT8（仅权重、无需校准数据） | `quantize_onnx` → `model_quantized.onnx` |
| `onnx_static_int8` | str | 是 | — | ONNX 静态 INT8 QDQ（需校准图片，精度更高） | `quantize_onnx_static` → `model_quantized.onnx` |
| `openvino_int8` | str | 是 | — | OpenVINO IR + NNCF 后训练 INT8（CPU 首选） | `openvino_int8` → `model.xml`/`model.bin` |
| `nncf_int4` | str | 是 | — | LLM INT4 分组量化 → OpenVINO IR | `llm_quantize` → IR + `engine_nncf.txt` |
| `gguf` | str | 是 | — | HF LLM → GGUF（llama.cpp / ollama 生态） | `export_gguf` → `model.gguf` |
| `qat_int8` | str | 是 | — | 量化感知训练（FX QDQ 或 LSQ）→ 全 INT8 ONNX | `qat` → `model_qat.onnx` |

**参数**：

| 参数 | 类型 | 必填 | 默认值 | 环境变量 | 说明（适用方法） |
|---|---|---|---|---|---|
| `--model` | str | **是** | — | `COMPRESSION_MODEL` | 模型 ID / 名称 / 名称前缀 / UUID |
| `--method` | str | **是** | — | `COMPRESSION_METHOD` | 见上表 |
| `--opset` | int | 否 | `12` | `COMPRESSION_OPSET` | ONNX opset（`onnx_dynamic_int8`） |
| `--image-size` | int | 否 | `224` | `COMPRESSION_IMAGE_SIZE` | 校准图片尺寸（`onnx_static_int8`、`qat_int8`） |
| `--input-shape` | str | 否 | `1x3x224x224` | `COMPRESSION_INPUT_SHAPE` | 形如 `1x3x32x32`（静态 ONNX、OpenVINO、QAT） |
| `--calibration-samples` | int | 否 | `16` / `8` | `COMPRESSION_CALIBRATION_SAMPLES` | 校准样本数（ONNX 默认 16，OpenVINO 默认 8） |
| `--calib-dataset` | str | 否 | — | `COMPRESSION_CALIB_DATASET` | 平台数据集（名/ID）作为校准集；缺省时用随机数据 |
| `--precision` | str | 否 | `int8` | `COMPRESSION_PRECISION` | OpenVINO IR 精度：`int8` / `fp16` |
| `--model-kind` | str | 否 | `cnn` | `COMPRESSION_MODEL_KIND` | `cnn` / `yolo` |
| `--imgsz` | int | 否 | `640` | `COMPRESSION_IMGSZ` | YOLO 输入尺寸 |
| `--bits` | int | 否 | `4` | `COMPRESSION_BITS` | LLM 权重量化位数：`4` / `8`（`nncf_int4`） |
| `--group-size` | int | 否 | `128` | `COMPRESSION_GROUP_SIZE` | INT4 分组：32/64/128/256 |
| `--sym` | bool | 否 | `true` | `COMPRESSION_SYM` | 对称量化 |
| `--engine` | str | 否 | `nncf` | `COMPRESSION_ENGINE` | `nncf` / `optimum` / `auto` |
| `--outtype` | str | 否 | `f16` | `COMPRESSION_OUTTYPE` | GGUF 类型：`f16` `bf16` `q8_0` `q4_0` `q4_K` `q4_K_S` `q4_K_M` `q6_K` |
| `--scheme` | str | 否 | `fx` | `COMPRESSION_SCHEME` | QAT 方案：`fx`（QDQ 图）/ `lsq`（可学习步长） |
| `--lsq-bits` | int | 否 | `8` | `COMPRESSION_LSQ_BITS` | LSQ 权重位宽 |
| `--epochs` | int | 否 | `3` | `COMPRESSION_EPOCHS` | QAT 训练轮数 |
| `--lr` | float | 否 | `1e-3` | `COMPRESSION_LR` | QAT 学习率 |
| `--dataset` | str | 否 | — | `COMPRESSION_DATASET` | QAT 训练数据（按类图片目录数据集） |

```bash
compression quantize --model CifarCNN --method onnx_dynamic_int8 --json
compression quantize --model CifarCNN --method openvino_int8 --precision int8 \
    --calib-dataset CIFAR10-test --calibration-samples 32 --watch
compression quantize --model Qwen2.5-0.5B --method nncf_int4 --bits 4 --group-size 128 --watch
compression quantize --model Qwen2.5-0.5B --method gguf --outtype q4_K_M --watch

### 3.2 `prune`

| `--method` | 类型 | 必填 | 默认值 | 说明 | 平台参数 |
|---|---|---|---|---|---|
| `l1` | str | 是（七选一） | — | 结构化剪枝，L1 滤波器范数 | `prune`, `algorithm=l1` |
| `l2` | str | 是 | — | 结构化剪枝，L2 范数（平台默认） | `prune`, `algorithm=l2` |
| `bnslim` | str | 是 | — | BN-Slim：重要性 = \|BN γ\| | `prune`, `algorithm=bnslim` |
| `autoslim` | str | 是 | — | 稀疏训练 + γ 剪枝（需数据） | `prune`, `algorithm=autoslim` |
| `group_fisher` | str | 是 | — | Fisher 信息量（需数据，实测推荐） | `prune`, `algorithm=group_fisher` |
| `dmcp` | str | 是 | — | DMCP 一阶 Taylor（需数据） | `prune`, `algorithm=dmcp` |
| `sparsify` | str | 是 | — | 非结构化幅度剪枝 | `sparsify` |

| 参数 | 类型 | 必填 | 默认值 | 环境变量 | 说明 |
|---|---|---|---|---|---|
| `--model` | str | **是** | — | `COMPRESSION_MODEL` | PyTorch 模型（需与其同目录的 `arch.py`） |
| `--method` | str | **是** | — | `COMPRESSION_METHOD` | 见上表 |
| `--rate` | float | 否 | `0.3` | `COMPRESSION_RATE` | 剪枝率（0<rate<1） |
| `--dataset` | str | 数据驱动方法**必填** | — | `COMPRESSION_DATASET` | 按类图片目录数据集（`group_fisher`/`dmcp`/`autoslim`） |
| `--fisher-image-size` | int | 否 | `224` | `COMPRESSION_FISHER_IMAGE_SIZE` | Fisher 估计图片尺寸 |
| `--fisher-iters` | int | 否 | `12` | `COMPRESSION_FISHER_ITERS` | Fisher 迭代次数 |
| `--fisher-batch-size` | int | 否 | `8` | `COMPRESSION_FISHER_BATCH_SIZE` | Fisher batch |
| `--autoslim-epochs` | int | 否 | `2` | `COMPRESSION_AUTOSLIM_EPOCHS` | AutoSlim 稀疏训练轮数 |
| `--autoslim-l1` | float | 否 | `1e-2` | `COMPRESSION_AUTOSLIM_L1` | AutoSlim L1 正则强度 |
| `--autoslim-lr` | float | 否 | `3e-3` | `COMPRESSION_AUTOSLIM_LR` | AutoSlim 学习率 |
| `--strategy` | str | 否 | `uniform` | `COMPRESSION_STRATEGY` | `sparsify` 策略：`uniform` / `global` |
| `--input-shape` | str | 否 | `1x3x224x224` | `COMPRESSION_INPUT_SHAPE` | 前向自检 / ONNX 导出输入 |
| `--opset` | int | 否 | `12` | `COMPRESSION_OPSET` | 剪枝后 ONNX opset |
| `--export-onnx` | bool | 否 | `true` | `COMPRESSION_EXPORT_ONNX` | 是否同时导出 ONNX |

> 结果指标补充：`head_layers_adjusted`（自动对齐分类头层数）、`forward_check`（前向自检）、
> `topology_warning`（分支拓扑告警：true 表示该结构超出顺序 CNN 剪枝范围）。

```bash
compression prune --model CifarCNN --method l2 --rate 0.3 --input-shape 1x3x32x32 --watch
compression prune --model CifarCNN --method group_fisher --rate 0.3 \
    --dataset CIFAR10-train --input-shape 1x3x32x32 --watch --json
compression prune --model CifarCNN --method sparsify --rate 0.5 --strategy global
```

### 3.3 `distill`

| `--method` | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `logits` | str | 是（二选一） | — | Hinton 软标签 KD：`L=(1-α)·CE(y,p_s)+α·T²·KL(p_t/T‖p_s/T)` |
| `features` | str | 是 | — | FitNets 特征对齐（CNN 教师/学生，附加 MSE 投影项） |

| 参数 | 类型 | 必填 | 默认值 | 环境变量 | 说明 |
|---|---|---|---|---|---|
| `--model` / `--teacher` | str | **是** | — | `COMPRESSION_MODEL` | **教师模型**（PyTorch checkpoint + 同目录 `arch.py`） |
| `--method` | str | **是** | — | `COMPRESSION_METHOD` | `logits` / `features` |
| `--dataset` | str | **是** | — | `COMPRESSION_DATASET` | 按类图片目录数据集 |
| `--temperature` | float | 否 | `4.0` | `COMPRESSION_TEMPERATURE` | 蒸馏温度 T |
| `--alpha` | float | 否 | `0.7` | `COMPRESSION_ALPHA` | KD 权重 α∈[0,1] |
| `--width-mult` | float | 否 | `0.5` | `COMPRESSION_WIDTH_MULT` | 学生宽度系数（学生取自 `arch.py`） |
| `--epochs` | int | 否 | `5` | `COMPRESSION_EPOCHS` | 训练轮数 |
| `--lr` | float | 否 | `1e-3` | `COMPRESSION_LR` | 学习率 |
| `--batch-size` | int | 否 | `8` | `COMPRESSION_BATCH_SIZE` | batch |
| `--image-size` | int | 否 | `224` | `COMPRESSION_IMAGE_SIZE` | 图片尺寸 |
| `--input-shape` | str | 否 | 由 image-size 推导 | `COMPRESSION_INPUT_SHAPE` | 显式输入形状 |
| `--student-arch` | str | 否 | — | `COMPRESSION_STUDENT_ARCH` | 说明性参数：平台固定读取教师目录下的 `arch.py` |

```bash
compression distill --teacher CNNTeacher --method features --width-mult 0.5 \
    --dataset CIFAR10-train --epochs 5 --temperature 4 --alpha 0.7 --watch
compression distill --teacher CNNTeacher --method logits --dataset CIFAR10-train --json
```

compression quantize --model CifarCNN --method qat_int8 --scheme lsq --dataset CIFAR10-train --epochs 3
```


### 3.4 `finetune`

| `--method` | 类型 | 必填 | 默认值 | 说明 | 平台参数 |
|---|---|---|---|---|---|
| `lora` | str | 是（三选一） | — | LoRA SFT（指令对 → adapter + merged） | `llm_finetune`, `stage=sft` |
| `qlora` | str | 是 | — | `lora` 别名（INT4 基座属 GPU 二期） | `llm_finetune`, `stage=sft` |
| `dpo` | str | 是 | — | DPO 偏好优化（`L=-logσ(β·Δ)`） | `llm_finetune`, `stage=dpo` |

| 参数 | 类型 | 必填 | 默认值 | 环境变量 | 说明 |
|---|---|---|---|---|---|
| `--model` | str | **是** | — | `COMPRESSION_MODEL` | 待微调模型（HF 模型需 CUDA；PyTorch 走 CPU micro 原型） |
| `--method` | str | **是** | — | `COMPRESSION_METHOD` | `lora` / `qlora` / `dpo` |
| `--dataset` | str | 三选一 | — | `COMPRESSION_DATASET` | 平台数据集（`.jsonl/.csv/.xlsx/.txt`） |
| `--data` | str | 三选一 | — | `COMPRESSION_DATA` | 内联文本（每行 `问题 => 回答`） |
| `--model-repo` | str | 三选一 | — | `COMPRESSION_MODEL_REPO` | HuggingFace 数据集 ID |
| `--epochs` | int | 否 | `3`（SFT）/ `1`（DPO） | `COMPRESSION_EPOCHS` | 训练轮数 |
| `--lr` | float | 否 | `3e-3`（SFT）/ `5e-5`（DPO） | `COMPRESSION_LR` | 学习率 |
| `--lora-rank` | int | 否 | `8` | `COMPRESSION_LORA_RANK` | LoRA rank |
| `--hf-max-len` | int | 否 | `128` | `COMPRESSION_HF_MAX_LEN` | HF 分词最大长度 |
| `--dpo-beta` | float | 否 | `0.1` | `COMPRESSION_DPO_BETA` | DPO β（仅 `dpo`） |
| `--context` | int | 否 | `128` | `COMPRESSION_CONTEXT` | CPU micro-LM 上下文窗口（超长样本自动裁剪） |
| `--max-new` | int | 否 | `32` | `COMPRESSION_MAX_NEW` | 训练后采样长度 |

```bash
compression finetune --model Qwen2.5-0.5B --method lora --dataset alpaca-train --epochs 1 --watch
compression finetune --model Qwen2.5-0.5B --method dpo --dataset hh-rlhf-train --dpo-beta 0.1 --json
compression finetune --model CifarCNN --method lora --data $'1+1=? => 2\n2+2=? => 4' --epochs 1
```

### 3.5 `export`

| `--format` | 类型 | 必填 | 默认值 | 说明 | 平台方法 / 产物 |
|---|---|---|---|---|---|
| `onnx` | str | 是（五选一） | — | PyTorch → ONNX（onnxsim 简化、权重内联） | `export_onnx` → `model.onnx` |
| `onnx_int8` | str | 是 | — | ONNX 动态 INT8 | `quantize_onnx` → `model_quantized.onnx` |
| `openvino` | str | 是 | — | OpenVINO IR（`--precision fp16|int8`） | `openvino_int8` → `model.xml/.bin` |
| `gguf` | str | 是 | — | HF LLM → GGUF | `export_gguf` → `model.gguf` |
| `deploy_zip` | str | 是 | — | **不提交任务**：下载已完成任务的部署包 | `/deploy-export` → `deploy_<task>.zip` |

| 参数 | 类型 | 必填 | 默认值 | 环境变量 | 说明 |
|---|---|---|---|---|---|
| `--model` | str | 除 `deploy_zip` 外**必填** | — | `COMPRESSION_MODEL` | 模型引用 |
| `--format` | str | **是** | — | `COMPRESSION_FORMAT` | 见上表 |
| `--input-shape` | str | 否 | `1x3x224x224` | `COMPRESSION_INPUT_SHAPE` | 输入形状 |
| `--opset` | int | 否 | `12` | `COMPRESSION_OPSET` | ONNX opset |
| `--dynamic-axes` | bool | 否 | `false` | `COMPRESSION_DYNAMIC_AXES` | 动态 batch 轴 |
| `--model-kind` | str | 否 | `cnn` | `COMPRESSION_MODEL_KIND` | `cnn` / `yolo` |
| `--imgsz` | int | 否 | `640` | `COMPRESSION_IMGSZ` | YOLO 尺寸 |
| `--precision` | str | 否 | `fp16` | `COMPRESSION_PRECISION` | OpenVINO 精度（导出默认 fp16） |
| `--outtype` | str | 否 | `f16` | `COMPRESSION_OUTTYPE` | GGUF 类型 |
| `--task` | str | `deploy_zip` **必填** | — | `COMPRESSION_TASK` | 已完成任务 UUID |
| `--out-dir` | str | 否 | `./exports` | `COMPRESSION_OUT_DIR` | 本地保存目录 |

```bash
compression export --model CifarCNN --format onnx --input-shape 1x3x32x32 --watch
compression export --model Qwen2.5-0.5B --format gguf --outtype q4_K_M --watch --json
compression export --format deploy_zip --task <task_uuid> --out-dir ./exports
```

### 3.6 `task`（异步生命周期）

| 子命令 | 参数 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|---|
| `task status` | `<task_id>` | str | **是** | — | 打印单任务状态/进度/指标/产物 URL |
| `task list` | `--status` | enum | 否 | 全部 | `pending` / `running` / `succeeded` / `failed` / `canceled` |
| | `--method` | str | 否 | — | 平台方法过滤（如 `llm_quantize`、`export_gguf`） |
| | `--limit` | int | 否 | `20` | 返回条数 |
| `task wait` | `<task_id>` | str | **是** | — | 阻塞直到终态（失败→exit 6，超时→exit 7） |
| `task download` | `<task_id>` `--out-dir` | str | 是 / 否 | `./exports` | 下载产物 zip（目录产物自动打包） |
| `task deploy` | `<task_id>` `--out-dir` | str | 是 / 否 | `./exports` | 下载部署包（Dockerfile + `serve.py` + `model_card.json` + `model/`） |
| `task delete` | `<task_id>` `--yes` | flag | 是 / 否 | 交互确认 | 删除任务（运行中先取消） |

```bash
compression task list --status succeeded --limit 10
compression task wait  <task_uuid> --poll-interval 3 --wait-timeout 900
compression task download <task_uuid> --out-dir ./exports
compression task deploy   <task_uuid> --out-dir ./exports
compression task delete   <task_uuid> --yes
```

### 3.7 `models` / `datasets` / `local-datasets`

| 子命令 | 参数 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|---|
| `models` | `--framework` | str | 否 | — | `pytorch` / `onnx` / `huggingface` / `tensorflow` |
| | `--limit` | int | 否 | `50` | 返回条数 |
| `datasets` | `--limit` | int | 否 | `100` | 已注册数据集（含 `source_type=local` 的 reference 数据集） |
| `local-datasets` | — | — | — | — | 扫描只读根 `/datasets-local`：类型/类别数/样本数/体积/是否已导入 |

```bash
compression models --framework pytorch
compression datasets
compression local-datasets
```

---

## 4. 退出码约定

| 退出码 | 含义 | 触发场景 | 典型处理 |
|---|---|---|---|
| **0** | 成功 | 提交成功（未 `--watch`）或任务 `succeeded`（`--watch`） | 继续，读取 `task_id` |
| **1** | 通用错误 | 未归类异常、响应结构异常 | 记录日志人工排查 |
| **2** | 用法/参数错误 | 参数非法、`--method` 不在枚举、缺必填项、引用歧义 | 修正调用参数（前端先校验） |
| **3** | 认证/权限错误 | 401/403；无 token 且无法登录 | 刷新 `COMPRESSION_TOKEN` |
| **4** | 未找到 | 模型/数据集/任务不存在（404） | 提示用户重新选择资产 |
| **5** | 服务端校验失败 | 422（参数组合不被接受，如数据驱动剪枝缺 `--dataset`） | 回显 API `detail` |
| **6** | 任务失败 | 终态 `failed`（`--watch`） | 读取 `error_message`；平台支持 `retry` |
| **7** | 等待超时 | `--watch` 超过 `--wait-timeout` | 改为 `task status` 异步查询 |
| **8** | 网络/传输错误 | 连不上 API、下载中断 | 重试（下载可重跑覆盖） |
| **9** | 本地文件错误 | 下载结果为空、输出目录不可写 | 检查磁盘/权限 |
| **130** | 用户中断 | `Ctrl+C` | — |

```bash
compression exit-codes   # 打印该契约（脚本可直接 `diff` 校验）
```


---

## 5. 输出格式

**表格（默认）**：适合人工核对；列宽自动适配中英文宽度。

```text
$ compression task status aecd3aa5-4d8f-40f4-8cf2-f49d4460321e
FIELD     VALUE
--------  ---------------------------------------------------------
task_id   aecd3aa5-4d8f-40f4-8cf2-f49d4460321e
model     CifarCNN10 (87c53cf6-d576-46f8-a8d4-a643c7301e4e)
method    prune
status    succeeded
progress  100.0%
artifact  available (see download_url)
size      372.4KB (input 383.0KB)

METRIC               VALUE
-------------------  -----
total_filters        224
pruned_filters       69
prune_algorithm      group_fisher
params_after         45915
forward_check        ok
head_layers_adjusted 1
onnx_exported        true

artifact zip : /api/v1/compressions/aecd3aa5-.../download
deploy bundle: /api/v1/compressions/aecd3aa5-.../deploy-export
```

**JSON（`--json` / `--output-format json` / `COMPRESSION_OUTPUT_FORMAT=json`）**：稳定的机器可读结构，供网站前端直接消费。

```json
{
  "task_id": "aecd3aa5-4d8f-40f4-8cf2-f49d4460321e",
  "model_id": "87c53cf6-d576-46f8-a8d4-a643c7301e4e",
  "model_name": "CifarCNN10",
  "method": "prune",
  "status": "succeeded",
  "progress": 100.0,
  "output_size_bytes": 381321,
  "input_size_bytes": 392228,
  "artifact_available": true,
  "download_url": "/api/v1/compressions/aecd3aa5-.../download",
  "deploy_export_url": "/api/v1/compressions/aecd3aa5-.../deploy-export",
  "metrics": { "total_filters": 224, "pruned_filters": 69, "forward_check": "ok" }
}
```

> `--json` 模式下诊断信息走 stderr 且被抑制，**stdout 只含一份 JSON**，可安全 `json.loads(stdout)`。

---

## 6. 与推理模块的对接

CLI 与 HTTP API 一一对应（推荐：**Web 请求用 API，批处理/运维用 CLI**）。

| 动作 | CLI | HTTP 等价 |
|---|---|---|
| 提交压缩任务 | `compression quantize --model X --method nncf_int4 --json` | `POST /api/v1/compressions/` `{"model_id":…,"method":"llm_quantize","params":{…}}` |
| 查询任务 | `compression task status <id> --json` | `GET /api/v1/compressions/{id}` |
| 进度历史 | — | `GET /api/v1/compressions/{id}/progress-history` |
| 等待完成 | `compression task wait <id>` | 轮询上式（前端任务中心已有进度条） |
| 产物下载 | `compression task download <id>` | `GET /api/v1/compressions/{id}/download` |
| 部署包 | `compression task deploy <id>` / `export --format deploy_zip` | `GET /api/v1/compressions/{id}/deploy-export` |
| 推理（CNN/ONNX/OV） | — | `POST /api/v1/inference/predict` `{"task_id":…,"inputs":[…]}` |
| 推理（LLM IR） | — | `POST /api/v1/inference/chat/completions`（OpenAI 兼容，支持 SSE 流式） |
| 重跑评估/基准 | — | `POST /api/v1/compressions/{id}/reevaluate` `{"kind":"eval"}` |
| 重试失败任务 | — | `POST /api/v1/compressions/{id}/retry` |

**推理模块推荐的「压缩前置」流程**：

```text
用户选择模型 + 目标压缩方法
  └─(异步) 提交任务 → 立即返回 task_id，前端展示进度（轮询 /progress-history）
        ├─ succeeded ─┬─ CNN/ONNX/OV : 直接用 /inference/predict（传 task_id，无需下载）
        │             └─ LLM IR      : 直接用 /inference/chat/completions（传 task_id）
        └─ failed    →  展示 error_message，按 CLI 退出码语义分类提示（6=任务失败 / 5=参数问题）
```

> 生产部署建议使用 `deploy_zip` 产物（`Dockerfile` + `serve.py` + `model_card.json` + `model/`），
> 与推理模块现有容器化流程一致；`model_card.json` 内含 `method/params/metrics`，可直接入库做版本追踪。

