# 模型压缩作业 CLI 使用说明（compress_job.py）

> 适用版本：压缩作业入口 `compress_job.py`（仓库根目录 / 镜像内 `/app/compress_job.py`）
> 关联文档：[`compression-cli.md`](./compression-cli.md)（平台 CLI，走 API 队列）、
> [`compression-env.md`](./compression-env.md)（环境与依赖）、
> [`../examples/compression-demo-welding.md`](../examples/compression-demo-welding.md)（焊接模型实测 demo）、
> [`compression-report.md`](./compression-report.md)（实验报告）

---

## 1. 这是什么

`compress_job.py` 是**单文件压缩作业入口**：调度器把「作业上下文 + 输入输出路径 + 超参」
当作命令行参数传进来，脚本在**本进程内**完成压缩，把产物与报告写入 `--output-dir`。

与平台的关系：

| 方式 | 入口 | 依赖 | 适用场景 |
|------|------|------|----------|
| **作业 CLI（本文）** | `python /app/compress_job.py --job-id=… --base-model-path=…` | 无（不需要 DB/Redis/Web） | 调度器旁挂作业、离线单机、交付给同事 |
| 平台 CLI | `python compression.py quantize …` | 平台 API + DB + Redis worker | 平台 Web 排队、看板、多用户 |
| HTTP 服务 | `compress_job.py --serve` | fastapi/uvicorn（镜像已含） | 以服务方式提供压缩能力 |
| 平台 API 代理 | `examples/inference_module/api/compression_proxy.py` | 平台 API | 推理服务前置压缩 |

数据流：

```
 模型文件/模型目录 ─┐
 数据集/校准集 ────┼─► compress_job.py ─► <output-dir>/artifact/<压缩产物>
 作业参数 ────────┘                     ├─ result.json   （作业结果：metrics/耗时/产物）
                                        ├─ metrics.json  （纯指标）
                                        └─ logs.jsonl    （结构化日志，带 job/tenant/trace）
```

---

## 2. 三种调用方式

```bash
# ① 容器（推荐交付方式；镜像见 deploy/compress/）
docker run --rm \
  -v /data/models:/models -v /data/datasets:/datasets -v /data/out:/out \
  compress-job:cpu \
    --job-id=1001 --tenant-id=acme --trace-id=7f3c \
    --base-model-path=/models/welding_transformer.pt \
    --output-dir=/out/1001 \
    --method=quantize --algo=onnx_dynamic_int8

# ② 镜像内 / 服务器上的裸 python
python /app/compress_job.py --help

# ③ 仓库源码直接跑（自动把 backend/ 加入 sys.path）
python compress_job.py --dry-run --method=prune --algo=l2 --rate=0.3 \
  --base-model-path=/models/cnn.pt --output-dir=/tmp/out
```

### 与训练作业模板的对齐

把 `train.py` 的写法原样替换即可（`{{inputs.*}}` 对应 `--base-model-path/--dataset-path/--output-dir`）：

```bash
python /app/compress_job.py \
  --job-id={{job.id}} --tenant-id={{job.tenant_id}} --trace-id={{job.trace_id}} \
  --base-model-path={{inputs.base_model_path}} \
  --dataset-path={{inputs.dataset_path}} \
  --output-dir={{inputs.output_path}} \
  --method=quantize --algo=onnx_dynamic_int8 \
  --epochs=80 --batch-size=16 --learning-rate=0.001   # 该步用不到的超参也会记入 result.json
```

---

## 3. 参数总表

### 3.1 作业上下文（调度器注入）

| 参数 | 默认 | 说明 |
|------|------|------|
| `--job-id` | `local`（或 `COMPRESS_JOB_JOB_ID`） | 作业 ID，写入 `result.json` 与每条日志 |
| `--tenant-id` | `local` | 租户 ID，日志透传，便于多租户采集/计费 |
| `--trace-id` | 空 | 链路追踪 ID，日志透传，与调度平台 trace 对齐 |

### 3.2 输入 / 输出

| 参数 | 必填 | 说明 |
|------|------|------|
| `--base-model-path` | 是 | 待压缩模型：`.pt/.pth` 检查点、`.onnx`、HF 模型目录（`config.json` 所在目录）；配 `--model-dir` 时可只给文件名 |
| `--model-dir` | 否 | 模型目录：`--base-model-path`/`--arch-file`/`--external-data-path` 给相对路径时的基准目录 |
| `--external-data-path` | 外部权重时**必填** | ONNX 外部权重地址：给 `.onnx.data` 文件，或给它所在目录。**不默认同目录**，缺失即退出码 3（详见 §5.1） |
| `--allow-implicit-external-data` | 否 | 兼容开关：显式允许「与 `.onnx` 同目录」的隐式解析（默认关） |
| `--dataset-path` | 视方法 | 按类图片目录（`classA/*.jpg`）、`.jsonl/.csv/.xlsx` 指令集、含 `.txt` 的语料目录、`.txt` 单文件 |
| `--eval-dataset-path` | 否 | 评估/校准集；缺省复用 `--dataset-path`（分类评测/困惑度评测输入） |
| `--output-dir` | `outputs` | 输出根目录：产物写 `artifact/`，报告写 `result.json/metrics.json/logs.jsonl` |
| `--arch-file` | 否 | 模型定义 `.py`（含模型类或 `build_model(config)` 工厂）。`--model-dir`/模型同目录没有 `arch.py` 时，脚本把模型与 arch 暂存到 `<output-dir>/_stage/` 后再加载 |
| `--llama-cpp-path` | 否 | 离线 GGUF 转换工具目录（含 `convert_hf_to_gguf.py`），等价环境变量 `LLAMA_CPP_PATH`；缺省时引擎会尝试联网拉取 |
| `--hf-cache-dir` | 否 | HuggingFace/ModelScope 下载缓存目录（导出为 `HF_HOME`/`HF_HUB_CACHE`），用于离线复用已下载模型 |

> 检查点支持：`torch.save(model)` 完整模块；`{"model"/"net"/"module"/"model_ema"/"state_dict"/"model_state_dict": …}`；
> `{"config": {…}, "model_state_dict": …}` + `--arch-file`（通过 arch 中的 `build_model(config)` 重建，第三方交付包常见）。

### 3.3 方法 / 算法

| 参数 | 说明 |
|------|------|
| `--method` | `quantize` · `prune` · `distill` · `finetune` · `export` · `pipeline` |
| `--algo` | 族内算法（见下表）；`--method=pipeline` 时不使用 |
| `--pipeline` | 多步串联：`"prune:group_fisher,quantize:onnx_dynamic_int8"`；前一步产物自动作为下一步输入 |

**方法 × 算法矩阵（`--algo` → 平台方法）**

| `--method` | `--algo` 取值 | 平台方法 | 数据要求 | 产物 |
|------------|---------------|----------|----------|------|
| `quantize` | `onnx_dynamic_int8` | `quantize_onnx` | 无 | `model_quantized.onnx`（权重 INT8） |
| `quantize` | `onnx_int4` / `onnx_int4_wq` | `quantize_onnx_int4` | 无 | `model_int4.onnx`（INT4/MatMulNBits，**实验性**，见 §9） |
| `quantize` | `onnx_static_int8` | `quantize_onnx_static` | 校准图片目录 | QDQ INT8 ONNX |
| `quantize` | `openvino_int8` | `openvino_int8` | 校准图片目录 | OpenVINO IR（`model.xml`+`model.bin`） |
| `quantize` | `nncf_int4` | `llm_quantize` | 无 | INT4 OpenVINO IR（LLM 权重压缩） |
| `quantize` | `gguf` | `export_gguf` | 无 | `*.gguf`（llama.cpp 用） |
| `prune` | `l1`·`l2`·`bnslim`·`autoslim`·`group_fisher`·`dmcp` | `prune` | 结构化剪枝需按类图片（后三者必须有数据） | `model_pruned.pt` + ONNX（**仅含 Conv2d 的 CNN**） |
| `prune` | `sparsify` | `sparsify` | 无 | 稀疏化检查点（非结构化，不减小体积） |
| `distill` | `logits`·`features` | `distill` | 按类图片目录 | `student.pt`（+ONNX） |
| `finetune` | `lora`·`qlora` | `llm_finetune`（stage=sft） | 指令数据（目录/jsonl/csv/xlsx 或 `--data`） | `adapter_lora.pt`、`model_lora_merged.pt` |
| `finetune` | `dpo` | `llm_finetune`（stage=dpo） | 偏好数据（`prompt`+`chosen`/`rejected`） | `adapter_lora.pt` + DPO 指标 |
| `export` | `onnx` | `export_onnx` | 无（建议给 `--input-shape`） | `model.onnx` |
| `export` | `onnx_int8` | `quantize_onnx` | 无 | INT8 ONNX |
| `export` | `openvino` | `openvino_int8`+`--precision=fp16` | 无 | FP16 OpenVINO IR |
| `export` | `gguf` | `export_gguf` | 无 | GGUF（f16/q8_0/q4_K_M…） |
| `pipeline` | — | 依次调度上述步骤 | 见各步 | `artifact/step1 … stepN` |

> 不支持的组合会在执行前被拦下（退出码 **4**）并列出该 `--method` 下可用的 `--algo`。

### 3.4 通用超参

| 参数 | 平台字段 | 作用 / 取值建议 | 生效方法 |
|------|----------|-----------------|----------|
| `--epochs` | `epochs` | 训练轮数 ≥1 | QAT、蒸馏、微调 |
| `--batch-size` | `batch_size` | 批大小 | QAT、蒸馏、微调 |
| `--learning-rate` | `lr` | 学习率；蒸馏常用 `1e-3`，LoRA 常用 `2e-4~3e-4` | QAT、蒸馏、微调 |
| `--rate` | `rate` | 剪枝率 `0<rate<1` | `prune`、`sparsify` |
| `--bits` | `bits` | 权重量化位宽 `4`/`8` | `nncf_int4`、`onnx_int4` |
| `--group-size` | `group_size`/`block_size` | 量化分组/块大小 `32`·`64`·`128`·`256`；越小精度越高、体积略大 | `nncf_int4`、`onnx_int4` |
| `--precision` | `precision` | OpenVINO 精度 `int8`·`fp16` | `openvino_int8`、`export:openvino` |
| `--outtype` | `outtype` | GGUF 类型：`f16`·`q8_0`·`q4_0`·`q4_K_M`·`q5_K_M`·`q6_K`… | `*:gguf` |
| `--temperature` | `temperature` | 蒸馏温度 T（2~6 常用） | `distill` |
| `--alpha` | `alpha` | KD 软标签权重 α（0~1） | `distill` |
| `--width-mult` | `width_mult` | 学生网络宽度系数（0.5=半宽） | `distill` |
| `--lora-rank` | `lora_rank` | LoRA 秩 8/16/32/64 | `finetune` |
| `--dpo-beta` | `dpo_beta` | DPO β，常用 0.1 | `finetune:dpo` |
| `--hf-max-len` | `hf_max_len` | HF 分词最大长度（显存/速度权衡） | `finetune`（HF） |
| `--max-samples` | `calibration_samples` | 校准样本上限（越多越准越慢） | `onnx_static_int8`、`openvino_int8` |
| `--input-shape` | `input_shape` | 输入形状：`1x7`、`1,3,32,32`、`[1,3,224,224]`；决定导出图输入尺寸 | `export*`、`quantize*`、`prune` |
| `--image-size` | `image_size`（并自动设 `input_shape=[1,3,S,S]`） | 图片边长，同时决定评估尺寸 | 分类系全部方法 |
| `--opset` | `opset` | ONNX opset（默认 12；Transformer 建议 17） | `export*`、`quantize*` |
| `--model-kind` | `model_kind` | `cnn`·`nlp`·`tabular`·`yolo`；YOLO 走 ultralytics 分支 | 导出、评估 |
| `--imgsz` | `imgsz` | YOLO 导出输入尺寸（默认 640） | `export:onnx`（yolo） |
| `--dynamic-axes` / `--no-dynamic-axes` | `dynamic_axes` | ONNX 动态 batch 轴（服务端变批量） | `export:onnx` |
| `--seed` | — | 随机种子（python/numpy/torch 同时设） | 全部（复现性） |
| `--device` | — | `cpu`/`cuda` 偏好（信息性；实际由镜像与 `CUDA_VISIBLE_DEVICES` 决定） | 全部 |
| `--stage` | `stage` | 微调阶段 `sft`/`dpo`（默认按 algo 推断） | `finetune` |
| `--data` | `data` | 内联训练数据，每行 `问题 => 回答` | `finetune`（冒烟） |

### 3.5 执行控制

| 参数 | 默认 | 说明 |
|------|------|------|
| `--mode` | `local` | `local`=进程内执行；`api`=提交到平台 API 并轮询 |
| `--api-base` | `COMPRESSION_API_BASE` / `http://127.0.0.1:8001` | API 模式平台地址（自动补 `/api/v1`） |
| `--api-token` | `COMPRESSION_TOKEN` | Bearer token；缺省用 `COMPRESSION_EMAIL/COMPRESSION_PASSWORD` 登录换取 |
| `--platform-model-id` | — | API 模式下的平台模型 ID/名称（本机路径不会上传） |
| `--timeout` | `0`（不限） | 软超时（秒），超时退出码 **6** |
| `--progress-file` / `--no-progress-file` | 开 | 在 `--output-dir` 下写 `progress.json`（快照）+ `progress.log`（每档一行） |
| `--progress-interval` | `10` | 进度落盘/记录粒度（百分比）：每跨过该档位写一条里程碑 |
| `--json-logs` / `--no-json-logs` | 开 | **仅影响 stdout 格式**（JSON Lines ↔ 人类可读）；`logs.jsonl` 始终是 JSON Lines |
| `--dry-run` | 关 | 只解析并打印「**inputs 输入清单**（参数名/来源 cli·env·default/解析结果/是否必填）→ CLI 步骤 → 平台方法 → 参数」，不执行；输入有问题时输出 `input_error` 并返回对应退出码 |
| `--serve` | 关 | HTTP 服务：`POST /compress`、`GET /healthz`、`GET /jobs/{id}` |
| `--host` / `--port` | `0.0.0.0` / `8080` | `--serve` 监听地址与端口 |

### 3.6 环境变量

所有 CLI 参数都有 `COMPRESS_JOB_` 前缀的等价环境变量（**命令行优先**）：

```bash
COMPRESS_JOB_METHOD=quantize COMPRESS_JOB_ALGO=onnx_dynamic_int8 \
COMPRESS_JOB_BASE_MODEL_PATH=/models/x.onnx COMPRESS_JOB_OUTPUT_DIR=/out/1 \
python /app/compress_job.py --job-id=1
```

| 变量 | 作用 |
|------|------|
| `COMPRESS_JOB_<参数名大写>` | 覆盖任意参数默认值（`COMPRESS_JOB_EPOCHS=2`、`COMPRESS_JOB_JSON_LOGS=0`、`COMPRESS_JOB_MODEL_DIR=/models`、`COMPRESS_JOB_EXTERNAL_DATA_PATH=/models/x.onnx.data`、`COMPRESS_JOB_ALLOW_IMPLICIT_EXTERNAL_DATA=1`、`COMPRESS_JOB_PROGRESS_INTERVAL=25`…） |
| `COMPRESSION_API_BASE` / `COMPRESSION_TOKEN` / `COMPRESSION_EMAIL` / `COMPRESSION_PASSWORD` | API 模式连接平台 |
| `MODEL_STORAGE_PATH` | 平台存储根；本地模式会自动指向输出目录 |
| `LLAMA_CPP_PATH` | 含 `convert_hf_to_gguf.py` 的目录 → **离线 GGUF**（命令行可用 `--llama-cpp-path` 覆盖） |
| `HF_HOME` | HF/ModelScope 缓存目录（命令行可用 `--hf-cache-dir` 覆盖） |
| `CUDA_VISIBLE_DEVICES` | GPU 作业选卡（编排注入） |

> 「哪些输入来自命令行 / 环境变量 / 代码默认」不用猜：`--dry-run` 与每次作业的 `result.json.inputs`
> 都会逐项标出 `source`（`cli`/`env`/`default`）、`required`、`value` 与 `resolved`。

---

## 4. 退出码

| 码 | 含义 | 典型原因 |
|----|------|----------|
| `0` | 成功 | — |
| `2` | 参数错误 | 缺 `--method`/`--algo`、`--outtype` 非法、pipeline 语法错误 |
| `3` | 输入缺失/不可写 | 模型或数据集路径不存在、缺少必填项 |
| `4` | 方法或算法不支持 | 如 `--method=prune --algo=foo` |
| `5` | 压缩执行失败 | 平台作业函数抛错（模型结构不符、数据格式不符…） |
| `6` | 超时 | `--timeout` 到期 |
| `7` | 缺少可选依赖 | 例如未装 LLM 扩展、`onnx_int4` 缺 `onnx-ir` |
| `130` | 用户中断 | Ctrl-C / SIGINT |

---

## 5. 产物与报告格式

### 5.0 输出目录规则（硬约定）

**输出目录参数只有一个名字：`--output-dir`**（环境变量 `COMPRESS_JOB_OUTPUT_DIR`；
调度器模板 `--output-dir={{inputs.output_path}}`；没有 `--out/--outdir/--output-path` 之类的别名）。
所有产物、报告、日志、进度都写在它下面；每次作业的摘要末尾会列出「输出目录 + 已写出文件」：

```
<output-dir>/
├── artifact/       压缩产物（`model.onnx` / `model_quantized.onnx` / OpenVINO IR 目录 / `*.gguf` …）
├── result.json     作业总报告（含 `inputs` 清单、`metrics`、`steps`、`artifact_bytes`）
├── metrics.json    指标
├── logs.jsonl      结构化日志（与 stdout 同内容，逐条 JSON Lines）
├── progress.json   进度快照（`status/step/steps_total/percent/message/elapsed_s/artifact`）
├── progress.log    进度里程碑（**每 10% 一行**，可直接 `tail -f`）
└── _stage/         暂存副本（内联后的单文件 ONNX、`arch.py` …），跑完可删
```

```
<output-dir>/
├── artifact/                     # 压缩产物（文件名由方法决定）
│   └── step1/ step2/ …           # --method=pipeline 时按步骤分目录
├── result.json                   # 作业级结果（下游读取这个）
├── metrics.json                  # 纯指标（与平台 task.metrics 同结构）
├── logs.jsonl                    # 结构化日志（逐行 JSON）
├── _stage/                       # 需要时暂存的模型 + arch.py
└── _eval/                        # 评估工作目录（临时）
```

`result.json` 关键字段：

| 字段 | 说明 |
|------|------|
| `status` | `succeeded` / 失败时由退出码体现 |
| `job_id` / `tenant_id` / `trace_id` | 与命令行一致，便于对账 |
| `method` / `algo` | 请求的方法（pipeline 时为 `pipeline`） |
| `base_model_path` / `dataset_path` / `output_dir` | 实际使用的路径 |
| `artifact` / `artifact_resolved` | 产物路径 / 目录产物解析后的可加载文件 |
| `artifact_bytes` | 产物字节数（**压缩率对照用**） |
| `duration_s` | 墙钟耗时（秒） |
| `inputs` | **输入清单**：每项含 `name`（命令行参数）、`source`（`cli`/`env`/`default`）、`required`、`value`、`resolved`（解析后路径）、`note`（缺省约定的说明） |
| `metrics` | 方法指标（`pruned_filters`、`quant_type`、`trainable_params`…）与 `eval` 评测结果 |
| `steps[]` | 每步的平台方法、参数、产物、指标（pipeline 时逐个列出） |

`logs.jsonl` 事件：`start` →（`staging`/`data`）→ `step_start` → `progress` → `step_done` →（`eval_start`/`eval_done`）→ `done`；
出错为 `error`，超时 `error(exit_code=6)`。每条都带 `job_id/tenant_id/trace_id/ts`。

### 5.1 ONNX 外部权重（external data，即 `.onnx` + `.onnx.data`）

`torch>=2.9` 的新导出器（`dynamo=True`）会把大权重写到旁挂文件，得到
`<name>.onnx`（小）+ `<name>.onnx.data`（大）一对。**地址必须显式给出，不默认同目录**：

```bash
# 给文件
python /app/compress_job.py --base-model-path /m/model.onnx \
  --external-data-path /m/model.onnx.data --method=quantize --algo=onnx_dynamic_int8 ...
# 给目录（等价写法）
python /app/compress_job.py --base-model-path /m/model.onnx \
  --external-data-path /m                --method=quantize --algo=onnx_dynamic_int8 ...
# 相对路径 + --model-dir（三者都以该目录为基准）
python /app/compress_job.py --model-dir /m --base-model-path model.onnx \
  --external-data-path model.onnx.data   --method=quantize --algo=onnx_dynamic_int8 ...
```

行为规则与实测（6 组，全部在本机/镜像内跑过）：

| 场景 | 命令（关键参数） | 结果 |
|------|------------------|------|
| 不给 `--external-data-path` | `--base-model-path m/model.onnx` | ❌ 退出码 **3**：`该 ONNX 使用外部权重（external data），但未显式给出权重地址：请加 --external-data-path <文件或目录>（图内声明的 location: model.onnx.data）。同目录的 m/model.onnx.data 不会被默认采用；…` |
| 给目录 | `--external-data-path m/` | ✅ 权重内联为单文件后量化，产物 `model_quantized.onnx` **184,065 B**（单文件，无旁挂） |
| 给文件 | `--external-data-path m/model.onnx.data` | ✅ 同上 184,065 B |
| 显式兼容开关 | `--allow-implicit-external-data` | ✅ 按「同目录」旧行为解析（184,065 B），日志会打 warning |
| 相对路径 + `--model-dir` | `--model-dir m --base-model-path model.onnx --external-data-path model.onnx.data` | ✅ 184,065 B（三者的相对路径都相对 `--model-dir`） |
| 地址指错（目录里没有该 `.data`） | `--external-data-path other_dir/` | ❌ 退出码 **3**：`无法从 other_dir 载入外部权重（ValidationError: Data of TensorProto (tensor name: …) should be stored in …/model.onnx.data, but it is not regular file.）` |

实现细节：

* 探测方式：解析 `.onnx` 的 `TensorProto.external_data[location]`（`load_external_data=False`，不读权重）；
  文件 >64 MB 直接视为权重内联，跳过解析。
* 给了地址后，用 `onnx.external_data_helper.load_external_data_for_model(proto, 目录)` 载入，再
  `save_model(..., save_as_external_data=False)` **内联成单文件**暂存到 `<output-dir>/_stage/`，
  所以后续引擎步骤与产物都不再依赖旁挂文件。
* 产品侧导出（`--method=export`）用旧导出器（`dynamo=False`）+ `onnxsim`，**不会产生** `.data`。
* 平台上传：传 `.zip`（含两个文件），服务端同样内联成单文件 `original.onnx`。

### 5.2 日志与进度（stdout 还是文件？两个都写）

| 通道 | 内容 | 开关 |
|------|------|------|
| **stdout** | 每条日志实时 `print`：默认 JSON Lines（与文件同内容），`--no-json-logs` 转人类可读；作业摘要表在 `--json-logs` 时打到 stderr、否则 stdout | `--json-logs/--no-json-logs` |
| `<output-dir>/logs.jsonl` | **同一份日志逐条追加**（始终 JSON Lines），容器日志被截断也可事后完整排查 | 始终开 |
| `<output-dir>/progress.json` | 最新进度快照，原子替换写入：`status/step/steps_total/step_label/percent/message/started_at/updated_at/elapsed_s/artifact` | `--progress-file/--no-progress-file` |
| `<output-dir>/progress.log` | 里程碑追加行，**每 `--progress-interval`%（默认 10）一行**，含真实进度消息，可 `tail -f` | 同上 |

进度语义：

* 每个步骤开始写 `0%`，引擎回调（导出 30/70/95、量化 10/50/95 …）会**补齐档位**，
  所以 `progress.log` 里每个步骤都必然出现 `0/10/20/…/100`，不会跳档；
* 步骤成功 → 100% + 下一步从 0% 重新计数；整个作业结束 → `progress.json.status = succeeded`；
* 失败/中断 → `status: failed/interrupted` 且带原因（**连参数校验失败也会落盘**），不会停在 `running`；
* `--progress-interval 25` 可改成每 25% 一次（粒度由调度器需要决定）。

`logs.jsonl` 事件序列：`start` → `inputs` → `progress_file` →（`staging`）→ `step_start` →
`progress` / `progress_milestone` → `step_done` →（`eval_start`/`eval_done`）→ `done`；失败为 `error`。

### 5.3 交付包版本核对（同名新旧包必看）

交付的 `compress-code-<日期>.zip` 里带 **`BUILD_INFO.txt`**（构建时间、git 分支/提交/是否 dirty、
`compress_job.py` 与 `backend/app/jobs/compression.py` 等关键文件的 sha256）+ `SHA256SUMS`；
zip 自身在同目录有 `<zip>.sha256`。核对方法：

```bash
cat BUILD_INFO.txt
sha256sum -c SHA256SUMS
grep -c _prepare_quant_input backend/app/jobs/compression.py   # ≥1 → 含 ONNX 自动降级链（2026-09-22 起）
python scripts/check_compress_package.py    # 布局/引擎/CLI（含 --output-dir 规范性）/进度文件自检
```

> 注意：**只替换 `compress_job.py` 不生效** —— `IndexError`/形状标注修复在引擎
> `backend/app/jobs/compression.py`，请整包（或同时替换两个文件）更新。

---

## 6. 各方法典型命令（可直接复制）

### 6.1 量化（最常用，无需数据）

```bash
# ① PyTorch → ONNX（表格/Transformer 用 --input-shape + --model-kind）
python /app/compress_job.py --job-id=101 --trace-id=t-101 \
  --base-model-path=/models/welding_transformer.pt \
  --arch-file=/models/welding_transformer_arch.py \
  --output-dir=/out/101 --method=export --algo=onnx \
  --input-shape=1x7 --model-kind=tabular --opset=17

# ② ONNX → 权重 INT8（动态量化，推理提速 ~2-4x）
python /app/compress_job.py --job-id=102 --trace-id=t-102 \
  --base-model-path=/out/101/artifact/model.onnx \
  --output-dir=/out/102 --method=quantize --algo=onnx_dynamic_int8

# ③ CNN：静态 INT8（需要校准图片）
python /app/compress_job.py --job-id=103 --base-model-path=/models/resnet.pt \
  --dataset-path=/datasets/ILSVRC-calib --output-dir=/out/103 \
  --method=quantize --algo=onnx_static_int8 --image-size=224 --max-samples=128

# ④ OpenVINO IR（Intel CPU/核显推理）
python /app/compress_job.py --job-id=104 --base-model-path=/models/resnet.onnx \
  --dataset-path=/datasets/ILSVRC-calib --output-dir=/out/104 \
  --method=quantize --algo=openvino_int8 --precision=int8 --max-samples=128

# ⑤ LLM INT4（OpenVINO IR，Qwen/Llama 等 HF 目录）
python /app/compress_job.py --job-id=105 --base-model-path=/models/Qwen2.5-0.5B \
  --output-dir=/out/105 --method=quantize --algo=nncf_int4 --bits=4 --group-size=128

# ⑥ GGUF（llama.cpp 生态）
python /app/compress_job.py --job-id=106 --base-model-path=/models/Qwen2.5-0.5B \
  --output-dir=/out/106 --method=quantize --algo=gguf --outtype=q4_K_M
```

### 6.2 剪枝（结构化，仅 CNN）

```bash
# 数据无关：l1/l2/bnslim（按权重或 BN 缩放因子排序剪 filter）
python /app/compress_job.py --job-id=201 --base-model-path=/models/cnn.pt \
  --output-dir=/out/201 --method=prune --algo=l2 --rate=0.3 --image-size=32

# 数据驱动：group_fisher / autoslim / dmcp（需要按类图片目录）
python /app/compress_job.py --job-id=202 --base-model-path=/models/cnn.pt \
  --dataset-path=/datasets/CIFAR10-train --eval-dataset-path=/datasets/CIFAR10-test \
  --output-dir=/out/202 --method=prune --algo=group_fisher --rate=0.3 --image-size=32
```

> 前向自检、head 维度对齐、剪枝后 ONNX 导出均由平台作业函数处理；
> **不含 Conv2d 的模型（纯 Transformer/MLP）会明确报错**：`No Conv2d layers found, pruning is not applicable`。

### 6.3 蒸馏（小模型学生）

```bash
python /app/compress_job.py --job-id=301 --base-model-path=/models/teacher.pt \
  --dataset-path=/datasets/CIFAR10-train --eval-dataset-path=/datasets/CIFAR10-test \
  --output-dir=/out/301 --method=distill --algo=logits \
  --width-mult=0.5 --temperature=4 --alpha=0.7 --epochs=3 --batch-size=64 --learning-rate=0.001
```

### 6.4 微调（LoRA / QLoRA / DPO）

```bash
# 指令数据 SFT（GPU：HF 因果 LM；CPU：内置 micro-LM 原型，仅流程冒烟）
python /app/compress_job.py --job-id=401 --base-model-path=/models/Qwen2.5-0.5B \
  --dataset-path=/datasets/alpaca-train/alpaca.jsonl \
  --output-dir=/out/401 --method=finetune --algo=lora \
  --epochs=1 --batch-size=2 --learning-rate=0.0003 --lora-rank=16 --hf-max-len=512

# 偏好数据 DPO
python /app/compress_job.py --job-id=402 --base-model-path=/models/Qwen2.5-0.5B \
  --dataset-path=/datasets/hh-rlhf-test/preferences.jsonl \
  --output-dir=/out/402 --method=finetune --algo=dpo --epochs=1 --learning-rate=0.0001 --dpo-beta=0.1

# 内联数据冒烟（无需数据集目录）
python /app/compress_job.py --job-id=403 --base-model-path=/models/cnn.pt \
  --output-dir=/out/403 --method=finetune --algo=lora --epochs=1 \
  --data="焊接电流异常怎么办 => 检查送丝速度与保护气流量"
```

### 6.5 导出

```bash
python /app/compress_job.py --job-id=501 --base-model-path=/models/cnn.pt \
  --output-dir=/out/501 --method=export --algo=onnx --input-shape=1,3,32,32 --opset=17
python /app/compress_job.py --job-id=502 --base-model-path=/out/501/artifact/model.onnx \
  --output-dir=/out/502 --method=export --algo=openvino --precision=fp16
python /app/compress_job.py --job-id=503 --base-model-path=/models/Qwen2.5-0.5B \
  --output-dir=/out/503 --method=export --algo=gguf --outtype=f16
```

### 6.6 流水线（多步串联）

```bash
# 剪枝 → 量化：第一步的 ONNX 自动喂给第二步
python /app/compress_job.py --job-id=601 --base-model-path=/models/cnn.pt \
  --dataset-path=/datasets/CIFAR10-train --output-dir=/out/601 \
  --method=pipeline --pipeline="prune:group_fisher,quantize:onnx_dynamic_int8" \
  --rate=0.3 --image-size=32
```

串联规则：每步产物经 `resolve_artifact_file()` 解析（目录产物取其中的 `model.xml`/`model_quantized.onnx`/`model_pruned.pt`…）
再作为下一步输入；产物分落 `artifact/step1`、`artifact/step2`…，`result.json.steps[]` 记录每步明细。

---

## 7. 服务模式（可选）

```bash
docker run --rm -p 8080:8080 -v /data/models:/models -v /data/out:/out \
  compress-job:cpu --serve --host=0.0.0.0 --port=8080

# 调用（参数与 CLI 同名，支持 snake_case）
curl -s -X POST http://localhost:8080/compress -H 'Content-Type: application/json' -d '{
  "job_id": "svc-1", "tenant_id": "acme", "trace_id": "abc",
  "args": {"base_model_path": "/models/x.onnx", "output_dir": "/out/svc-1",
           "method": "quantize", "algo": "onnx_dynamic_int8"}
}'

# 健康检查（探测 torch/openvino/nncf/transformers/cuda 能力）
curl -s http://localhost:8080/healthz
# 查询作业结果
curl -s http://localhost:8080/jobs/svc-1
```

---

## 8. 常见错误 → 处理

| 现象 / 报错 | 退出码 | 原因与处理 |
|-------------|--------|-----------|
| `--method 必填`、`--method=prune 需要同时指定 --algo` | 2 | 补参数；先用 `--dry-run` 验证 |
| `不支持的方法组合: prune:foo` | 4 | 按报错中列出的可选 algo 修正 |
| `模型路径不存在` / `数据集路径不存在` | 3 | 容器内路径与挂载点不一致（`-v 宿主:容器`），或相对路径基准不是宿主 cwd |
| `No Conv2d layers found, pruning is not applicable` | 5 | 该模型是 Transformer/MLP，改用 `quantize`/`distill` |
| `quantize_onnx requires an ONNX model` | 5 | 先 `--method=export --algo=onnx`，或用 `--method=pipeline` 串联 |
| `Cannot load checkpoint …` / `contains a state_dict but no matching architecture` | 5 | 加 `--arch-file`（模型类或 `build_model(config)` 工厂） |
| `Got invalid dimensions for input … Expected: 1` | 5 | ONNX 为静态 batch：按该批量喂入，或导出时加 `--dynamic-axes` |
| `input_shape_size == requested_shape_size was false`（`gemm_input_reshape`） | 5 | onnxruntime 动态量化 + 动态 batch attention 图的已知限制：改静态 batch 导出再量化，或走 OpenVINO INT8 |
| `No module named 'onnx_ir'`（`--algo=onnx_int4`） | 7 | INT4 需 `onnx-ir`；安装会连带升级 `protobuf/numpy`（与 `transformers` 的 `protobuf<7` 冲突），默认不预装 |
| `llm_finetune on HuggingFace models requires a CUDA runtime` | 5 | HF 微调需 GPU 镜像 `compress-job:gpu`；CPU 只支持 PyTorch 检查点的 micro 原型 |
| GGUF 长时间停在 `Fetching llama.cpp conversion tooling` | — | 无外网：把 `convert_hf_to_gguf.py` 放入 `deploy/compress/llama-cpp/` 后重建镜像（`LLAMA_CPP_PATH`），见该目录 README |
| 评估尺寸不匹配（224 vs 32） | — | 脚本按产物静态输入尺寸自动对齐，日志中会出现 `eval_align` 提示；显式 `--image-size` 冲突时以产物为准 |
| `No module named 'compression._common'; 'compression' is not a package` | 5 | 从**仓库根目录**直接跑脚本时，同目录的启动器 `compression.py` 遮蔽了标准库 `compression`（`torch → gzip`）。脚本已自动把该目录移出 `sys.path`（`_drop_shadowing_self_dir`），若仍复现请更新到最新 `compress_job.py`；镜像内无此文件，不会触发 |
| `找不到 app/jobs/compression.py` | 3 | 压缩**引擎实现体**没随包交付：把 `compress_job.py` 放在仓库根目录、`backend/` 内，或使用镜像（`/app/backend/app/...`）；代码包请先跑 `scripts/check_compress_package.py` 自检 |
| `ValidationError: Data of TensorProto ( tensor name: …) should be stored in /path/model.onnx.data, but it is not regular file` | 5 / 3 | ONNX 外部权重没读到。CLI 现在要求显式给地址：`--external-data-path <文件或目录>`（相对路径按 `--model-dir` 解析）；给错地址时会以退出码 3 报「无法从 … 载入外部权重」。旧行为需显式加 `--allow-implicit-external-data`。平台上传请用 zip（服务端自动内联成单文件） |
| `IndexError: list index out of range`（日志紧邻 `Running shape inference...`，栈底在 `onnxruntime/tools/symbolic_shape_infer.py` 的 `_infer_Concat`） | 5 | 该 `.onnx` 的图（多为旧导出器产物）过不了 ORT 的**符号形状推断**。引擎已内置降级链：默认预处理 → `skip_symbolic_shape=True` → 不做预处理，日志会写 `预处理完成（skip_symbolic_shape）`；若仍失败，改用 `--method=export --algo=onnx` 从 `.pt` 重新导出（推荐），或 `--algo=openvino_int8` |
| `[ShapeInferenceError] Inferred shape and existing shape differ in dimension 0: (64) vs (5)` | 5 | 图内 shape 标注自相矛盾。引擎会**自动剔除 `graph.value_info`（冗余形状标注）后重试**，并在 `result.json.metrics` 里记 `shape_repair: stripped_N_value_info`；实测厂商 welding ONNX 由此成功量化（166,134 B / 精度损失 0.06%） |
| `input_shape_size == requested_shape_size was false`（`gemm_input_reshape` / `node_view_9`） | 5 | 该 ONNX 是**静态 batch=1** 的图（内部 `Reshape` 写死单条）。评测/基准已自动降批逐条推理并打 warn；若需要 batch>1 推理，请从 `.pt` 重新导出（`--method=export --algo=onnx`，必要时 `--dynamic-axes`） |

---

## 9. 能力边界（实测结论）

| 能力 | 状态 | 说明 |
|------|------|------|
| ONNX INT8 动态量化 | ✅ 稳定 | CNN / Transformer / 表格模型通用，无需数据 |
| ONNX / OpenVINO 静态 INT8 | ✅ | 需校准图片目录，精度通常更优 |
| 结构化剪枝 | ✅（限 CNN） | 需要 Conv2d；纯 Transformer/MLP 不支持 |
| 知识蒸馏 | ✅ | 需按类图片目录（teacher 与学生同输入空间） |
| LoRA / QLoRA / DPO | GPU ✅ / CPU 原型 | CPU 仅 micro-LM 原型（流程验证） |
| GGUF 导出 | ✅（需网络或离线工具） | 见 §8 |
| INT4 权重仅量化（`onnx_int4`） | ⚠️ 实验性 | 需 `onnx-ir`，与 `transformers` 依赖冲突，默认未启用 |
| 动态 batch + 动态 INT8 | ⚠️ 受限 | onnxruntime 动态量化与动态 batch attention 图的已知限制（§8） |
| 第三方/旧导出的 ONNX（`IndexError` 形状推断、`Inferred 64 vs 5` 标注冲突） | ✅ 自动降级 | 引擎依次尝试：默认预处理 → `skip_symbolic_shape=True` → 原始图；量化失败时自动剔除 `graph.value_info` 重试，`metrics.preprocess` / `metrics.shape_repair` 可追溯。实测厂商 welding ONNX：497,551 B → **166,134 B（66.24%）**，精度 0.9356→0.9350（损失 0.06%） |
| 静态 batch（含 batch=1）ONNX | ⚠️ 按固定批量喂入 | 评测/基准/推理会自动降批（必要时逐条）并打 warn；需要 batch>1 请重新导出（`--dynamic-axes`） |

---

## 10. 相关文件

| 文件 | 说明 |
|------|------|
| `compress_job.py` | 本 CLI（单文件作业入口） |
| `deploy/compress/Dockerfile.cpu` / `.gpu` / `.fast` | 作业镜像（独立构建 / CUDA / 复用平台镜像秒级构建） |
| `scripts/compress-image.sh` | 构建镜像 / `docker save` 环境包（`dist/compress-job-cpu-<日期>.tar.gz`）/ 冒烟自检 |
| `scripts/compress-bundle.sh` | 打**代码包** `--bundle-only` → `dist/compress-code-<日期>.zip`；不带参数 → **全量包** `dist/compress-full-<日期>.zip`（代码 + 镜像 tar + SHA256SUMS） |
| `scripts/check_compress_package.py` | **代码包自检**：布局 / 语法 / 引擎导入（量化剪枝等入口）/ CLI `--dry-run`，`--run model.onnx` 可加跑一次真实作业 |
| `deploy/compress/examples/welding_eval.py` | 表格分类模型的精度/延迟评测脚本（复现训练切分） |
| `deploy/compress/examples/welding_demo.sh` | 焊接模型压缩 demo 一键脚本 |
| `examples/compression-demo-welding.md` | 焊接模型压缩 demo（压缩率/精度损失实测） |
| `docs/compression-report.md` | 模型压缩实验报告 |
| `docs/compression-eval-report.md` | **服务器现有资产的压缩方法评估报告**（资产清单 + 11 任务实测 + 异常记录，数据源 `out/summary.json`） |
| `docs/method-cards.schema.json` | 「压缩方法」界面方法卡片的 JSON Schema（前端 `method-cards.json` 由其约束） |




