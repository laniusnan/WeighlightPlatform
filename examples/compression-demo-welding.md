# Demo：焊接缺陷判定模型压缩（压缩率 62.6%，精度损失 0.06%）

> 交付判据：**压缩率 > 60%** 且 **精度损失 < 3%**
> 实测结论：压缩率 **62.60%**（对 `.pt`）/ **63.01%**（对厂商 ONNX），精度损失 **0.06 个百分点**，
> 推理延迟 **0.664 → 0.164 ms/样本（约 4x 提速）** → **达标 ✅**
> 相关文档：[CLI 参数说明](../docs/compression-job.md) · [实验报告](../docs/compression-report.md)

---

## 1. 被测对象

来自交付包 `welding-transformer-package-20260622T0620Z.zip`：

| 项 | 值 |
|----|----|
| 模型 | `WELD-QC-TRANSFORMER-20260622`（Tabular Transformer Encoder） |
| 结构 | 7 个数值特征 → 5 分类；`d_model=64`、`nhead=4`、`num_layers=3`、`ffn=160`、dropout=0.12 |
| 参数量 | 118,053 |
| 权重 | `model/welding_transformer.pt`（`{format, config, metrics, model_state_dict, optimizer}`）492,041 B |
| 同包其它格式 | `safetensors`、`state_dict.pth`、`onnx`（+ 外置 `.onnx.data`，合计 497,551 B）、`config.json`、`metrics.json` |
| 训练数据 | `data/training/ds_41631c12a3_welding_training.csv`（60,000 行，7 特征 + `缺陷类型`） |
| 厂商基线 | 测试准确率 **0.935641**、loss 0.176976（12 epoch，分层切分 74/13/13，seed 20260622） |
| 模型定义 | `scripts/welding_transformer.py`（含 `WeldingTabularTransformer` 与 `build_model(config)`） |

标签：`合格 / 气孔 / 裂纹 / 未熔合 / 沉陷`。

**为什么选量化而不是剪枝**：该模型是纯 Transformer（无 `Conv2d`），平台结构化剪枝仅支持 CNN
（实测报 `No Conv2d layers found, pruning is not applicable`）；表格/注意力模型的标准做法是
**权重 INT8 动态量化**：无需校准数据、精度损失极小、CPU 推理明显提速。

---

## 2. 准备

```bash
# ① 解包（宿主机任意目录）
unzip welding-transformer-package-20260622T0620Z.zip -d /data/welding
PKG=/data/welding/welding-transformer-package-20260622T0620Z

# ② 构建压缩作业镜像（CPU；约 5-15 分钟，一次即可）
cd <repo>
docker build -f deploy/compress/Dockerfile.cpu -t compress-job:cpu .
# 已有平台镜像时可用秒级构建：
# docker build -f deploy/compress/Dockerfile.fast -t compress-job:cpu --build-arg BASE_IMAGE=backend:cpu-ship .

# ③ 统一挂载（下面命令复用）
VIEW="-v $PKG:/pkg:ro -v /data/welding-out:/out -v $PWD:/workspace:ro"
```

---

## 3. 步骤 1：PyTorch → FP32 ONNX（基线产物）

```bash
docker run --rm $VIEW compress-job:cpu \
  --job-id=weld-01 --tenant-id=welding --trace-id=demo-1 \
  --base-model-path=/pkg/model/welding_transformer.pt \
  --arch-file=/pkg/scripts/welding_transformer.py \
  --output-dir=/out/w01-fp32-onnx \
  --method=export --algo=onnx --input-shape=1x7 --model-kind=tabular --opset=17
```

关键点：

* `--arch-file`：厂商 `.pt` 只存 `model_state_dict`，脚本把模型与 `welding_transformer.py`
  暂存到 `<output>/_stage/`，并用其中的 **`build_model(config)`** 重建网络（平台已支持该格式）。
* `--input-shape=1x7`：表格模型输入是 `[batch, 7]`，**不是**默认的图片 `1x3x224x224`。

实测输出（日志摘要）：

```text
[1/1] export:onnx → export_onnx     耗时 4.4s
artifact: /out/w01-fp32-onnx/artifact/model.onnx   507,726 B (495.8 KB)
metrics : model_kind=tabular
```

---

## 4. 步骤 2：FP32 → INT8（权重动态量化）

```bash
docker run --rm $VIEW compress-job:cpu \
  --job-id=weld-02 --tenant-id=welding --trace-id=demo-2 \
  --base-model-path=/out/w01-fp32-onnx/artifact/model.onnx \
  --output-dir=/out/w02-int8-onnx \
  --method=quantize --algo=onnx_dynamic_int8
```

实测输出：

```text
[1/1] quantize:onnx_dynamic_int8 → quantize_onnx   耗时 2.7s
artifact: /out/w02-int8-onnx/artifact/model_quantized.onnx   184,041 B (179.7 KB)
metrics : quant_type=dynamic_int8
```

体积对比：FP32 ONNX 507,726 B → INT8 ONNX 184,041 B（**下降 63.75%**）；
相对厂商 `.pt`（492,041 B）下降 **62.60%**；相对厂商 ONNX 合计（497,551 B）下降 **63.01%**。

---

## 5. 步骤 3：精度与性能验证（验收）

用 `deploy/compress/examples/welding_eval.py` 在**与训练完全一致**的测试集上对比基线与压缩产物：

```bash
# 注意：镜像 ENTRYPOINT 是 compress_job.py，跑评测脚本要显式换 entrypoint
docker run --rm --entrypoint python $VIEW compress-job:cpu \
  /app/examples/welding_eval.py \
  --csv  /pkg/data/training/ds_41631c12a3_welding_training.csv \
  --config /pkg/model/welding_transformer_config.json \
  --arch /pkg/scripts/welding_transformer.py \
  --baseline /pkg/model/welding_transformer.pt \
  --candidate /out/w02-int8-onnx/artifact/model_quantized.onnx \
  --out /out/welding-eval.json --latency-samples 2000
```

> 用 `Dockerfile.fast` 构建的镜像里示例在 `/app/examples/`；用源码目录时也可直接挂载仓库后跑
> `/workspace/deploy/compress/examples/welding_eval.py`（两种路径等价）。

评测脚本做的事：

1. 用标准库读 CSV，按训练脚本同样的 `train_test_split(test_size=0.26, stratify=y, random_state=20260622)`
   再对半切分，得到 **7800 条测试样本**（与厂商 `metrics.json` 的切分一致）；
2. 按 `config.json` 的 `mean/std` 做 z-score 归一化；
3. 分别用 torch（`.pt`）与 onnxruntime（INT8 `.onnx`）逐批推理，计算准确率、loss、
   per-class P/R/F1、混淆矩阵、单样本延迟；
4. 直接输出「压缩率 / 精度损失 / 是否达标」结论文本。

实测结果：

```text
测试集: 7800 条（seed=20260622，分层切分，与训练一致）
--------------------------------------------------------------------------
model                             size   accuracy      loss    ms/sample
--------------------------------------------------------------------------
model_quantized.onnx           179.7KB     0.9350    0.1793        0.164
welding_transformer.pt         480.5KB     0.9356    0.1778        0.664
--------------------------------------------------------------------------
压缩率（体积下降）: 62.60%
精度变化          : -0.06 个百分点（绝对损失 0.06%）
loss 变化         : +0.0015
延迟变化          : -0.501 ms/样本
交付判据（压缩率>60% 且 精度损失<3%）: 通过
--------------------------------------------------------------------------
```

要点：**基线复算准确率 0.9356 = 厂商报告值 0.935641**，说明切分/归一化/推理链路完全对齐；
INT8 后仅少量样本预测翻转（0.9356 → 0.9350），延迟降到约 1/4。

原始评测 JSON：[`deploy/compress/examples/results/welding-eval.json`](../deploy/compress/examples/results/welding-eval.json)
（含 per-class P/R/F1 与混淆矩阵）。

---

## 6. 验收结论

| 指标 | 基线（`.pt`） | 压缩后（INT8 ONNX） | 判据 | 结果 |
|------|--------------|--------------------|------|------|
| 体积 | 480.5 KB (492,041 B) | **179.7 KB (184,041 B)** | 压缩率 > 60% | **62.60%** ✅ |
| 测试准确率 | 0.935641 | **0.9350** | 精度损失 < 3% | **0.06%** ✅ |
| 测试 loss | 0.1778 | 0.1793 | — | +0.0015 |
| 延迟（CPU） | 0.664 ms/样本 | **0.164 ms/样本** | — | **≈4x 提速** |
| 部署形态 | 需 PyTorch + `arch.py` + config | 单文件 ONNX，onnxruntime 直接加载 | — | 更易上产线 |

> 产线用法：加载 `model_quantized.onnx`，输入 `[N,7]`（7 个传感器特征，z-score 归一化），
> 输出 `[N,5]` logits，`argmax` 得到 `合格/气孔/裂纹/未熔合/沉陷`。

---

## 7. 常见追问

**Q1：为什么精度损失不是 0？**
INT8 动态量化把 MatMul/Gemm 权重压到 INT8（per-channel，激活动态量化），4/7800 个样本 argmax 翻转，
在 <3% 预算内属正常。若要进一步压损失，可用 `--algo=onnx_static_int8` 配合真实校准数据（QDQ 静态量化）。

**Q2：能不能压得更小？**
按收益/风险排序：
1. `--method=quantize --algo=onnx_int4`（INT4 权重仅量化，体积再降约 50% 量级）——**实验性**：
   需额外依赖 `onnx-ir`，会连带升级 `protobuf`，与 `transformers` 的 `protobuf<7` 冲突；
2. `--method=quantize --algo=nncf_int4`（OpenVINO INT4 IR）——面向 HF LLM；
3. 该模型无 `Conv2d`，无法用结构化剪枝继续瘦身。

**Q3：动态 batch 可行吗？**
`--dynamic-axes` 导出的图在 onnxruntime **动态量化**时会因 attention 的 Reshape 结构报
`input_shape_size == requested_shape_size was false`（本仓库保留了失败样例日志）。
生产建议：本模型批量小、延迟已 <0.2 ms，直接用静态 batch=1 的 INT8 产物；
需要变批量时用 **OpenVINO INT8/FP16 IR**（`--algo=openvino_int8` / `--precision=fp16`）。

**Q4：如何接入自动化调度？**

```bash
# 一条命令完成「导出 + 量化」
docker run --rm $VIEW compress-job:cpu \
  --job-id={{job.id}} --tenant-id={{job.tenant_id}} --trace-id={{job.trace_id}} \
  --base-model-path={{inputs.base_model_path}} \
  --arch-file=/pkg/scripts/welding_transformer.py \
  --output-dir={{inputs.output_path}} \
  --method=pipeline --pipeline="export:onnx,quantize:onnx_dynamic_int8" \
  --input-shape=1x7 --model-kind=tabular --opset=17
```

作业结束后读 `{{inputs.output_path}}/result.json`（`artifact_bytes`、`metrics`）与
`{{inputs.output_path}}/artifact/step2/model_quantized.onnx`。

---

## 8. 复现清单

| 文件 | 作用 |
|------|------|
| `deploy/compress/examples/welding_demo.sh` | 本 demo 一键脚本（导出 → 量化 → 评测 → 结论 → 输出目录/进度摘要） |
| `deploy/compress/examples/welding_eval.py` | 精度/延迟评测（复现训练切分） |
| `deploy/compress/examples/results/welding-eval.json` | 本次评测原始结果 |
| `deploy/compress/examples/results/*/result.json` | 两个作业的完整 `result.json` |
| `compress_job.py` | 压缩作业 CLI |
| `docs/compression-job.md` | 参数说明（§5.0 输出目录规则、§5.2 日志与进度、§5.3 版本核对） |
| `docs/compression-report.md` | 实验报告 |

---

## 9. 输出目录、日志与进度（交付方对接用）

### 9.1 输出目录参数只有一个名字：`--output-dir`

所有产物、报告、日志、进度都写在这个目录下（无 `--out/--outdir/--output-path` 别名；
调度器模板 `--output-dir={{inputs.output_path}}`；环境变量 `COMPRESS_JOB_OUTPUT_DIR`）：

```
/out/w02-int8-onnx/
├── artifact/model_quantized.onnx   # 压缩产物
├── result.json                     # 作业总报告（含 inputs 清单、metrics、steps）
├── metrics.json                    # 纯指标
├── logs.jsonl                      # 结构化日志（与 stdout 同内容）
├── progress.json                   # 进度快照（最新状态，原子替换）
├── progress.log                    # 进度里程碑（每 10% 一行，可 tail -f）
└── _stage/                         # 暂存副本（本 demo 为导出产物，跑完可删）
```

`welding_demo.sh` 最后会自动打印两份摘要（步骤 1/步骤 2 各自的目录内容、`progress.json`、
`progress.log` 末几行、`result.json` 关键字段），样例：

```
==> 步骤 2（quantize:onnx_dynamic_int8）输出目录：/data/welding-out/w02-int8-onnx
    _stage
    artifact
    logs.jsonl
    metrics.json
    progress.json
    progress.log
    result.json
  progress.json : status=succeeded percent=100% step=1/1 elapsed=0.4s
                  artifact=/out/w02-int8-onnx/artifact/model_quantized.onnx
  progress.log  : 11 行（档位由 --progress-interval 决定，默认每 10% 一行）
      2026-09-18T19:12:07+0800  80% step=1/1 quantize:onnx_dynamic_int8 → quantize_onnx Done.
      2026-09-18T19:12:07+0800  90% step=1/1 quantize:onnx_dynamic_int8 → quantize_onnx Done.
      2026-09-18T19:12:07+0800 100% step=1/1 quantize:onnx_dynamic_int8 → quantize_onnx Done.
  result.json   : status=succeeded method=quantize/onnx_dynamic_int8 artifact_bytes=184041
  metrics       : {"quant_type": "dynamic_int8", "preprocess": "standard"}
```

### 9.2 日志：stdout 和文件都写

| 通道 | 说明 |
|------|------|
| stdout | 每条日志实时 `print`，默认 JSON Lines（与 docker logs 采集兼容）；`--no-json-logs` 转人类可读 |
| `<output-dir>/logs.jsonl` | **同一份日志逐条追加**，容器日志被截断也能事后回放 |

事件序列：`start → inputs → progress_file →（staging）→ step_start → progress / progress_milestone → step_done →（eval_start/eval_done）→ done`。

### 9.3 进度：每 10% 一次，失败也落盘

| 文件/事件 | 语义 |
|------|------|
| `progress.log` | 每跨过 `--progress-interval`%（默认 10）写一行；引擎只报 30/70/95 时自动补齐 10/20/…，保证 `0..100` 不跳档 |
| `progress.json` | 最新快照：`status/step/steps_total/percent/message/elapsed_s/artifact`；作业结束 `succeeded`，失败/中断为 `failed/interrupted` + 原因 |
| `progress_milestone` 日志 | 每个档位同时进 stdout 与 `logs.jsonl`，便于调度器抓取 |
| `--no-progress-file` | 完全关闭进度文件 |
| `--progress-interval 25` | 改为每 25% 一次 |

### 9.4 交付包版本核对（同名包有新旧两版时必看）

```bash
# 代码包内（zip 解压后）
cat BUILD_INFO.txt                        # 构建时间 / git / compress_job.py 与 compression.py 的 sha256
sha256sum -c SHA256SUMS                   # 逐文件校验
grep -c _prepare_quant_input backend/app/jobs/compression.py   # ≥1 = 含 ONNX 自动降级链（2026-09-22 起）
python scripts/check_compress_package.py  # 布局/引擎/CLI（--output-dir 规范性）/进度文件自检
# zip 自身的校验值在其同级 <zip>.sha256
```
> 只替换 `compress_job.py` **不生效**：形状推断/标注修复在引擎 `backend/app/jobs/compression.py` 里，请整包更新。

