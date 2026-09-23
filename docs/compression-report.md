# 模型压缩实验报告

| 项目 | 内容 |
|------|------|
| 报告名称 | 模型压缩能力实验报告（Model Compression Report） |
| 实验对象 | 压缩作业 CLI `compress_job.py` + 焊接缺陷判定模型 `WELD-QC-TRANSFORMER-20260622` |
| 实验时间 | 2026-09-18 |
| 实验环境 | Ubuntu 22.04（WSL2）/ Python 3.11 · torch 2.9.1+cpu · onnx 1.22.0 · onnxruntime 1.22.0 · openvino 2026.3.1 · nncf 3.3.0 · transformers 4.49.x |
| 运行镜像 | `compress-job:cpu`（见 `deploy/compress/Dockerfile.cpu`） |
| 结论 | 焊接模型：**压缩率 62.60%**（>60%）、**精度损失 0.06%**（<3%）、延迟降至 0.25x → **验收通过** |

> 说明：仓库根目录 `exp.md` 在本报告生成时为 **0 字节（空文件）**，因此本文按「实验目的 → 环境 → 对象 →
> 方法 → 参数 → 结果 → 分析 → 结论 → 复现」的标准实验报告结构撰写；若需套用其它模板字段，可按需重排章节。

---

## 一、实验目的

1. 验证「推理前模型压缩」可作为**独立作业**被调度器以命令行方式直接调用
   （形如 `python /app/compress_job.py --job-id={{job.id}} … --output-dir={{inputs.output_path}}`）。
2. 用真实交付模型验证压缩效果：**压缩率 > 60% 且精度损失 < 3%**。
3. 产出可交付物：压缩镜像、CLI 与参数文档、demo、报告。

## 二、实验环境与依赖

| 类别 | 内容 |
|------|------|
| 硬件 | 本地工作站（CPU 推理）；另备 GPU 镜像用于 HF 微调 |
| 容器 | `compress-job:cpu`（python:3.11 + uv 同步 backend 运行依赖） |
| 关键依赖 | `torch 2.9.1+cpu`、`onnx 1.22.0`、`onnxruntime 1.22.0`、`onnxsim`、`openvino 2026.3.1`、`nncf 3.3.0`、`transformers 4.49.x`、`scikit-learn` |
| 运行方式 | `docker run` 单作业进程；无 DB / Redis / Web 依赖 |
| 数据 | `ds_41631c12a3_welding_training.csv`（60,000 行；7 特征 + 缺陷类型），分层切分 74/13/13（seed 20260622） |
| 评测 | `deploy/compress/examples/welding_eval.py`（复现训练切分 → 逐批推理 → 准确率/loss/延迟） |

## 三、实验对象

| 项 | 值 |
|----|----|
| 模型 | Tabular Transformer Encoder（7 特征 → 5 类：合格/气孔/裂纹/未熔合/沉陷） |
| 参数量 | 118,053 |
| 结构 | `d_model=64, nhead=4, num_layers=3, dim_feedforward=160, dropout=0.12`（`norm_first=True`，`batch_first=True`） |
| 交付格式 | `.pt`（`{format, config, metrics, model_state_dict, optimizer}`）、`state_dict.pth`、`safetensors`、`onnx` + 外置 `.onnx.data` |
| 基线指标（厂商 `metrics.json`） | 测试准确率 0.935641、loss 0.176976、12 epoch、batch 1024 |
| 模型定义 | `scripts/welding_transformer.py`（`WeldingTabularTransformer` + `build_model(config)`） |

## 四、实验设计

| 变量 | 设置 |
|------|------|
| 对照组 | 厂商 `.pt` 检查点直接推理（torch，CPU） |
| 实验组 1 | 导出 FP32 ONNX（`export:onnx`，`--input-shape=1x7 --opset=17`） |
| 实验组 2 | FP32 ONNX → **INT8 权重动态量化**（`quantize:onnx_dynamic_int8`，per-channel） |
| 尝试但不可用 | 结构化剪枝（模型无 `Conv2d`）；动态 batch + INT8（onnxruntime 量化限制） |
| 度量 | 产物字节数（压缩率）、测试集准确率与 loss（精度损失）、单样本延迟 |
| 判据 | 压缩率 ≥ 60% **且** |精度损失| < 3 个百分点 |

## 五、关键参数配置

| 作业 | 命令要点 | 说明 |
|------|----------|------|
| weld-01 | `--method=export --algo=onnx --input-shape=1x7 --model-kind=tabular --opset=17` | 表格模型输入 `[batch,7]`，必须显式给形状 |
| weld-02 | `--method=quantize --algo=onnx_dynamic_int8` | ONNX 权重 INT8（动态量化，无需校准集） |
| weld-05 | `--method=prune --algo=l1 --rate=0.3` | 结果：`No Conv2d layers found, pruning is not applicable` |
| 评测 | `welding_eval.py --seed 20260622 --batch-size 512 --latency-samples 2000` | 复现训练切分，统计准确率/loss/延迟 |

## 六、实验结果

### 6.1 体积

| 产物 | 字节数 | 相对 `.pt` | 相对厂商 ONNX（含 `.data`） |
|------|--------|-----------|----------------------------|
| `welding_transformer.pt`（基线） | 492,041 | — | — |
| FP32 ONNX（`model.onnx`） | 507,726 | +3.2%（含图与常量） | +2.0% |
| **INT8 ONNX（`model_quantized.onnx`）** | **184,041** | **−62.60%** | **−63.01%** |

### 6.2 精度与性能（测试集 7800 条，CPU）

| 模型 | 体积 | 准确率 | loss | 延迟 (ms/样本) |
|------|------|--------|------|----------------|
| `.pt`（基线） | 480.5 KB | **0.9356** | 0.1778 | 0.664 |
| INT8 ONNX | **179.7 KB** | **0.9350** | 0.1793 | **0.164** |
| 变化 | **−62.60%** | **−0.06 pp（0.06%）** | +0.0015 | **−75.6%** |

> 基线复算 0.9356 与厂商 `metrics.json` 的 0.935641 一致 → 切分/归一化/推理链路对齐。

### 6.3 逐类指标（INT8 ONNX，节选）

| 类别 | precision | recall | F1 | support |
|------|-----------|--------|----|---------|
| 合格 | 0.9714 | 0.9709 | 0.9711 | 6625 |
| 气孔 | 0.7297 | 0.7401 | 0.7349 | 881 |
| 裂纹 | 0.7675 | 0.7292 | 0.7479 | 240 |
| 未熔合 | 0.6444 | 0.6739 | 0.6587 | 46 |
| 沉陷 | 0.5714 | 1.0000 | 0.7273 | 8 |

（完整 P/R/F1、混淆矩阵与原始数值见 `deploy/compress/examples/results/welding-eval.json`。）

### 6.4 平台方法矩阵验证（累积实验记录）

| 方法 | 结果 | 备注 |
|------|------|------|
| `export:onnx`（CNN / 表格 Transformer） | ✅ | 含 onnxsim 简化；表格模型需 `--input-shape` |
| `quantize:onnx_dynamic_int8` | ✅ | CIFAR-10 CNN：372 → 101 KB；焊接模型：508 → 184 KB |
| `quantize:openvino_int8` / `export:openvino` | ✅ | IR 目录产物；分类评测（CIFAR-10 10,000 图）跑通 |
| `prune:l2` / `prune:group_fisher` / `prune:autoslim` | ✅（CNN） | `forward_check=ok`、head 维度自动对齐 |
| `distill:logits/features` | ✅ | 需按类图片目录 |
| `finetune:lora`（CPU micro 原型 / GPU HF） | ✅ | CPU 跑 micro-LM 原型；HF 需 CUDA |
| `quantize:nncf_int4`（LLM） | ✅ | Qwen2.5-0.5B，CPU 165 s 输出 INT4 IR |
| `export:gguf` | ✅（需网络/离线工具） | 受限网络环境下未完成时间测量 |
| `quantize:onnx_int4` | ⚠️ | 需 `onnx-ir`（与 `transformers` 的 `protobuf<7` 冲突），默认未启用 |
| `prune`（无 Conv2d 模型） | ✖（预期） | `No Conv2d layers found, pruning is not applicable` |
| 动态 batch + `onnx_dynamic_int8` | ✖ | `input_shape_size == requested_shape_size was false`（onnxruntime 限制） |

### 6.5 直接量化厂商导出的 ONNX（外部权重 + 引擎自动修图）

场景：不做 `export`，直接拿厂商的 `welding_transformer.onnx`（31,503 B）+
`welding_transformer.onnx.data`（466,048 B，外部权重），且两者**分处不同目录**（模拟编排把
模型与数据分别挂载 `--input/base_model_path/`、`--input/data_path/`）。

```bash
python compress_job.py --method=quantize --algo=onnx_dynamic_int8 \
  --base-model-path=/input/base_model_path/welding_transformer.onnx \
  --external-data-path=/input/data_path/welding_transformer.onnx.data \
  --output-dir=/out/1001
```

引擎自动完成的三个降级动作（日志可见，`result.json.metrics` 可追溯）：

| 步骤 | 现象 | 自动处理 |
|------|------|----------|
| 1 | `quant_pre_process` 默认符号形状推断抛 `IndexError`（`_infer_Concat`） | 改用 `skip_symbolic_shape=True` → 预处理成功（`metrics.preprocess = "skip_symbolic_shape"`） |
| 2 | `quantize_dynamic` 抛 `Inferred shape and existing shape differ in dimension 0: (64) vs (5)` | 剔除 `graph.value_info` 235 条冗余形状标注后重试（`metrics.shape_repair = "stripped_235_value_info"`） |
| 3 | 该图内部 `Reshape` 写死单条形状（batch>1 报 `input_shape_size == requested_shape_size`） | 评测/基准自动降批逐条推理并打 warn |

结果（同一测试集 7800 条，seed=20260622）：

| 模型 | 体积 | 准确率 | loss | 延迟 (ms/样本) |
|------|------|--------|------|----------------|
| `.pt`（基线） | 480.5 KB | **0.9356** | 0.1778 | 0.657 |
| 厂商 ONNX 直接量化（INT8） | **162.2 KB** | **0.9350** | 0.1793 | **0.188** |
| 变化 | **−66.24%** | **−0.06 pp（0.06%）** | +0.0015 | −71.4% |

→ 与「`.pt` 重新导出再量化」路径（184,041 B / −62.60%）结论一致：**压缩率 >60%、精度损失 ≈0.06%**，
两条路径都满足交付判据；直接量化厂商 ONNX 体积更小，但产物仍是 batch=1 图（沿用原图限制）。


## 七、结果分析

1. **量化是该模型的最优解**：INT8 把 118k 个 FP32 权重压到 1 字节/参数，理论下限约 75%；
   实测 62.60% 已满足 >60%（其余体积来自 LayerNorm/常量与图结构）。
2. **精度损失可忽略**：仅 4/7800 个样本 argmax 翻转（0.9356 → 0.9350），per-channel 量化 + 动态激活
   对表格特征扰动极小，<3% 预算余量充足。
3. **顺带获得推理收益**：延迟 0.664 → 0.164 ms/样本（约 4x），产物为**单文件 ONNX**，部署只需 onnxruntime。
4. **剪枝不适用**：无卷积层时平台明确拒绝（避免无效剪枝，属正确设计）。
5. **动态 batch 的取舍**：动态图 + 动态量化会触碰 onnxruntime 对 attention 图的 Reshape 限制；
   本场景固定 batch=1 更稳更快，需变批量时改走 OpenVINO IR。

## 八、结论与建议

* **结论**：压缩作业 CLI 可在无平台依赖的容器内独立运行，一条命令完成「导出 + 量化」，
  焊接模型达成 **压缩率 62.60% / 精度损失 0.06%**，满足验收标准。
* **建议**：
  1. 表格 / Transformer / CNN 分类模型的默认配方：`export:onnx` → `quantize:onnx_dynamic_int8`；
  2. 有校准数据时优先 `onnx_static_int8`（QDQ），通常精度更优；
  3. 产线产物推荐 `model_quantized.onnx`（单文件、无 PyTorch 依赖）；
  4. 需要更大压缩率时再评估 INT4（需引入 `onnx-ir` 并解决依赖冲突）。

## 九、风险与后续工作

| 风险 / 待办 | 说明 |
|-------------|------|
| INT4 依赖冲突 | `onnx-ir` 会升级 `protobuf/numpy`，需独立镜像或版本对齐后再启用 |
| GGUF 网络依赖 | 离线环境需预置 llama.cpp 工具链（`deploy/compress/llama-cpp/` + `LLAMA_CPP_PATH`） |
| HF 微调仅 GPU | CPU 镜像只跑 micro-LM 原型；真实 LoRA/DPO 需 `compress-job:gpu` |
| CPU 服务器镜像 | 把压缩 CLI 正式烘进 `backend:latest`（`scripts/compress-image.sh`） |
| 评测通用化 | 当前精度验收用示例脚本；后续可把「CSV 表格分类评测」并入平台评估器 |

## 十、复现步骤

```bash
# 1) 构建压缩镜像
docker build -f deploy/compress/Dockerfile.cpu -t compress-job:cpu .

# 2) 一键 demo（导出 → 量化 → 验收）
unzip welding-transformer-package-20260622T0620Z.zip -d /data/welding
deploy/compress/examples/welding_demo.sh \
  /data/welding/welding-transformer-package-20260622T0620Z /data/welding-out compress-job:cpu

# 3) 查看产物与报告
cat /data/welding-out/w02-int8-onnx/result.json
cat /data/welding-out/welding-eval.json
```

## 附录 A：产物清单

| 文件 | 说明 |
|------|------|
| `compress_job.py` | 单文件压缩作业 CLI |
| `deploy/compress/Dockerfile.{cpu,gpu,fast}` | 作业镜像定义 |
| `scripts/compress-image.sh` | 构建 / `docker save` 环境包 `dist/compress-job-cpu-<日期>.tar.gz` / 冒烟 |
| `scripts/compress-bundle.sh` | 打代码包（`--bundle-only`）或含镜像的全量包 |
| `docs/compression-job.md` | CLI 与参数说明 |
| `examples/compression-demo-welding.md` | 焊接模型 demo（含实测数据） |
| `deploy/compress/examples/welding_eval.py` | 表格分类评测脚本 |
| `deploy/compress/examples/results/` | 评测 JSON、作业 `result.json`、失败样例日志、INT8 产物 |

