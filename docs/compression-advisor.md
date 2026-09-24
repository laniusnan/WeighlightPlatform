# 压缩方法推荐规则（Advisor）

> 目标：输入**任务场景 / 模型 / 数据 / 部署指标**，输出「推荐用什么方法 + 为什么」。
> 规则实现：`frontend/src/components/lab/methodAdvisor.ts`（纯函数 `recommend()`）；
> UI：`frontend/src/components/lab/MethodAdvisorPanel.tsx`（接入 `/lab/compressions` 页面顶部）。
> 卡片文案与实测数值来自 `frontend/src/components/lab/method-cards.json`，其数字全部出自
> [`docs/compression-eval-report.md`](./compression-eval-report.md) 的真实运行。

## 1. 输入字段

| 字段 | 取值 | 用途 |
|------|------|------|
| `task` | 图像分类 / 目标检测 / 文本生成 / 表格分类 / 点云 | 决定方法族（视觉 vs LLM） |
| `modelKind` | cnn / yolo / transformer / llm / tabular / pointnet | 决定可用算子与限制（如剪枝需 Conv2d） |
| `modelMb` | 数值（MB） | 体积基数，用于展示压缩收益 |
| `data` | 无 / 少量图片 / 完整标注图片 / 指令 JSONL / 偏好对 | **是否能用需要校准/训练的方法**（关键分水岭） |
| `device` | 通用 CPU / Intel CPU / GPU / 边缘 | 影响 OpenVINO 是否值得、HF 微调是否可行 |
| `accuracyBudgetPp` | 可接受掉点（百分点） | ≤0.5pp 时加权 QAT / 静态 INT8 |
| `targetCompressionPct` | 目标压缩率（%） | ≥60% 时加权流水线/量化 |
| `needSpeed` | 是否要求提速 | 加权静态 INT8 / OpenVINO / INT4 |
| `needLlamaCpp` | 是否要 llama.cpp·ollama 生态 | 触发 GGUF |
| `needTaskAdaptation` | 是否要补领域能力 | 触发 LoRA/DPO |

## 2. 规则表（含实测依据）

| # | 触发条件 | 推荐（权重） | 为什么（实测证据） |
|---|----------|--------------|--------------------|
| R1 | `modelKind=llm` 或 `task=text-generation` | `llm_quantize`（INT4，95） | 免校准；体积 999.6 MB → **339 MB（−66%）**；平台基准 **p50 1307→704 ms（2.16×）**、峰值内存 2381→**1243 MB** |
| R2 | R1 且 `needLlamaCpp` | + `export_gguf`（q4_K_M，92） | llama.cpp/ollama 只认 GGUF；实测 0.5B → **511 MB GGUF（−48.8%）**、31.4 s、离线工具即可完成 |
| R3 | R1 且 `needTaskAdaptation` | + `llm_finetune`（LoRA，90） | 任务适配而非压缩：rank 8 只训 **1.08 M 参数**、适配器 **4.4 MB**、32 样本 1 epoch loss 1.277→0.838（19.2 s，GPU） |
| R4 | 有按类图片 且 `device∈{cpu,intel-cpu}` | `quantize_onnx_static`（96） | 本批**精度损失最小 −0.11pp**、**延迟最低 p50 0.079 ms / p99 0.247 ms**、体积 −72.4% |
| R5 | 任意（非 LLM） | `quantize_onnx`（有数据 82 / 无数据 92） | 免校准、体积最小 −73.7%、掉点 0.52pp；但 CPU 上 p50 是静态量化的 **5×**（0.440 vs 0.079 ms） |
| R6 | `device=intel-cpu` 且 视觉 且 有图片 | + `openvino_int8`（78） | −68.1%；但本批掉点最多（−1.78pp）且峰值内存最高（367 MB）→ 仅 Intel 部署值得 |
| R7 | 无数据 且 `cnn` | `prune`（L1/L2，55） | 无数据可剪（参数 94,986→45,915）；**但实测不微调精度崩到 20.1%/17.4%**，必须接 QAT/微调 |
| R8 | 视觉 且 有图片 且 目标压缩率 ≥60% | `pipeline prune→dynamic INT8`（85） | 单靠量化到不了：实测 **54,159 B（−86.2%）**；精度继承剪枝损失，需后续微调 |
| R9 | 视觉 且 有图片 且 精度预算 ≤0.5pp | + `qat`（88） | 训练中模拟量化；**实测轮次是关键**：1 epoch/4k 仅 62.8%，10 epoch/4k 升到 **66.83%（+4.03pp，val loss 0.9225→0.8093）**，但仍未追平静态 PTQ（68.68%）——全量 50k 在本机 OOM（exit 137），故**不主张 QAT > PTQ** |
| R10 | 视觉 且 有图片 且 目标压缩率 <60% | `distill`（60，附警告） | 10 epoch 可把随机学生拉到 61.3%；但本平台 `width_mult` 对顺序网络**不生效**（学生=教师参数量），真要压缩请用剪枝/量化 |
| R11 | `data=none` | `export_onnx`（45） | 没有数据时至少先导出 ONNX（精度等价），后续任何量化都能接 |
| R12 | 视觉 且 `data=none` | `sparsify`（30，强警告） | 实测 30% 稀疏 → 体积仅 −3.1%、精度 −19.8pp、稠密推理不提速 → 只当中间态 |

## 3. 三个示例（输入 → 输出）

**示例 A：CIFAR-10 CNN，有完整训练集，CPU，要求 −60% 且掉点 ≤1pp，要提速**
→ **首选 `quantize_onnx_static`**（有校准集、掉点 0.11pp、p50 0.079 ms）；
备选 `quantize_onnx`（无校准时的兜底，但慢 5×）、`qat`（精度预算更紧时）。

**示例 B：Qwen2.5-0.5B 部署到 CPU 服务，无标注数据，要 llama.cpp 生态跑**
→ **首选 `llm_quantize` INT4**（免校准、−66%、2.16×、内存减半）；
同时 **`export_gguf` q4_K_M**（511 MB，可直接被 ollama/llama.cpp 加载）。

**示例 C：目标检测 YOLO，只有少量图片，边缘设备，允许掉 2pp，要求体积最小**
→ **首选 `openvino_int8`**（Intel 边缘 + 少量图片做校准；注意本批掉点 1.78pp 接近上限）
→ 或 `quantize_onnx_static`；**不推荐** `prune`（平台剪枝需要 arch.py 且未微调会崩精度）。

## 4. 与「方法卡」的关系

* 方法卡（`docs/method-cards.schema.json` + `method-cards.json`）= **静态说明书**：某方法需要什么、适合什么、限制是什么；
* Advisor = **决策层**：在卡片之上按你的约束排序并给出理由与可复制命令；
* 两者共用同一份实测证据，改一处数字两处同步（`scripts/build_method_cards.py` 重新生成）。
