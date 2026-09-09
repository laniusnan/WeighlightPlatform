# 可移植算法与特性分析（xtuner / lmdeploy / mmrazor → 本平台）

> 目标：从 workspace 的 `xtuner`、`lmdeploy`、`mmrazor` 中甄别**非定制、通用**的模型轻量化/部署能力，
> 评估能否移植到 full-stack-fastapi（CPU-only Docker，FastAPI + Arq worker）。

## 0. 结论摘要

| 仓库 | 定位 | 可移植性 |
|---|---|---|
| mmrazor | 神经架构搜索 / 剪枝 / 蒸馏 / 量化（学术算法全集） | **算法层通用性最高**：KD 蒸馏、L1/L2 结构化剪枝、BN Slim、LSQ 类量化思路可借鉴 |
| xtuner | 大模型高效微调（QLoRA / DPO / ORPO 等） | 需要 GPU + DeepSpeed，当前 CPU 平台**不移植**；但“SFT 后量化”方法论进入评估/路线图 |
| lmdeploy | LLM 高性能部署（Turbomind / PyTorch 后端、W4A16/AWQ、OpenAI API、KV 管理） | 工程层绑定 C++/TRT/GPU kernel，**整体不移植**；其中“weight-only INT4 量化思路 + OpenAI 兼容接口”等概念本平台已有对位实现 |

## 1. mmrazor —— 推荐优先移植的“算法库”

### 1.1 通用算法及其归属

| 能力 | 出处目录 | 是否通用 | 本平台当前状态 | 建议 |
|---|---|---|---|---|
| Logits/Feature **知识蒸馏 (KD/OFD)** | `mmrazor/models/distillers/` | ✅ 经典算法，与具体骨干解耦 | 无 | ✅ **已落地（P1）**：新增 `distill` 方法（teacher=原模型，student=arch.py 学生 + `width_mult`，数据集=类别文件夹 zip） |
| L1/L2 结构化卷积剪枝 | `mmrazor/models/algorithms/pruning/` | ✅ | 已有 L2 结构化剪枝 `prune` | ✅ **已落地**：`prune` 支持 `algorithm=l1\|l2\|bnslim` |
| BN Slim（BatchNorm 缩放因子稀疏化） | pruning/structural | ✅ 经典 | 无 | 可作 `prune` 的选项（对带 BN 的 CNN） |
| 迭代全局稀疏（GlobalSparse / 幅度迭代剪枝 IMP） | pruning | ✅ | 已有一次性 `sparsify` | ✅ **已落地（策略层）**：`sparsify` 支持 `strategy=uniform\|global`；带数据的迭代+重训排后续 |
| LSQ / 可学习量化（训练中） | quantization | ✅ 通用思想 | 已有 FX QAT | 借鉴其“逐层量化参数可学习”，作为 QAT 改进项 |
| OpenVINO 量化导出 | `quantizers/openvino_quantizer.py` + exporters | ✅ 与本平台目标一致 | 已有 `openvino_int8` | 对比其校准采样策略，补齐 `quantize_onnx_static` 的校准逻辑 |
| NAS / DARTS | algorithms/nas | ⚠️ 算法通用但需 GPU 搜索 | 无 | **不建议** CPU 移植；作为 GPU Worker Roadmap |

### 1.2 可直接借鉴但不宜照搬的实现细节
- 蒸馏 loss 组合（KD λ、温度 T、feature 对齐层）；
- 剪枝的“对候选通道打分 → 排序 → 删除 → 微调”pipeline（本平台可在剪枝后自动接可选 `qat` 恢复精度）；
- 量化 exporter 的“对称/非对称、per-channel、校准集采样数”统一参数命名。

## 2. xtuner —— 能力与边界

### 2.1 值得记录的能力（不移植，进 Roadmap）
| 能力 | 位置 | 结论 |
|---|---|---|
| QLoRA / LoRA 低秩微调 | `xtuner/` recipes（qwen/llama…）+ deepspeed | 依赖 GPU/DeepSpeed；作为平台**GPU Worker 二期**（`finetune_lora`） |
| DPO / ORPO / Reward Model | recipes | 微调后增强，同上 |
| 与量化组合的“训练后量化 + 微调恢复” | 方法论 | 与本平台 `qat`、`llm_quantize` 流程形成互补文档 |
| 数据处理/模板（对话模板、tokenizer） | configs | 仅适用于 HF 系 LLM，通用性低 |

## 3. lmdeploy —— 工程部署层分析

| 能力 | 位置 | 可移植结论 |
|---|---|---|
| W4A16/AWQ/GPTQ 风格 **weight-only INT4** | `lmdeploy/lite`、`tools` | 算法思想通用。本平台已有 `llm_quantize`（OpenVINO INT4/INT8）、`export_gguf`（llama.cpp q8_0/tq…）；可选新增 **RTN/GPTQ 近似 INT4（CPU group-wise）** |
| OpenAI 兼容 Chat Completions（SSE） | `lmdeploy/serve` | **概念与本平台一致**：本平台已有 `/api/v1/chat/completions`（SSE/流式），无需移植 |
| KV Cache / continuous batching / PagedAttention | `lmdeploy/pytorch`、turbomind | C++/GPU/显存管理强绑定，不移植 |
| TensorRT / Triton 后端 | turbo | 不移植 |
| 基准测试 harness（吞吐/latency/首 token） | `benchmark/` | 通用方法论；本平台已有 CPU benchmark（p50/p99/吞吐/内存），可借鉴其“预热 + 分位数”细节 |

## 4. 建议的“可落地”改进清单（按优先级）

| 优先级 | 功能 | 来源 | CPU 可行 | 落地方式 |
|---|---|---|---|---|
| P0 | `prune` 增加 `algorithm=l1|l2` + BN-Slim 选项 | mmrazor | ✅ 是 | 扩展现有 prune 实现，无需新依赖 |
| P0 | `sparsify` 增加迭代幅度裁剪 `rounds/schedule` | mmrazor | ✅ 是 | 扩展现有实现 |
| P1 | 新增 `distill`（KD，CPU 小模型） | mmrazor distillers | ✅ 是（小模型） | 新 method + 向导表单 + 文档 |
| P1 | LLM 增加 RTN/GPTQ 风格 group-wise INT4 选项 | lmdeploy lite | ⚠️ 需与 OpenVINO 导出链路一致 | ✅ **已落地（P1，RTN 风格）**：`llm_quantize` 支持 `group_size=32…256` 与 `sym`；GPTQ/AWQ（需 GPU 离线校准）列入 Roadmap |
| P2 | QAT 的 LSQ 化（可学习步长） | mmrazor | ⚠️ | QAT 增强 |
| P2 | GPU Worker（xtuner LoRA/DPO + NAS） | xtuner/mmrazor | ❌ | 独立 GPU 队列/部署图 |

## 5. 与本平台现有 9 种方法的对应关系

export_onnx / quantize_onnx / quantize_onnx_static / openvino_int8 / prune / sparsify /
qat / export_gguf / llm_quantize —— 与上表对照后可见：**mmrazor 的剪枝/量化思想、lmdeploy 的 weight-only 量化方法论、xtuner 的微调流程**分别落在前 6 类/后 3 类上；
本次分析用于指导后续在不引入 GPU/C++ 依赖的前提下，把“通用算法”逐步沉淀为本平台一等公民方法。

## 6. 同步到技术使用文档
- 本文件：`docs/portable-features.md`（算法可移植性总览）
- `README.md` 增加“可移植特性与算法路线图”小节
- 实现某方法后：更新前端方法列表 `frontend/src/components/lab/constants.ts`、`backend/app/models.py` 的 `CompressionMethod`、压缩向导参数、deployment-cpu.md

## 附：P1 端到端验证记录（2026-09）
- `llm_quantize` RTN INT4 参数已落地并通过编译；尝试离线 tiny-GPT2 端到端时，
  本地 venv 与平台 docker 镜像均在 `optimum.exporters.base` 报
  `NormalizedConfig.__init__() got multiple values for argument 'allow_new'`。
- 版本：optimum 2.1.0 / optimum-intel 1.27.0 / transformers 4.57.6 / nncf 3.3.0。
  建议将 `transformers` 固定到 4.54.x（或升级支持 4.57 的 optimum-intel）后重跑 E2E。
- 该冲突同样影响改动前已存在的 `llm_quantize` 方法（非本次新增问题）。
- GPTQ/AWQ（GPU 离线校准）与 GGUF `q4_K_M`（需 llama-quantize 工具链）
  在当前环境（无 GPU / 无工具链 / 无 HF 外网）无法端到端，保留为 Roadmap。
- 进一步实验：将 transformers 降到 4.56.2 / 4.55.4 后错误依旧 —— 初步判定是 **optimum 2.1.0 导出层 bug**
  （与 transformers 版本无关）。建议在有外网环境把 optimum/optimum-intel 升到支持 transformers 4.57 的新版后重跑。
- 服务器已查：`LLAMA_CPP_PATH` 未设置，`/opt/app` 下无 llama-quantize —— GGUF q4_K_M 端到端待先装好 llama.cpp 工具链。
- 再升级验证：optimum 2.3.0 + optimum-intel 2.1.0 在 transformers 4.57.6 / 4.55.4 / 4.54.1 下
  均报同一错误 —— 初步确认 bug 在 **optimum 2.x 导出层的 NormalizedConfig**（与其允许的
transformers 区间无关）。建议向 optimum 仓库提交 issue（附本段版本矩阵与复现脚本
`scripts/e2e-llm-rtn.py`）；后续可研究不经 optimum 的纯 NNCF 权重压缩路径。
- 本地环境已还原锁版本（optimum 2.1.0 / optimum-intel 1.27.0 / transformers 4.57.6）。

## mmrazor 资产落地清单（1+2+3 完成）
1. **QAT 约束参数**：`qat` 支持 `weight_mode=per_channel|per_tensor` 与 `weight_symmetric=true|false`
   （对齐 mmrazor OpenVINOQuantizer 的 backend 属性：weight 支持 per_channel、对称 [-127,127]）。
2. **纯 NNCF INT4 可行脚本**：`scripts/e2e-nncf-int4.py` 已跑通
   torch MLP → ONNX → OpenVINO → `nncf.compress_weights(INT4_ASYM, group=32)` → CPU 运行
   （fp32 67207B → INT4 37917B）。LLM 解码器动态导出留作后续工程项。
3. **蒸馏器**：`distill`（mmrazor distillers 思路）此前已落地；NAS/LSQ、GPTQ/AWQ、GPU Worker 见 Roadmap。
- `llm_quantize` 新增 `engine=auto|optimum|nncf`：本地 HF 目录默认走 NNCF 纯路径
  （OV trace → INT4 group → IR+tokenizer+config）；optimum 路径保留给修复后的环境。
- engine=nncf decoder E2E：完整 HF 目录（含离线训练 BPE tokenizer）已构造并跑通 tokenizer 阶段；
  随后 OpenVINO torch 前端 trace GPT2 报 `TorchScript trace failed: unordered_map::at`（与 torch.onnx 同源，环境级：
  python3.14 + 当前 torch/transformers 组合）。engine=nncf 代码就绪待可在 python3.11/可 trace 的 torch 环境验证；
  第 3 条（无 KV 朴素生成运行时）需拿到任意 decoder IR 后再接。

---
## 环境固化：ML 解码器运行时（py3.11 + torch 2.9 + transformers 4.49）

**验证结论（2026-09）：** decoder trace/量化在全栈主环境（py3.14 + torch2.13 +
transformers4.57）失败（`TorchScript trace … unordered_map::at`）；在下列组合下
**GPT 类 decoder 全链路通过**（OV trace → NNCF INT4 → IR 产物）。

| 包 | 版本（已验证） |
|---|---|
| python | 3.11.15 |
| torch | 2.9.1+cpu（本地 CUDA12.6 可换 +cu126） |
| transformers | 4.49.0 |
| tokenizers | <0.22 |
| openvino | 2026.3.1 |
| nncf | 3.3.0 |
| optimum / optimum-intel | 最新（路由：engine=nncf 优先） |

- 一键创建：`bash scripts/setup-ml-py311.sh [.venv311]`（uv 管理）。
- 注意：`app.models` 依赖 py3.14 的 PEP649 延迟注解，3.11 下请用 stub
  直接加载 `app.jobs.compression`（参考验证脚本），平台任务运行仍走主环境。

## chat/completions 双路径（2026-09）

- `engine=nncf` 产物带 `engine_nncf.txt` 标记 → 路由用 `NncfChatModel`
  （`backend/app/jobs/llm_runtime_nncf.py`）无 KV 朴素生成：每个 token 一次完整前向，
  动态 seq 输入；非流式 `generate()` 与流式 `generate_text_stream()` 均已用真实
  INT4 IR 冒烟通过（tiny 模型退化为重复 token 属预期）。
- optimum 产物（`openvino_model.xml` 等）仍走 `OVModelForCausalLM` + 流式不变。
- 真实 LLM 验证：服务器已布置 llama 量化产物后，平台内 `任务->LLM 量化
  (engine=nncf)` → 产物目录即可直接挂到 `POST /api/v1/inference/chat/completions`
  （需要时先 `.venv311` 内 URL 拉取完整 HF 目录）。

## A/B/C 里程碑（2026-09，venv311 实测）
- **A tiny-LLaMA E2E**：真实 LLaMA 架构（LlamaConfig+LlamaForCausalLM，离线随机权重+
  BPE tokenizer）在 py3.11 组合下：OV trace → NNCF INT4（group=32）→ `engine_nncf.txt`
  产物 → `NncfChatModel` 生成出文本。→ 平台 `llm_quantize engine=nncf` 对 LLaMA 家族可直用。
- **B 采样参数**：`NncfChatModel.generate/generate_text_stream` 与
  `ChatRequest` 新增 `top_p`(默认1.0)/`min_p`(默认0.0)；nucleus+min-p 组合过滤，
  temperature=0 时退化为贪心。实测 do_sample=True 多样输出 ✅。
- **C stateful 起步**：optimum-intel 在 4.49 组合下对 LLaMA 导出成功
  （`stateful=True`，openvino_model.xml 4 输入），`OVModelForCausalLM.generate`
  产出 (1,13) 含 KV 缓存路径 ✅。工程化剩余：平台 worker 环境对齐 py3.11 后，
  `engine=optimum` 即平台级 stateful 运行时；否则先用 naive 双路径。
- 服务器真实 llama 平台级 E2E 待办（本机无模型/服务器不可达）；GGUF/llama.cpp 对照同待办。

## CPU 部署环境对齐（2026-09，py3.11 组合入锁）
已改并 `uv lock` 重解析成功：
- backend `requires-python>=3.11`；`transformers>=4.49,<4.50`；新增 `tokenizers>=0.21,<0.22`；
  移除未使用的 `torchao`；torch override `2.9.1+cpu`（root pyproject）；Dockerfile.cpu 基镜像 `python:3.14→3.11`。
- `app/models.py` 加 `from __future__ import annotations`（PEP563），py3.11 可正常 import（实测 `PY311_CORE_IMPORT_OK`）。
- 重新部署：`./scripts/deploy-to-server.sh [HOST]` → 服务器 `docker compose -f compose.cpu.yml up -d --build`。
- 模型获取：HF 在此网络被墙，走 ModelScope（`Qwen/Qwen2.5-0.5B-Instruct`）本地下载后 rsync 到服务器
  `/home/zhangxuefei/Model/`，随后平台 URL 拉取选本地目录 / 直接指向该目录做 engine=nncf 量化。
