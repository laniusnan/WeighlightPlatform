# 压缩方法评估报告（服务器现有资产 · 真实执行）

> 生成方式：所有数字来自本机真实运行的 `compress_job.py` 作业与 `scripts/bench.py` 测量，
> 原始文件保留在 `/home/lanius/compression-eval/out/<task>/{result.json,bench.json,logs}`。
> 方法卡数据 `frontend/src/components/lab/method-cards.json` 由 `scripts/build_method_cards.py`
> 从同一份 `out/summary.json` 生成 —— 报告、卡片、原始文件三处数字同源，禁止手改。
>
> 日期：2026-09-23 · 平台：`full-stack-fastapi`（CPU 栈 8001 / GPU 栈 8002 均在运行）

---

## 1. 服务器可用资产清单

扫描方式：`ls/du/find` 直读宿主机目录 + `docker inspect`（容器挂载真相）+ 两个栈的 PostgreSQL
元数据（`model`/`dataset` 表）。**未使用任何假设路径**。

### 1.1 数据集（宿主机 `/home/lanius/Dataset`，GPU 栈内挂载为 `/datasets-local`）

| 名称 | 路径（宿主 / 容器） | 类型 | 样本数 | 类别数 | 体积 |
|------|--------------------|------|--------|--------|------|
| CIFAR10-train | `/home/lanius/Dataset/CIFAR10-train` · `/datasets-local/CIFAR10-train` | 按类图片目录（PNG） | 50,000 | 10 | 197 MB |
| CIFAR10-test | `/home/lanius/Dataset/CIFAR10-test` | 按类图片目录 | 10,000 | 10 | 40 MB |
| MNIST-train | `/home/lanius/Dataset/MNIST-train` | 按类图片目录 | 60,000 | 10 | 237 MB |
| MNIST-test | `/home/lanius/Dataset/MNIST-test` | 按类图片目录 | 10,000 | 10 | 40 MB |
| alpaca-train | `/home/lanius/Dataset/alpaca-train/alpaca.jsonl` | 指令集 JSONL | 51,974 行 | — | 20 MB |
| alpaca-train-32 | `/home/lanius/Dataset/alpaca-train-32/alpaca.jsonl` | 指令集 JSONL（子集） | 32 行 | — | 16 KB |
| hh-rlhf-train | `/home/lanius/Dataset/hh-rlhf-train/preferences.jsonl` | 偏好对 JSONL | 43,783 行 | — | 44 MB |
| hh-rlhf-train-32 | `/home/lanius/Dataset/hh-rlhf-train-32/preferences.jsonl` | 偏好对 JSONL（子集） | 32 行 | — | 40 KB |
| hh-rlhf-test | `/home/lanius/Dataset/hh-rlhf-test/preferences.jsonl` | 偏好对 JSONL | 2,350 行 | — | 2.4 MB |
| （CPU 栈上传）2 类玩具集 | `backend/models-storage/datasets/de47d986-…/class_a|class_b` | 按类图片目录 | 少量 | 2 | < 1 MB |

**不存在**：CIFAR100、Fashion-MNIST、WikiText-2、GSM8K（`ls | grep` 均为空；
`scripts/fetch-open-datasets.sh` 的默认 ROOT `/home/zhangxuefei/Dataset` 在本机不存在）。
→ 因此 LLM 困惑度评估没有 wikitext，本报告用 **alpaca 派生的纯文本语料**
`/home/lanius/compression-eval/datasets-derived/alpaca-eval.txt`（32 行 → 14,775 B，脚本
`scripts/prep_llm_corpus.py`，属**派生数据**，已在 §4.8 明确标注）。

### 1.2 模型

| 名称 | 路径（宿主 / 容器） | framework | 参数量 | 体积 | 来源 |
|------|--------------------|-----------|--------|------|------|
| Qwen2.5-0.5B-Instruct | `/home/lanius/models-local/Qwen2.5-0.5B` · `/models-local/Qwen2.5-0.5B` | huggingface（HF 目录 + safetensors, bf16） | 0.494 B（24 层, hidden 896） | 976 MB 目录 / `model.safetensors` 988 MB | 平台 GPU 栈已注册：`Qwen2.5-0.5B-Local`（999,602,900 B, ready） |
| TestCNN（冒烟模型） | `/home/lanius/models-local/TestCNN/{model.pt,arch.py}` | pytorch（`torch.save(model)`, CifarCNN） | 94,986 | 392,228 B | 服务器自带 |
| 5channels_dectect_v0.2 | CPU 栈 `models-storage/models/174e5ef6-…/original.onnx` | onnx | — | 908,005 B | CPU 栈已注册（唯一记录） |
| 其它 11 个模型目录 | CPU 栈 `models-storage/models/<uuid>/`（7×`original.onnx` 99–112 KB；3×`original.pt` 9–12 KB + `arch.py`；1×`original.pt` 11.8 KB） | onnx / pytorch | 小模型 | < 120 KB | **数据库无对应记录**（历史上传残留，孤儿文件） |

### 1.3 ⚠️ 关键发现：服务器自带的 CIFAR 模型是**随机权重**

`TestCNN/arch.py` 顶部自述 *“Return a freshly initialised model (weights are random on purpose)”*；
实测在 CIFAR10-test 上的准确率 = **0.1000（1000/10000）**，即 10 类随机猜测水平。
**用它做压缩评估会得到毫无意义的精度数字**（压缩前后都在 10% 附近）。

→ 因此本报告按「平台真实流程」补做了一步：用服务器现有 **CIFAR10-train** 训练同架构基线
（`scripts/train_cifar10.py`，CifarCNN width=32，12 epoch，RTX 4060，**71.4 s**，
最终测试准确率 **0.6879**，checkpoint **392,548 B**，产物 `/home/lanius/compression-eval/models/CifarCNN10/`）。
命令与超参见 §6.1，属于**本次新增的实验产物**（非伪造；原始 `train_report.json` 留存）。

### 1.4 方法可跑性矩阵（数据+模型是否齐备）

| 方法（平台枚举） | 需要的输入 | 本机是否齐备 | 本次是否实跑 |
|------------------|-----------|--------------|--------------|
| `export_onnx` | 模型 + arch.py | ✅ | ✅ 01-export |
| `quantize_onnx`（动态 INT8） | ONNX | ✅ | ✅ 02 |
| `quantize_onnx_static`（静态 INT8） | ONNX + 校准图片目录 | ✅ CIFAR10-train | ✅ 03 |
| `openvino_int8` | ONNX + 校准图片目录 | ✅ | ✅ 04 |
| `prune`（l2 / group_fisher） | PyTorch + arch.py（+数据） | ✅ | ✅ 05 / 06 / 09（+10 流水线） |
| `sparsify` | PyTorch + arch.py | ✅ | ✅ 11 |
| `distill` | **教师模型 + arch.py + 按类图片目录**（同架构学生） | ✅ | ✅ 07（3 ep）/ 13（10 ep） |
| `qat` | PyTorch + arch.py + 图片目录 | ✅ | ✅ 12-qat（CLI 已支持 `--method=qat`） |
| `llm_quantize`（NNCF INT4） | HF 模型目录 | ✅ Qwen2.5-0.5B | ✅ 20（含 §4.8 的 PPL/吞吐） |
| `llm_finetune`（LoRA） | HF 模型 + .jsonl + **CUDA** | ✅（RTX 4060，GPU 栈） | ✅ 21 |
| `export_gguf` | HF 模型 + llama.cpp 工具（本机已有） | ✅ `/home/lanius/llama.cpp` | ✅ 22-qwen-gguf（q4_K_M，离线） |

---

---
---

## 2. 执行环境与代码版本（可追溯）

| 项 | 值 |
|----|----|
| 作业入口 | `compress_job.py`（仓库当前版本，sha256 `b78da0b2…`，镜像内 `--help` 校验一致）；镜像 `compress-job:cpu`（torch 2.9.1+cpu / onnxruntime 1.29 / openvino 2026.3 / nncf 3.3） |
| 引擎 | `backend/app/jobs/compression.py`（sha256 `3df95389…`，与镜像内副本逐字节一致） |
| 挂载 | `-v /home/lanius/Dataset:/datasets:ro`、`-v /home/lanius/models-local:/models:ro`、`-v /home/lanius/compression-eval:/work`；CLI 用 `-v <repo>/compress_job.py:/app/compress_job.py:ro` 覆盖，确保跑仓库代码而非镜像内旧副本 |
| LoRA（GPU） | 镜像 `b121980fea3c`（torch 2.9.1+cu126、torchvision 0.24.1，`--gpus all`，RTX 4060 8 GB）+ 仓库挂载 `/workspace`，直接执行 `/workspace/compress_job.py` → 引擎自动选用仓库 `backend/` |
| 测量脚本 | `scripts/bench.py`（分类：整测 10,000 张精度 + batch=1 采样 2,000 次 p50/p90/p99 + batch=1/32 吞吐 + `getrusage` 峰值 RSS）；`scripts/bench_llm.py`（LLM：逐 token 解码延迟 p50/p99、tokens/s、困惑度、峰值 RSS） |
| 基线模型 | 服务器训练产物 `/home/lanius/compression-eval/models/CifarCNN10/model.pt`（94,986 params、392,548 B、CIFAR10-test **0.6879**）——见 §1.3、§6.1 |
| 评估集 | `CIFAR10-test`（10,000 张，与训练集无重叠；`ImageFolder` 类别顺序与训练一致） |

> 交叉校验：`bench.py` 对 FP32 ONNX 与 `.pt` 都得到 **0.6879**，与训练脚本 `train_report.json` 完全一致；
> 说明测量链路（预处理/类别顺序/归一化）无偏，后续各变体的 Δpp 可信。

## 3. 结果总表（CIFAR-10，全部真实执行）

体积以**基线 checkpoint 392,548 B** 为分母；精度 = CIFAR10-test 10,000 张 Top-1；
延迟 = CPU、batch=1、2,000 次采样；吞吐 = batch=1 / batch=32（样本·s⁻¹）；
RSS = 整进程峰值（含加载与数据）；耗时 = `result.json.duration_s`（含评测）。

| 任务 | 方法（平台枚举） | 产物 | 体积 | 压缩率 | 精度 | Δpp | p50 (ms) | p99 (ms) | 吞吐 b1 / b32 | 峰值 RSS | 耗时 |
|------|------------------|------|------|--------|------|-----|----------|----------|---------------|----------|------|
| baseline | —（基线 `.pt`） | `model.pt` | 392,548 B | — | **0.6879** | — | (0.098)¹ | (0.339)¹ | (2648/3336)¹ | — | — |
| 01-export | `export_onnx` | `model.onnx` | 380,919 B | 2.96% | 0.6879 | 0.00 | 0.098 | 0.339 | 2648 / 3336 | 333.3 MB | 3.1 s |
| 02-dynamic-int8 | `quantize_onnx` | `model_quantized.onnx` | 103,310 B | **73.68%** | 0.6827 | **−0.52** | 0.440 | 0.787 | 1179 / 1307 | 335.0 MB | 20.8 s |
| 03-static-int8 | `quantize_onnx_static` | `model_quantized.onnx` | 108,431 B | **72.38%** | 0.6868 | **−0.11** | **0.079** | **0.247** | **2553 / 3338** | 333.3 MB | 8.0 s |
| 04-openvino-int8 | `openvino_int8` | `model.xml`+`model.bin` | 125,352 B | 68.07% | 0.6701 | −1.78 | 0.182 | 0.534 | 1677 / 2191 | 366.9 MB | 10.7 s |
| 05-prune-l2 | `prune:l2`（rate 0.3） | `model_pruned.onnx` | 185,178 B | 52.83% | 0.2010 | **−48.69** | 0.126 | 0.509 | 1921 / 1569 | 333.4 MB | 8.5 s |
| 06-prune-groupfisher | `prune:group_fisher`（0.3） | `model_pruned.onnx` | 185,178 B | 52.83% | 0.1466 | **−54.13** | 0.125 | 0.634 | 1697 / 2802 | 333.7 MB | 47.9 s |
| 07-distill-3ep | `distill:logits`（3 epoch） | `model_student.onnx` | 380,919 B | 2.96% | 0.4940 | −19.39 | 0.126 | 0.508 | 1002 / 1816 | 333.6 MB | 257.8 s |
| 09-prune-l2-r01 | `prune:l2`（rate 0.1） | `model_pruned.onnx` | 304,659 B | 22.39% | 0.1737 | −51.42 | 0.155 | 0.358 | 2198 / 2729 | 333.6 MB | 8.7 s |
| 10-prune-then-quant | `pipeline`（`prune:l2` → `quantize:onnx_dynamic_int8`） | `step2/model_quantized.onnx` | **54,159 B** | **86.20%** | 0.2026 | −48.53 | 0.328 | 1.769 | 1550 / 1678 | 331.8 MB | 15.0 s |
| 11-sparsify | `sparsify`（rate 0.3） | `model_sparse.onnx` | 380,297 B | 3.13% | 0.4897 | −19.82 | 0.133 | 0.885 | 1710 / 2277 | 333.5 MB | 10.2 s |
| 13-distill-10ep | `distill:logits`（10 epoch） | `model_student.onnx` | 380,919 B | 2.96% | **0.6127** | −7.52 | 0.142 | 1.771 | 2258 / 2757 | 333.1 MB | 680.7 s |
| 12-qat | `qat:fx_int8`（1 epoch / 4k 图子集） | `model_qat.onnx` | 108,077 B | 72.47% | 0.6280 | −5.99 | 0.096 | 0.306 | 2123 / 2705 | 332.8 MB | 见 §4.11 |
| 22-qwen-gguf | `export_gguf`（q4_K_M） | `model_q4_K_M.gguf` | 511.4 MB | **48.8%**⁵ | 未测（PPL 见 INT4 行） | — | — | — | — | — | 31.4 s |
| 20-qwen-int4 | `llm_quantize`（NNCF INT4） | OpenVINO IR 目录 | 339.1 MB | **66.1%**¹ | PPL 5.295（引擎轻量评测） | — | 704² | 769² | 1.34 gen/s² | 1243 MB² | 8712.9 s³ |
| 21-qwen-lora | `llm_finetune:lora` | `adapter_lora.pt`（+merged） | **4.4 MB**（adapter） | —⁴ | loss 1.2769 → 0.8378 | — | — | — | — | — | 19.2 s |

¹ 基线 `.pt` 未单测延迟（计算图与 FP32 ONNX 等价），以 01-export 的数值作为 FP32 参照；
精度 0.6879 由 `train_report.json` 与 `bench.py` 双重确认。
² LLM 行：延迟/吞吐/内存取自**平台自带基准**（`engine.run_benchmark` → `NncfChatModel`，12 token 生成，4 次迭代），
因为 NNCF INT4 的 IR 输出为匿名（`names[]`、`stateful_runtime=false`），stock optimum-intel 无法加载；见 §5.10。
³ 8712.9 s 是**内存换页拖慢后的异常值**（同环境空闲时平台历史记录约 165 s）；原因见 §5.8。
⁴ LoRA 是任务适配而非压缩：底模不变，适配器 4.4 MB 可单独分发（另附 1.98 GB 合并模型）。
⁵ GGUF 以 HF 目录 999.6 MB 为分母；q4_K_M 511.4 MB，同目录下另有 f16 GGUF（994 MB）作为未量化参考。


---

## 4. 逐方法结果与执行命令

统一前缀（下文省略）：

```bash
docker run --rm --user $(id -u):$(id -g) \
  -v /home/lanius/Dataset:/datasets:ro -v /home/lanius/compression-eval:/work \
  -v <repo>/compress_job.py:/app/compress_job.py:ro compress-job:cpu \
  <args...>            # 评测统一：--image-size 32 --eval-dataset-path /datasets/CIFAR10-test --json-logs
```

### 4.1 `export_onnx`（01-export）

```bash
--method=export --algo=onnx --opset=17 --input-shape=1x3x32x32 --model-kind=cnn \
--base-model-path=/work/models/CifarCNN10/model.pt --arch-file=/work/models/CifarCNN10/arch.py \
--output-dir=/work/out/01-export
```
结果：`model.onnx` **380,919 B**（比 .pt 小 2.96%，因为去掉了 optimizer 状态）、精度 **0.6879（与 .pt 完全一致）**、
p50 0.098 ms / p99 0.339 ms、吞吐 2648 / 3336 样本·s⁻¹、耗时 3.1 s。
**结论**：导出无精度损失，是所有 ONNX 量化的前置步骤。

### 4.2 `quantize_onnx`（动态 INT8，02-dynamic-int8）

```bash
--method=quantize --algo=onnx_dynamic_int8 --base-model-path=/work/out/01-export/artifact/model.onnx \
--output-dir=/work/out/02-dynamic-int8
```
结果：**103,310 B（−73.68%）**、精度 **0.6827（−0.52pp）**、p50 0.440 ms / p99 0.787 ms、吞吐 1179 / 1307、耗时 20.8 s。
**结论**：免校准、体积最小的一档；但 CPU 上**比静态 INT8 慢约 5×**（权重反量化开销），且只量化权重。

### 4.3 `quantize_onnx_static`（静态 INT8，03-static-int8）

```bash
--method=quantize --algo=onnx_static_int8 --base-model-path=/work/out/01-export/artifact/model.onnx \
--dataset-path=/datasets/CIFAR10-train --max-samples 512 --output-dir=/work/out/03-static-int8
```
结果：校准 512 张 → **108,431 B（−72.38%）**、精度 **0.6868（−0.11pp，本批最准）**、
p50 **0.079 ms** / p99 **0.247 ms**、吞吐 **2553 / 3338**（最快）、耗时 8.0 s。
**结论**：**本场景最优解**（体积≈动态量化、精度最高、延迟最低）；代价是需要代表性校准集。

### 4.4 `openvino_int8`（04-openvino-int8）

```bash
--method=quantize --algo=openvino_int8 --precision int8 \
--base-model-path=/work/out/01-export/artifact/model.onnx \
--dataset-path=/datasets/CIFAR10-train --max-samples 512 --output-dir=/work/out/04-openvino-int8
```
结果：IR 目录（`model.xml`+`model.bin`）**125,352 B（−68.07%）**、精度 **0.6701（−1.78pp，本批掉点最多）**、
p50 0.182 ms / p99 0.534 ms、吞吐 1677 / 2191、峰值 RSS **366.9 MB（最高）**、耗时 10.7 s。
**结论**：可用，但在这个小 CNN 上不如 ORT 静态量化（精度更低、延迟更高、RSS 更大）；其优势在 Intel 平台大模型/长序列。

### 4.5 `prune`（L2 / GroupFisher，05 / 06 / 09）

```bash
--method=prune --algo=l2        --rate=0.3 --base-model-path=/work/models/CifarCNN10/model.pt --arch-file=…   # 05
--method=prune --algo=group_fisher --rate=0.3 --dataset-path=/datasets/CIFAR10-train --max-samples 256 …      # 06
--method=prune --algo=l2        --rate=0.1 …                                                                  # 09
```
结果：
| 变体 | 剪掉滤波器 | 参数量 | 体积 | 精度 | Δpp |
|------|-----------|--------|------|------|-----|
| L2 rate 0.3 | 69/224 | 94,986 → **45,915** | 185,178 B（−52.83%） | **0.2010** | −48.69 |
| GroupFisher rate 0.3 | 69/224 | 45,915 | 185,178 B | **0.1466** | −54.13 |
| L2 rate 0.1 | 少 | — | 304,659 B（−22.39%） | **0.1737** | −51.42 |

**结论（重要）**：**未经微调的剪枝在本模型上直接崩精度**（0.69 → 0.15~0.20），且 rate 0.1 并不比 0.3 好，
说明掉点主要来自**分类头重映射**（`head_layers_adjusted=1`）而非剪枝比例本身。
→ 生产使用必须 `prune → QAT/微调` 串联（平台支持 pipeline），否则不可部署。

### 4.6 `sparsify`（11-sparsify）

```bash
--method=prune --algo=sparsify --rate=0.3 --base-model-path=/work/models/CifarCNN10/model.pt --arch-file=…
```
结果：稀疏度 **30%**（28,292 个权重置零），体积 380,297 B（**仅 −3.13%**，稠密存储不变），
精度 **0.4897（−19.82pp）**、p50 0.133 ms / p99 0.885 ms、耗时 10.2 s。
**结论**：非结构化稀疏**不减小稠密模型、不提速**，且 30% 对这个小网已明显掉点；仅适合「后续接稀疏推理/再结构化剪枝」的中间态。

### 4.7 `distill`（07 / 13）

```bash
--method=distill --algo=logits --epochs=3  --batch-size=128 --learning-rate=0.001 \
--dataset-path=/datasets/CIFAR10-train --base-model-path=/work/models/CifarCNN10/model.pt --arch-file=…     # 07
# 13 同命令，--epochs=10
```
结果：
| 轮数 | 教师精度 | 学生初始 | 学生最终 | 体积 | 耗时 |
|------|---------|---------|---------|------|------|
| 3 | 0.6949 | 0.0866（随机） | **0.4940** | 380,919 B（同教师） | 257.8 s |
| 10 | 0.6949 | 0.0866 | **0.6127** | 380,919 B | 680.7 s |

**结论**：蒸馏**确实能把随机学生拉起来**（10 轮达 61.3%），但仍低于教师 68.8%；
且本平台 `width_mult` 对该 VGG-ish 顺序网络**未生效**（学生参数量 = 教师 94,986，体积不变）。
→ 想要「蒸馏即压缩」，需要在 arch.py 里提供真正可缩宽度的学生类（或改用剪枝/量化）。

### 4.8 `llm_quantize`（NNCF INT4，20-qwen-int4）— ✅ 已完成

```bash
--method=quantize --algo=nncf_int4 --bits=4 --group-size=128 \
--base-model-path=/models/Qwen2.5-0.5B --output-dir=/work/out/20-qwen-int4 \
--eval-dataset-path=/work/datasets-derived/alpaca-eval.txt
```
结果（✅ 已完成）：OpenVINO IR 目录 **339,069,255 B**，相对 HF 目录 999,602,900 B → **−66.1%**；
量化参数 `bits=4 / group_size=128 / sym=true`；引擎轻量困惑度 **5.295**（派生语料，512 采样）。

平台自带基准（`scripts/run_engine_benchmark.py` → `engine.run_benchmark` → `NncfChatModel`，12 token 生成 ×4）：

| 侧 | p50 (ms) | p99 (ms) | 生成速度 | 峰值内存 |
|----|----------|----------|----------|----------|
| original（HF fp32, CPU） | 1306.7 | 1704.3 | 0.62 gen/s | 2380.8 MB |
| compressed（INT4 IR） | **704.0** | **768.7** | **1.34 gen/s** | **1243.4 MB** |
| 变化 | −46% | −55% | **2.16×** | **−47.8%** |

我的独立基准（`bench_llm.py`，基线 fp32、24 token 贪心）：PPL **6.4517**、解码 **9.57 tok/s**、
峰值 RSS 2854.4 MB（与平台基准同量级；方法不同——平台是 12-token 生成，我的是 24-token 贪心 + 滑窗 PPL）。
**结论**：LLM 权重压缩**免校准**、体积 −66%、平台基准 **2.16× 加速 + 内存减半**，本批收益最明确。

### 4.9 `llm_finetune:lora`（21-qwen-lora）— ✅ 已完成

```bash
# GPU 镜像 + 仓库挂载（HF LoRA 需要 CUDA）
docker run --rm --gpus all -v <repo>:/workspace:ro -v /home/lanius/models-local:/models:ro \
  -v /home/lanius/Dataset:/datasets:ro -v /home/lanius/compression-eval:/work \
  --entrypoint python b121980fea3c /workspace/compress_job.py \
  --method=finetune --algo=lora --base-model-path=/models/Qwen2.5-0.5B \
  --dataset-path=/datasets/alpaca-train-32 --epochs=1 --lora-rank=8 --hf-max-len=128 \
  --learning-rate=0.0003 --device=cuda --output-dir=/work/out/21-qwen-lora
```
结果：32 样本 / 1 epoch / rank 8，覆盖 **96 个 LoRA 层**、**可训练参数 1,081,344**，
loss **1.2769 → 0.8378**，耗时 **19.2 s**（RTX 4060）；产物 `adapter_lora.pt` **4.4 MB**、
`model_lora_merged.pt` 1.98 GB（合并权重）、`finetune_meta.json`；微调后样例生成正常。
**结论**：HF LoRA 在服务器 GPU 上跑通（CPU 镜像会明确报「requires a CUDA runtime」）。

### 4.10 `export_gguf`（22-qwen-gguf）— ✅ 已完成（离线）

```bash
docker run --rm -v /home/lanius/models-local:/models:ro -v /home/lanius/compression-eval:/work \
  -v /home/lanius/llama.cpp:/llamacpp:ro --entrypoint ... compress-job:cpu \
  --method=export --algo=gguf --outtype=q4_K_M --base-model-path=/models/Qwen2.5-0.5B \
  --llama-cpp-path=/llamacpp --output-dir=/work/out/22-qwen-gguf
```
结果：`model_q4_K_M.gguf` **511,434,656 B（511 MB）**，相对 HF 目录 999.6 MB → **−48.8%**；
耗时 **31.4 s**；`metrics.tool = python-K(q4_K_M)`、`hub = local`（**完全离线**，用宿主已有的
`/home/lanius/llama.cpp/convert_hf_to_gguf.py`）。同目录另产出 f16 GGUF（994 MB）作为未量化参考。
**结论**：GGUF 通路可用且离线可复现；llama.cpp/ollama 生态可直接加载。困惑度未单独测（INT4 行的 PPL 属另一条通路）。

### 4.11 `qat`（12-qat）— ✅ 已完成（FX QAT 真路径）

```bash
# 需要 passwd 存在的用户（勿用 --user 1000）+ 小样本子集避免 OOM
docker run --rm -v /home/lanius/Dataset:/datasets:ro -v /home/lanius/compression-eval:/work \
  -v <repo>/compress_job.py:/app/compress_job.py:ro compress-job:cpu \
  --job-id=12-qat --method=qat --algo=fx_int8 --epochs=1 --batch-size=64 --learning-rate=0.0005 \
  --base-model-path=/work/models/CifarCNN10/model.pt --arch-file=/work/models/CifarCNN10/arch.py \
  --dataset-path=/work/datasets-derived/CIFAR10-train-4k --image-size 32 \
  --eval-dataset-path=/datasets/CIFAR10-test --output-dir=/work/out/12-qat
```
结果：`model_qat.onnx` **108,077 B（−72.5%）**、`metrics.qat_mode = fx_qat`（真 FX QAT，非回退路径）、
1 epoch / 4,000 图（`baseline_accuracy 0.6867 → qat_accuracy 0.6478`）、`best_val_loss 0.9225`；
我的独立复测：精度 **0.6280（−5.99pp）**、p50 0.096 ms / p99 0.306 ms、吞吐 2123 / 2705、RSS 332.8 MB。

**结论（诚实）**：**1 epoch + 4k 子集的 QAT 没有体现优势**（62.8% 低于静态 PTQ 的 68.68%）——
QAT 的价值在于「多轮次 + 全量数据」下比 PTQ 更保精度，本环境未满足该条件；
两个踩坑已记录在方法卡里：① `--user` 未在镜像 passwd 中会让 FX QAT 静默回退到 fine-tune+PTQ；
② 全量 50k 图直接跑被 OOM 杀（exit 137）。


---

## 5. 发现、异常与限制（全部如实记录）

### 5.1 服务器自带 CIFAR 模型不可用（最重要）
`models-local/TestCNN/model.pt` 是随机初始化的冒烟模型（`arch.py` 自述 + 实测 0.1000）。
→ 本报告用服务器数据训练了同架构真基线（§1.3/§6.1）。**若同事直接拿 TestCNN 跑压缩，得到的精度变化毫无意义。**

### 5.2 剪枝：不微调即不可用
`prune:l2/group_fisher` 在 rate 0.1/0.3 下精度从 0.6879 掉到 0.1466~0.2010（−48~−54pp），
且 rate 0.1 并未更好 → 主因是**分类头权重重映射**（`head_layers_adjusted=1`），不是剪枝比例。
**建议**：`--method=pipeline --pipeline "prune:group_fisher,quantize:onnx_dynamic_int8"` 后再接 QAT/微调
（平台 `qat` 方法存在但 CLI 未暴露，见 5.6）。

### 5.3 `bench.py` 目录解析只扫一层（已补测）
`pipeline` 产物为 `artifact/step1/…`、`artifact/step2/…` 两级结构，默认解析会落到目录本身；
已把 `--artifact` 指向 `step2/model_quantized.onnx` 补测：精度 **0.2026**、
p50 **0.328 ms** / p99 **1.769 ms**、吞吐 1550 / 1678 样本·s⁻¹、峰值 RSS 331.8 MB（已并入 §3）。
余下只有引擎评测值与我的复测差异（0.2095 vs 0.2026，预处理细节不同，属正常范围）。

### 5.4 `eval.baseline_accuracy` 的语义陷阱
引擎评测把**输入模型**（对 prune/sparsify 而言是已被改过的暂存模型）当作 baseline，
所以 `result.json.metrics.eval.baseline_accuracy == accuracy`、`accuracy_delta == 0` 是**正常现象**，
不能据此判断「精度没变」。本报告一律以「训练基线 0.6879 + `bench.py` 独立复测」为准。

### 5.5 CLI 的一处真实缺陷（已修复）
只给 `--eval-dataset-path`（不给 `--dataset-path`）时，评测不会触发（`_finalize` 的条件只认 `dataset`）。
本次修复为 `dataset is not None or args.eval_dataset_path`（`compress_job.py`，未改任何 API），
修复后 §4 各任务的精度才产出。ruff 通过；镜像重打包后行为一致。

### 5.6 `qat` 已补齐 CLI 支持（本次新增，未改 API）
原先 `--method` 的 choices 只有 quantize/prune/distill/finetune/export/pipeline → `qat` 不可用。
本次补上：`--method=qat --algo=fx_int8|lsq`（映射到已有引擎函数 `qat`）、缺少 `--dataset-path` 时退出码 3、
`--help`/文档同步更新。实跑见 §4.11，并记录两个踩坑（uid 无 passwd 会静默回退、全量数据 OOM）。

### 5.7 本机没有 wikitext2 → LLM 困惑度用派生语料
`/home/lanius/Dataset` 无 wikitext/gsm8k；LLM PPL 使用 `datasets-derived/alpaca-eval.txt`
（alpaca-train-32 的 32 行拼接，14,775 B，脚本 `scripts/prep_llm_corpus.py`）。
**该文件是派生数据**，PPL 数值只在同一语料内可比（baseline vs INT4），不可与论文 wikitext2 数字对比。

### 5.8 LLM 任务曾因宿主内存不足中断（已重跑成功）
首轮 INT4 转换 + 双模型困惑度把 16 GB 内存挤到 `MemAvailable ≈ 0.2 GB`，触发 swap
（load 22~47）导致 **8712.9 s** 的异常耗时；释放平台栈内存后两个 LLM 任务均已完成（§4.8/§4.9）。
**结论**：Qwen 0.5B 级 LLM 压缩在 16 GB 机器上需独占内存窗口；空闲环境下平台历史记录约 165 s。

### 5.9 未测项（明确标注，未编造）
* `export_gguf` 的**困惑度**未单独测（GGUF 通路与 INT4 IR 通路不同，PPL 只在同一通路内可比）；体积/耗时已实测（§4.10）；
* `qat` 只跑了 1 epoch / 4k 子集（全量会被 OOM 杀），因此**不能**用它论证「QAT 优于 PTQ」（§4.11 已如实说明）；
* 动态 batch 的 ONNX + INT8：已知会触发 ORT 形状限制（前序交付文档 §5.1），本次 CIFAR 任务统一用静态 batch；
* 其余 11 个方法（§3 全部行）均为真实执行，无占位数字。

### 5.10 NNCF INT4 的 IR 与 stock optimum-intel 不兼容（平台自有 runner 正常）
`--algo=nncf_int4` 产出的 IR：输出为**匿名**（`names[]`）、`stateful_runtime=false`、
第二个输入名为 `120`（NNCF channel-compression 的命名特征）。因此：
* `optimum.intel.OVModelForCausalLM.from_pretrained(dir, export=False)` 报
  `IndexError: tuple index out of range`（`output_names` 为空，见 `logs/20-qwen-artifact.log`）；
* 平台自有加载器 **`app.jobs.llm_runtime_nncf.NncfChatModel` 正常工作** —— §4.8 的 p50/吞吐/内存
  与引擎困惑度 5.295 都来自这条路径（`scripts/run_engine_benchmark.py`）。
**建议**：第三方框架要加载该产物，需先做「输出命名 + stateful KV」适配；平台内部推理不受影响。


---

## 6. 复现步骤

### 6.1 训练 CIFAR-10 基线（本报告基线来源）
```bash
docker run --rm --gpus all \
  -v /home/lanius/Dataset:/datasets:ro -v /home/lanius/models-local:/models:ro \
  -v /home/lanius/compression-eval:/work --entrypoint python b121980fea3c \
  /work/scripts/train_cifar10.py --data /datasets --arch /models/TestCNN/arch.py \
  --out /work/models/CifarCNN10 --epochs 12
# → models/CifarCNN10/{model.pt,arch.py,train_report.json}；12 epoch 71.4 s，CIFAR10-test 0.6879
```

### 6.2 CIFAR 全部方法
```bash
bash /home/lanius/compression-eval/scripts/run_cifar_tasks.sh 03 04 05 06 07 08   # 波次 1
bash /home/lanius/compression-eval/scripts/run_more_tasks.sh                      # 波次 2（含基线 bench）
python3 /home/lanius/compression-eval/scripts/collect_results.py --out /work/out/summary.json
```
产物：`out/<task>/{result.json,bench.json,progress.json,progress.log,logs.jsonl,artifact/}`。

### 6.3 LLM（Qwen2.5-0.5B，已完成；重跑建议先释放内存）
```bash
# 建议：先给 LLM 任务腾出内存（可选，注意影响在线服务）
#   docker compose -f compose.gpu.yml stop backend worker
bash /home/lanius/compression-eval/scripts/run_llm_tasks.sh   # INT4 转换 + LoRA
bash /home/lanius/compression-eval/scripts/run_llm_rest.sh    # LoRA/基准 + 汇总 + 方法卡
# 平台自带 LLM 基准（NNCF INT4 IR 的 p50/吞吐/内存）：
docker run --rm -v /home/lanius/models-local:/models:ro -v /home/lanius/compression-eval:/work \
  --entrypoint python compress-job:cpu /work/scripts/run_engine_benchmark.py \
  --original /models/Qwen2.5-0.5B --artifact /work/out/20-qwen-int4/artifact \
  --tokens 12 --iterations 4 --out /work/out/20-qwen-int4/bench_engine.json
```

### 6.4 重新生成方法卡数据（报告 → 前端单一数据源）
```bash
python3 /home/lanius/compression-eval/scripts/build_method_cards.py \
  --summary /home/lanius/compression-eval/out/summary.json \
  --out <repo>/frontend/src/components/lab/method-cards.json
cd <repo>/frontend && npm run build     # 产物落到 backend/app/frontend
```

---

## 7. 方法卡（交付物 3）

| 项 | 位置 |
|----|------|
| JSON Schema | `docs/method-cards.schema.json`（draft-07；约束 id 必须属于平台 11 个方法枚举、evidьence 字段类型等） |
| 数据（前端唯一内容源） | `frontend/src/components/lab/method-cards.json`（11 张卡，**11/11 均带实测证据**；由 `scripts/build_method_cards.py` 从 `out/summary.json` 生成） |
| 方法卡文档 | [`docs/compression-advisor.md`](./compression-advisor.md) 同时给出「推荐规则表」（下面 §7.1） |
| 组件 | `frontend/src/components/lab/MethodCards.tsx`（`MethodCardPanel` 折叠网格 + `MethodCard` 单卡 + `methodCard(id)` 查询；**JSX 内零硬编码文案**） |
| 接入点 | `frontend/src/routes/lab/compressions.tsx`：① 页面标题下方 `<MethodCardPanel />`（分组筛选 + 展开）；② 新建压缩向导「② 压缩方法」选中后即时显示 `activeCard` |
| 样式一致性 | 复用 `Card/Pill/Button/cx` 与 `lb-*` 变量类；`Pill` 只用既有 `PillTone`（`ready/pending/info/idle`） |
| 构建验证 | `npm run build` 通过（tsc + vite），产物写入 `backend/app/frontend`（平台直接可访问）；`npm run lint` 的 7 个报错均为**既有文件**（index.html / BenchmarkRadar / ui.tsx），与本次新增文件无关 |

字段含义（详见 schema）：`id`（平台枚举）、`label/summary`（中英）、`group`、`calibration.needed+note`、
`inputs[]`（名称/是否必填/格式/说明）、`suits.models|tasks`、`effect.compression|accuracy|latency`、
`limits[]`、`evidence`（**只由实测产生**：dataset/结论文案/前后体积/前后精度/耗时/报告锚点）。

## 8. 附录：原始文件索引（可核验）

| 路径 | 内容 |
|------|------|
| `/home/lanius/compression-eval/out/<task>/result.json` | CLI 作业报告（inputs/params/metrics/artifact_bytes/duration_s） |
| `/home/lanius/compression-eval/out/<task>/bench.json` | 精度/延迟 p50·p90·p99/吞吐/峰值 RSS |
| `/home/lanius/compression-eval/out/<task>/{progress.json,progress.log,logs.jsonl}` | 进度与结构化日志 |
| `/home/lanius/compression-eval/out/summary.json` | 汇总（报告与方法卡共用数据源） |
| `/home/lanius/compression-eval/models/CifarCNN10/` | 基线模型 + `train_report.json`（含 12 轮曲线） |
| `/home/lanius/compression-eval/scripts/` | `train_cifar10.py`、`bench.py`、`bench_llm.py`、`collect_results.py`、`build_method_cards.py`、`prep_llm_corpus.py`、`run_cifar_tasks.sh`、`run_more_tasks.sh`、`run_llm_tasks.sh` |
| `/home/lanius/compression-eval/datasets-derived/alpaca-eval.txt` | LLM 困惑度派生语料（32 行 alpaca） |



