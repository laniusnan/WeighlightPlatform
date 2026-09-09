# GPU Worker 架构方案（规划稿，不实现假功能）

> 目标：为 xtuner（LoRA/DPO）、mmrazor（NAS/可学习量化）与 lmdeploy（TRT）等
> GPU 型能力提供独立的执行队列与容器，不污染现有 CPU worker。

## 1. 设计要点
- 保持现有 `compose.cpu.yml`（CPU worker 跑 9+2 种方法）不变；
- 新增可选 **GPU worker 服务**（`compose.gpu.yml` + 显式 `deploy.resources.reservations.devices`）；
- Arq 双队列：`arq:queue:default`（CPU）与 `arq:queue:gpu`（GPU 型任务）；
  API 层 `enqueue_job` 依据 method 前缀自动选择队列；
- worker 用相同 `CompressionTask` 表，metrics 统一存 `metrics` 字段 → 前端零改动复用评估/导出。

## 2. 未来 GPU 方法映射
| 方法(拟) | 来源 | 依赖 | 产物 |
|---|---|---|---|
| `finetune_lora` | xtuner | torch+GPU, peft | adapter + merged |
| `dpo_lora` | xtuner | GPU+数据集 | 偏好对齐模型 |
| `gptq_awq` | lmdeploy/lite | GPU 离线校准 | INT4 权重（可与现有 OpenVINO/GGUF 消费） |
| `nas` / `lsq_qat` | mmrazor | GPU 搜索/训练 | 结构化子网 / QAT 模型 |

## 3. 安全与配额
- GPU 容器 `--gpus` 显式分配，禁止默认全量直通；
- 任务仍受用户存储配额约束；上传/下载/评估复用现有 API；
- 队列长度/运行数上报 `utils/workers` 需扩展 `gpu_worker` 心跳键。

## 4. 落地顺序（建议）
1. `compose.gpu.yml` + `gpu worker` 命令 + 队列路由；
2. 从 xtuner 移植最小 `LoRA SFT`（recipe 驱动）；
3. mmrazor NAS/LSQ、lmdeploy TRT 视硬件逐项接入。

## 5. 备注
本文件为架构规划；实际镜像与调度改动在确认 GPU 硬件后实施，并同步
`CompressionMethod`/前端向导/`deployment-*.md`。
