# 压缩评估脚本（本次评估的实际执行工具链）

这些脚本是 `docs/compression-eval-report.md` 里所有数字的**产出工具**，随仓库提交以便复现。
报告正文里写的 `scripts/xxx.py` 即指本目录下的同名文件。

## 挂载约定（脚本内部用的是容器路径）

| 脚本里的路径 | 宿主路径 | 说明 |
|--------------|----------|------|
| `/work` | `/home/lanius/compression-eval` | 评估工作区：`models/`、`out/`、`logs/`、`datasets-derived/` |
| `/datasets` | `/home/lanius/Dataset` | 原始数据集（只读挂载） |
| `/app/compress_job.py` | 仓库根目录的 `compress_job.py` | 单文件 CLI（只读挂载进容器） |

换机器时只要用同样的两个挂载点跑，脚本无需改动（示例见各 `run_*.sh`）。

## 文件说明

| 脚本 | 作用 |
|------|------|
| `train_cifar10.py` | 训练真基线 `CifarCNN10`（12 epoch / 0.6879 / 392,548 B）——服务器自带的 TestCNN 是随机权重（0.1000） |
| `bench.py` | 视觉产物测量：Top-1 精度 + p50/p90/p99 延迟 + b1/b32 吞吐 + 峰值 RSS |
| `bench_llm.py` | LLM 产物测量：离线解码吞吐 + 语料困惑度（PPL） |
| `run_engine_benchmark.py` | 走**平台引擎**的基准（`engine.run_benchmark`，用于 NNCF INT4 的 IR） |
| `prep_llm_corpus.py` | 从 alpaca 派生纯文本评测语料（本机无 wikitext） |
| `collect_results.py` | 汇总 `out/*/result.json` + `bench.json` → `out/summary.json` |
| `build_method_cards.py` | `out/summary.json` → 前端方法卡 `method-cards.json`（单一数据源） |
| `run_cifar_tasks.sh` / `run_more_tasks.sh` | CIFAR 任务批（01–13） |
| `run_llm_tasks.sh` / `run_llm_rest.sh` / `run_qat_gguf.sh` | LLM 任务批（20/21/22）与 QAT/GGUF |
| `run_qat_retry.sh` / `run_qat_retry2.sh` | QAT 排障（passwd 回退、OOM 定位） |
| `run_qat_long.sh` | **QAT 长跑**：10 epoch × 全量 50k（OOM，exit 137）→ 10 epoch × 4k 子集（0.6683） |

## QAT 长跑结论（2026-09-24）

| 运行 | 轮数 / 数据 | 精度 | Δpp | val loss | 产物 | 备注 |
|------|-------------|------|-----|----------|------|------|
| `12-qat` | 1 / 4k | 0.6280 | −5.99 | 0.9225 | 108,077 B | 首次跑通 |
| `12-qat-10ep-4k` | 10 / 4k | **0.6683** | **−1.96** | 0.8093 | 108,077 B | 轮次有明显收益 |
| `12-qat-10ep-full` | 10 / 50k | — | — | — | — | **OOM 被杀（exit 137）** |

→ 轮次是必要条件，但 10 epoch / 4k 仍未追平静态 PTQ（68.68%）；
要给出「QAT > PTQ」的可信结论需要全量数据，本机 16 GB 内存做不到，报告如实标注为未证。

## 相关测试

推荐规则引擎的单元测试（无需浏览器/后端）：

```bash
cd frontend && npm run test:unit     # = bun test tests/unit
```
