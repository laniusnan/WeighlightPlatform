#!/usr/bin/env python3
"""compress_job.py — 模型压缩「作业」入口：单文件 CLI（可被调度器直接调用）。

与 `train.py` 同构：调度器把作业上下文与输入路径当作命令行参数传进来即可，
本脚本在**本进程内**完成压缩（不需要数据库、Redis，也不需要平台 API），把产物写入
`--output-dir`，并输出结构化 JSON 日志与 `result.json`。

用法（对齐模板变量）
--------------------
    python /app/compress_job.py \\
      --job-id={{job.id}} --tenant-id={{job.tenant_id}} --trace-id={{job.trace_id}} \\
      --dataset-path={{inputs.dataset_path}} --base-model-path={{inputs.base_model_path}} \\
      --output-dir={{inputs.output_path}} \\
      --method=quantize --algo=nncf_int4 --bits=4 --group-size=128 \\
      --epochs=80 --batch-size=16 --learning-rate=0.001

    # 微调（LoRA/DPO）
    python /app/compress_job.py --job-id=42 --tenant-id=t1 --trace-id=abc \\
      --base-model-path=/models/qwen2.5-0.5b --dataset-path=/datasets/alpaca.jsonl \\
      --output-dir=/outputs/42 --method=finetune --algo=lora --epochs=1 --learning-rate=0.0003

    # 组合流水线（先剪枝再量化，前一步产物自动作为后一步输入）
    python /app/compress_job.py --job-id=43 --trace-id=def --base-model-path=/models/cnn.pt \\
      --dataset-path=/datasets/CIFAR10-train --output-dir=/outputs/43 \\
      --method=pipeline --pipeline="prune:group_fisher,quantize:onnx_dynamic_int8" --rate=0.3

输入全部显式（没有隐藏默认）
------------------------------
所有输入都在「输入 / 输出」参数组里，旁挂文件不会被默默猜出来：

    # ONNX 外部权重：必须显式给出文件或所在目录（缺失 → 退出码 3）
    python /app/compress_job.py --base-model-path=/models/model.onnx \\
      --external-data-path=/models/model.onnx.data --output-dir=/out/1 \\
      --method=quantize --algo=onnx_dynamic_int8

    # 只给目录名，三个相对路径都以 --model-dir 为基准
    python /app/compress_job.py --model-dir=/models --base-model-path=model.onnx \\
      --external-data-path=model.onnx.data --arch-file=arch.py --output-dir=/out/1 \\
      --method=quantize --algo=onnx_dynamic_int8

    # 兼容旧行为（显式开启后，外部权重才按「与 .onnx 同目录」解析）
    --allow-implicit-external-data

    # 另有：--llama-cpp-path（离线 GGUF 工具目录）、--hf-cache-dir（HF 下载缓存）
    # 想核对一次运行到底用了哪些输入：`--dry-run` 会打印 inputs 清单；
    # 每次作业的 result.json 里也有 inputs（含 source=cli/env/default）。

方法 / 算法矩阵
---------------
| --method   | --algo                                                                  | 说明                  |
|------------|-------------------------------------------------------------------------|-----------------------|
| quantize   | onnx_dynamic_int8 · onnx_static_int8 · openvino_int8 · nncf_int4 · gguf  | 量化（INT8/INT4/GGUF） |
| prune      | l1 · l2 · bnslim · autoslim · group_fisher · dmcp · sparsify             | 剪枝 / 稀疏化          |
| distill    | logits · features                                                        | 知识蒸馏（KD/FitNets） |
| finetune   | lora · qlora · dpo                                                       | LoRA SFT / DPO        |
| export     | onnx · onnx_int8 · openvino · gguf                                       | 格式导出               |
| pipeline   | `"step[:algo],step[:algo]"`                                              | 多步串联               |

产物（`--output-dir` 下）
------------------------
    artifact/       压缩产物（model.onnx / model.xml / model.gguf / model_pruned.pt …）
    result.json     作业结果（job/tenant/trace、方法、参数、metrics、产物路径、耗时）
    metrics.json    纯指标（与平台 task.metrics 同结构）
    logs.jsonl      结构化日志（每行一条 JSON，可直接投喂采集器）

退出码
------
    0 成功 · 2 参数错误 · 3 输入缺失/不可写 · 4 方法或算法不支持 · 5 压缩执行失败
    6 超时 · 7 缺少可选依赖 · 130 用户中断

环境变量（前缀 `COMPRESS_JOB_`，可覆盖 CLI 默认值）
--------------------------------------------------
    COMPRESS_JOB_METHOD / _ALGO / _EPOCHS / _BATCH_SIZE / _LEARNING_RATE / _OUTPUT_DIR …
    COMPRESS_JOB_JSON_LOGS=0              关闭结构化日志（改人类可读）
    MODEL_STORAGE_PATH=<output-dir>       本地模式自动指向输出目录（无需真实平台存储）
"""
from __future__ import annotations

import argparse
import json
import os
import random
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

ENV_PREFIX = "COMPRESS_JOB_"


def _drop_shadowing_self_dir() -> None:
    """把「脚本所在目录」从 ``sys.path`` 摘掉，前提是它遮蔽标准库 `compression`。

    ``python compress_job.py`` 会把脚本目录放在 ``sys.path[0]``；仓库根目录还有一个同名
    启动器 ``compression.py``（以及可选的 ``compression/`` 包），于是 ``torch`` 内部
    ``gzip`` → ``compression._common`` 会 import 错对象，报::

        ModuleNotFoundError: No module named 'compression._common'; 'compression' is not a package

    本文件已作为 ``__main__`` 载入，摘掉自己的目录不影响自身；后端代码由
    :func:`_ensure_backend_on_path` 显式加入 ``sys.path``，也不受影响。
    与仓库内 ``compression.py`` 的处理方式保持一致。
    """
    try:
        here = Path(__file__).resolve().parent
    except OSError:  # pragma: no cover - 极端命名空间环境
        return
    if not ((here / "compression.py").is_file() or (here / "compression").is_dir()):
        return
    keep: list[str] = []
    for entry in sys.path:
        try:
            resolved = Path(entry or ".").resolve()
        except OSError:  # pragma: no cover
            keep.append(entry)
            continue
        if resolved != here:
            keep.append(entry)
    sys.path[:] = keep


_drop_shadowing_self_dir()

# --------------------------------------------------------------------------- #
# 可选依赖：服务模式（--serve）需要 fastapi + uvicorn。
# 模块级守卫式导入，缺失时不影响 CLI 的其它功能。
# 注：名称必须在模块命名空间里，FastAPI 才能在注册路由时解析函数注解。
# --------------------------------------------------------------------------- #
try:  # pragma: no cover - 取决于环境是否安装 Web 依赖
    import uvicorn
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse

    _SERVE_IMPORT_ERROR: Exception | None = None
except ModuleNotFoundError as _serve_exc:  # pragma: no cover
    FastAPI = HTTPException = Request = JSONResponse = uvicorn = None  # type: ignore[assignment]
    _SERVE_IMPORT_ERROR = _serve_exc


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(f"{ENV_PREFIX}{name}")
    return default if raw is None else raw.strip().lower() in {"1", "true", "yes", "on"}


def env_value(name: str, default: str | None = None) -> str | None:
    raw = os.environ.get(f"{ENV_PREFIX}{name}")
    return default if raw in (None, "") else raw


def _bootstrap_settings(output_dir: Path | None) -> None:
    """平台设置有三个必填项；本脚本不访问数据库，给占位值即可独立运行。"""
    os.environ.setdefault("SECRET_KEY", "compress-job-standalone")
    os.environ.setdefault("PROJECT_NAME", "compress-job")
    os.environ.setdefault("DATABASE_URL", "postgresql://compress:compress@127.0.0.1:5432/compress")
    os.environ.setdefault("FIRST_SUPERUSER", "compress-job@example.com")
    os.environ.setdefault("FIRST_SUPERUSER_PASSWORD", "compress-job")
    os.environ.setdefault("MODEL_STORAGE_PATH", str(output_dir or Path.cwd()))


def _ensure_backend_on_path() -> None:
    """仓库根目录 / backend/ / 平台镜像(/app/backend) 三种布局都能 import app.*。"""
    here = Path(__file__).resolve().parent
    for candidate in (here / "backend", here, Path("/app/backend"), Path("/app")):
        if (candidate / "app" / "jobs" / "compression.py").is_file():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            return
    raise JobError(
        "找不到 app/jobs/compression.py — 请把本文件放在仓库根目录、backend/ 内，"
        "或使用压缩镜像（/app/backend/app/...）",
        3,
    )


@dataclass
class JobContext:
    job_id: str = "local"
    tenant_id: str = "local"
    trace_id: str = ""
    json_logs: bool = True
    log_file: Path | None = None
    records: list[dict[str, Any]] = field(default_factory=list)
    inputs: list[dict[str, Any]] = field(default_factory=list)
    progress: Any = None  # ProgressTracker（在 run_local 里创建，失败路径用它收尾）

    def log(self, event: str, message: str = "", level: str = "info", **extra: Any) -> None:
        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "level": level,
            "event": event,
            "job_id": self.job_id,
            "tenant_id": self.tenant_id,
            "trace_id": self.trace_id,
            "message": message,
            **extra,
        }
        self.records.append(record)
        if self.json_logs:
            print(json.dumps(record, ensure_ascii=False), flush=True)
        else:
            print(f"[{record['ts']}] {level.upper():<5} {event}: {message} {extra or ''}".rstrip(),
                  flush=True)
        if self.log_file is not None:
            try:
                with self.log_file.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            except OSError:  # pragma: no cover - best effort
                pass


@dataclass
class ProgressTracker:
    """把作业进度落盘到 `--output-dir` 下，并保证**每 interval%（默认 10%）**记录一次。

    产出两个文件（都在 `--output-dir` 根下，与 `artifact/` 同级）：

    | 文件 | 内容 |
    |------|------|
    | `progress.json` | 最新状态快照：`status/step/steps_total/step_label/percent/message/elapsed_s/artifact` |
    | `progress.log`  | 追加式里程碑日志：每 10% 一行（`时间 30% step=1/2 … 消息`），可 `tail -f` |

    同时每个里程碑会写一条 `progress_milestone` 结构化日志（stdout + `logs.jsonl`）。
    """

    ctx: JobContext
    root: Path
    interval: int = 10
    steps_total: int = 1
    enabled: bool = True
    started: float = field(default_factory=time.time)
    step: int = 0
    step_label: str = ""
    percent: int = 0
    message: str = ""
    status: str = "running"
    artifact: str | None = None
    _next_milestone: int = 0
    _last_write: float = field(default=0.0, repr=False)

    # ---- 路径 ----
    @property
    def json_path(self) -> Path:
        return self.root / "progress.json"

    @property
    def log_path(self) -> Path:
        return self.root / "progress.log"

    # ---- 对外 API ----
    def start_step(self, index: int, total: int, label: str) -> None:
        self.step, self.steps_total, self.step_label = index, total, label
        self.percent, self.message = 0, f"开始 {label}"
        self._emit(0)
        self._next_milestone = self.interval  # 0% 已在上面记录，避免重复
        self._write(force=True)

    def update(self, percent: float, message: str = "") -> None:
        """引擎进度回调：补齐里程碑 + 刷新快照。"""
        reported = int(max(0, min(100, round(float(percent)))))
        if reported > self.percent:
            self.percent = reported
        if message:
            self.message = message
        while self._next_milestone <= (reported // self.interval) * self.interval:
            self._emit(self._next_milestone)
            self._next_milestone += self.interval
        self._write()

    def finish(self, status: str, artifact: str | None = None, message: str = "") -> None:
        """收尾（成功路径）：补齐剩余的里程碑直到 100%，落盘最终状态。"""
        while self._next_milestone <= 100:
            self._emit(self._next_milestone)
            self._next_milestone += self.interval
        self.percent = 100
        self.status = status
        self.artifact = artifact
        self.message = message or ("完成" if status == "succeeded" else f"作业{status}")
        self._write(force=True)

    def fail(self, message: str = "", status: str = "failed") -> None:
        """收尾（失败/中断路径）：只落盘当前进度与错误信息，不补里程碑。"""
        self.status = status
        self.message = message or f"作业{status}"
        try:
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} {self.percent:3d}% "
                    f"step={self.step}/{self.steps_total} {self.step_label} "
                    f"[{status}] {self.message}".rstrip() + "\n"
                )
        except OSError:  # pragma: no cover - 只读输出目录
            pass
        self._write(force=True)

    # ---- 内部 ----
    def _emit(self, percent: int) -> None:
        if not self.enabled:
            return
        self.percent = max(self.percent, percent)
        self.ctx.log("progress_milestone", f"{percent}% {self.message}".strip(),
                     step=str(self.step), percent=percent)
        try:
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} {percent:3d}% "
                    f"step={self.step}/{self.steps_total} {self.step_label} {self.message}".rstrip()
                    + "\n"
                )
        except OSError:  # pragma: no cover - 只读输出目录
            pass

    def _write(self, force: bool = False) -> None:
        if not self.enabled:
            return
        now = time.time()
        if not force and now - self._last_write < 1.0:
            return
        self._last_write = now
        snapshot = {
            "job_id": self.ctx.job_id,
            "tenant_id": self.ctx.tenant_id,
            "trace_id": self.ctx.trace_id,
            "status": self.status,
            "step": self.step,
            "steps_total": self.steps_total,
            "step_label": self.step_label,
            "percent": self.percent,
            "interval": self.interval,
            "message": self.message,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(self.started)),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "elapsed_s": round(now - self.started, 3),
            "artifact": self.artifact,
        }
        tmp = self.json_path.with_suffix(".json.tmp")
        try:
            tmp.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(self.json_path)
        except OSError:  # pragma: no cover - 只读输出目录
            pass


@dataclass
class ModelShim:
    """平台的作业函数只需要这几个属性（无需数据库中的 Model 行）。"""

    file_path: str
    name: str
    framework: str = "pytorch"
    source_url: str | None = None
    size_bytes: int = 0
    version: int = 1
    id: str = "local"


@dataclass
class JobResult:
    status: str
    method: str
    algo: str | None
    params: dict[str, Any]
    metrics: dict[str, Any]
    artifact: str | None
    artifact_bytes: int
    duration_s: float
    steps: list[dict[str, Any]] = field(default_factory=list)


class JobError(Exception):
    def __init__(self, message: str, code: int = 5) -> None:
        super().__init__(message)
        self.code = code


def _require(condition: bool, message: str, code: int = 3) -> None:
    if not condition:
        raise JobError(message, code)


# --------------------------------------------------------------------------- #
# 3. 命令行：作业上下文 + 输入输出 + 方法/算法 + 通用超参
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="compress_job.py",
        description="模型压缩作业入口（单文件，可被调度器直接调用）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python /app/compress_job.py --job-id={{job.id}} --tenant-id={{job.tenant_id}} "
            "--trace-id={{job.trace_id}} \\\n"
            "      --base-model-path={{inputs.base_model_path}} "
            "--dataset-path={{inputs.dataset_path}} --output-dir={{inputs.output_path}} \\\n"
            "      --method=quantize --algo=nncf_int4 --bits=4 --group-size=128\n"
            "  python /app/compress_job.py --job-id=7 --base-model-path=/m/cnn.pt "
            "--output-dir=/out/7 --method=prune --algo=l2 --rate=0.3\n"
        ),
    )

    ctx = parser.add_argument_group("作业上下文（调度器注入）")
    ctx.add_argument("--job-id", default=env_value("JOB_ID", "local"), help="作业 ID")
    ctx.add_argument("--tenant-id", default=env_value("TENANT_ID", "local"), help="租户 ID")
    ctx.add_argument("--trace-id", default=env_value("TRACE_ID", ""), help="链路追踪 ID")

    io = parser.add_argument_group("输入 / 输出（所有输入都必须显式给出，见 --dry-run 的 inputs 清单）")
    io.add_argument("--base-model-path", default=env_value("BASE_MODEL_PATH"),
                    help="待压缩模型：.pt/.pth 检查点、.onnx、或 HuggingFace 模型目录；"
                         "配合 --model-dir 时可只给文件名")
    io.add_argument("--model-dir", default=env_value("MODEL_DIR"),
                    help="模型目录：--base-model-path 为文件名/相对路径时的基准目录；"
                         "也是 arch.py、外部权重文件的默认查找目录")
    io.add_argument("--external-data-path", default=env_value("EXTERNAL_DATA_PATH"),
                    help="ONNX 外部权重（<name>.onnx.data）的显式地址：给文件（/m/model.onnx.data）"
                         "或给目录（/m）。外部权重模型的必填项——不默认同目录，缺失即报错退出 3")
    io.add_argument("--allow-implicit-external-data", action=argparse.BooleanOptionalAction,
                    default=env_flag("ALLOW_IMPLICIT_EXTERNAL_DATA", False),
                    help="兼容开关：允许按“与 .onnx 同目录”隐式解析外部权重（默认关）")
    io.add_argument("--arch-file", default=env_value("ARCH_FILE"),
                    help="PyTorch 架构 .py（模型类或 build_model(config) 工厂）；"
                         "缺省按 --model-dir/--base-model-path 同目录的 arch.py 查找")
    io.add_argument("--dataset-path", default=env_value("DATASET_PATH"),
                    help="数据集：按类图片目录、.jsonl/.csv/.xlsx 指令集、或含 .txt 的语料目录")
    io.add_argument("--eval-dataset-path", default=env_value("EVAL_DATASET_PATH"),
                    help="可选：评估/校准数据集（缺省复用 --dataset-path）")
    io.add_argument("--output-dir", default=env_value("OUTPUT_DIR", "outputs"),
                    help="输出目录（产物写 artifact/，报告写 result.json/metrics.json）")
    io.add_argument("--llama-cpp-path", default=env_value("LLAMA_CPP_PATH"),
                    help="离线 GGUF 转换工具目录（等价环境变量 LLAMA_CPP_PATH；镜像内为 /opt/llama.cpp；"
                         "缺失时会尝试联网拉取工具）")
    io.add_argument("--hf-cache-dir", default=env_value("HF_CACHE_DIR") or os.environ.get("HF_HOME"),
                    help="HuggingFace/ModelScope 下载缓存目录（导出为 HF_HOME，用于离线复用已下载模型）")

    method = parser.add_argument_group("方法 / 算法（必填）")
    method.add_argument("--method", default=env_value("METHOD"),
                        choices=("quantize", "prune", "distill", "finetune", "export", "qat", "pipeline"),
                        help="压缩族（qat = 量化感知训练）")
    method.add_argument("--algo", default=env_value("ALGO"),
                        help="族内算法：quantize=[onnx_dynamic_int8|onnx_static_int8|openvino_int8|"
                             "nncf_int4|gguf] prune=[l1|l2|bnslim|autoslim|group_fisher|dmcp|sparsify] "
                             "distill=[logits|features] finetune=[lora|qlora|dpo] "
                             "export=[onnx|onnx_int8|openvino|gguf] qat=[fx_int8|lsq]")
    method.add_argument("--pipeline", default=env_value("PIPELINE"),
                        help='--method=pipeline 时使用：如 "prune:group_fisher,quantize:onnx_dynamic_int8"')

    hyper = parser.add_argument_group("通用超参（未用到的参数会记录在 result.json.params）")
    hyper.add_argument("--epochs", type=int, default=None, help="训练轮数（QAT / 蒸馏 / 微调）")
    hyper.add_argument("--batch-size", type=int, default=None, help="批大小")
    hyper.add_argument("--learning-rate", type=float, default=None, help="学习率（映射到平台 lr）")
    hyper.add_argument("--rate", type=float, default=None, help="剪枝率 0<rate<1（剪枝/稀疏化）")
    hyper.add_argument("--bits", type=int, default=None, help="量化位宽 4|8（INT4/INT8）")
    hyper.add_argument("--group-size", type=int, default=None, help="INT4 分组大小 32/64/128/256")
    hyper.add_argument("--precision", default=None, help="OpenVINO 精度 int8|fp16")
    hyper.add_argument("--outtype", default=None, help="GGUF 输出类型 f16|q8_0|q4_0|q4_K_M|q6_K…")
    hyper.add_argument("--temperature", type=float, default=None, help="蒸馏温度 T")
    hyper.add_argument("--alpha", type=float, default=None, help="蒸馏 KD 权重 α")
    hyper.add_argument("--width-mult", type=float, default=None, help="蒸馏学生宽度系数")
    hyper.add_argument("--lora-rank", type=int, default=None, help="LoRA rank")
    hyper.add_argument("--dpo-beta", type=float, default=None, help="DPO β")
    hyper.add_argument("--hf-max-len", type=int, default=None, help="HF 分词最大长度")
    hyper.add_argument("--max-samples", type=int, default=None,
                       help="限制样本数（校准样本 / Fisher 图片上限）")
    hyper.add_argument("--input-shape", default=None, help="输入形状，如 1x3x32x32")
    hyper.add_argument("--image-size", type=int, default=None, help="图片边长（分类系）")
    hyper.add_argument("--opset", type=int, default=None, help="ONNX opset")
    hyper.add_argument("--model-kind", default=None, choices=(None, "cnn", "nlp", "yolo", "tabular"),
                       help="模型类型（YOLO 导出走 ultralytics 分支；tabular=表格模型）")
    hyper.add_argument("--imgsz", type=int, default=None, help="YOLO 导出输入尺寸")
    hyper.add_argument("--dynamic-axes", action=argparse.BooleanOptionalAction, default=None,
                       help="ONNX 使用动态 batch 轴")
    hyper.add_argument("--seed", type=int, default=None, help="随机种子（python/numpy/torch）")
    hyper.add_argument("--device", default=None, choices=(None, "cpu", "cuda"), help="设备偏好")
    hyper.add_argument("--stage", default=None, help="微调阶段：sft|dpo（不填由 --algo 推断）")
    hyper.add_argument("--data", default=None, help="内联训练数据：每行 `问题 => 回答`")

    run = parser.add_argument_group("执行控制")
    run.add_argument("--mode", default=env_value("MODE", "local"), choices=("local", "api"),
                     help="local=进程内执行（默认）；api=提交到平台 API 并等待")
    run.add_argument("--api-base", default=env_value("API_BASE"), help="--mode api 时的平台地址")
    run.add_argument("--api-token", default=env_value("API_TOKEN"), help="--mode api 时的 Bearer token")
    run.add_argument("--platform-model-id", default=env_value("PLATFORM_MODEL_ID"),
                     help="--mode api 时的平台模型 ID/名称")
    run.add_argument("--timeout", type=float, default=float(env_value("TIMEOUT", "0") or 0),
                     help="软超时（秒），0=不限；超时退出码 6")
    run.add_argument("--progress-file", action=argparse.BooleanOptionalAction,
                     default=env_flag("PROGRESS_FILE", True),
                     help="在 --output-dir 下写进度文件（progress.json 快照 + progress.log 里程碑，默认开）")
    run.add_argument("--progress-interval", type=int,
                     default=int(env_value("PROGRESS_INTERVAL", "10") or 10),
                     help="进度落盘/记录粒度（百分比，默认 10）：每跨过该档位写一行里程碑")
    run.add_argument("--json-logs", action=argparse.BooleanOptionalAction,
                     default=env_flag("JSON_LOGS", True), help="结构化 JSON 日志（默认开）")
    run.add_argument("--dry-run", action="store_true", help="只打印解析后的方法与参数，不执行")
    run.add_argument("--serve", action="store_true", help="以 HTTP 服务方式启动（POST /compress）")
    run.add_argument("--host", default=env_value("HOST", "0.0.0.0"), help="--serve 监听地址")
    run.add_argument("--port", type=int, default=int(env_value("PORT", "8080") or 8080),
                     help="--serve 端口")
    return parser


def parse_shape(value: str) -> list[int]:
    text = value.strip()
    if text.startswith("["):
        return [int(dim) for dim in json.loads(text)]
    return [int(part) for part in text.replace("x", ",").replace("X", ",").split(",") if part.strip()]


def seed_everything(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:  # noqa: BLE001 - numpy is optional here
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:  # noqa: BLE001 - torch is optional here
        pass


# --------------------------------------------------------------------------- #
# 4. CLI → 平台方法/参数映射
# --------------------------------------------------------------------------- #
# (method, algo) -> (平台方法, 固定参数)
ALGO_MAP: dict[tuple[str, str], tuple[str, dict[str, Any]]] = {
    # 量化
    ("quantize", "onnx_dynamic_int8"): ("quantize_onnx", {}),
    ("quantize", "onnx_int4"): ("quantize_onnx_int4", {}),
    ("quantize", "onnx_int4_wq"): ("quantize_onnx_int4", {}),
    ("quantize", "onnx_static_int8"): ("quantize_onnx_static", {}),
    ("quantize", "openvino_int8"): ("openvino_int8", {"precision": "int8"}),
    ("quantize", "nncf_int4"): ("llm_quantize", {"engine": "nncf"}),
    ("quantize", "gguf"): ("export_gguf", {}),
    # 剪枝 / 稀疏化
    ("prune", "l1"): ("prune", {"algorithm": "l1"}),
    ("prune", "l2"): ("prune", {"algorithm": "l2"}),
    ("prune", "bnslim"): ("prune", {"algorithm": "bnslim"}),
    ("prune", "autoslim"): ("prune", {"algorithm": "autoslim"}),
    ("prune", "group_fisher"): ("prune", {"algorithm": "group_fisher"}),
    ("prune", "dmcp"): ("prune", {"algorithm": "dmcp"}),
    ("prune", "sparsify"): ("sparsify", {}),
    # 蒸馏
    ("distill", "logits"): ("distill", {"method": "logits"}),
    ("distill", "features"): ("distill", {"method": "features"}),
    # 微调
    ("finetune", "lora"): ("llm_finetune", {"stage": "sft"}),
    ("finetune", "qlora"): ("llm_finetune", {"stage": "sft"}),
    ("finetune", "dpo"): ("llm_finetune", {"stage": "dpo"}),
    # 导出
    ("export", "onnx"): ("export_onnx", {}),
    ("export", "onnx_int8"): ("quantize_onnx", {}),
    ("export", "openvino"): ("openvino_int8", {"precision": "fp16"}),
    ("export", "gguf"): ("export_gguf", {}),
    # 量化感知训练（FX QAT / LSQ）：需要 PyTorch 检查点 + arch.py + 按类图片目录
    ("qat", "fx_int8"): ("qat", {}),
    ("qat", "lsq"): ("qat", {"scheme": "lsq"}),
}

DATA_DRIVEN_PRUNE = {"autoslim", "group_fisher", "dmcp"}
CALIBRATION_METHODS = {"quantize_onnx_static", "openvino_int8"}
# 这些方法把 torch 模型导出成张量格式，输入形状决定后续推理/评估尺寸
IMAGE_INPUT_METHODS = {
    "export_onnx", "quantize_onnx", "quantize_onnx_static", "openvino_int8",
    "prune", "qat", "distill", "sparsify",
}
GGUF_OUTTYPES = {
    "f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "q5_0", "q5_1",
    "q2_K", "q3_K", "q4_K", "q5_K", "q6_K",
    "q4_K_S", "q4_K_M", "q5_K_S", "q5_K_M",
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def resolve_step(method: str, algo: str | None) -> tuple[str, dict[str, Any]]:
    """(method, algo) → (平台方法, 固定参数)；未知组合抛 JobError(4)。"""
    if not algo:
        raise JobError(f"--method={method} 需要同时指定 --algo", 2)
    key = (method, algo.lower())
    if key not in ALGO_MAP:
        available = ", ".join(sorted(a for m, a in ALGO_MAP if m == method)) or "(无)"
        raise JobError(f"不支持的方法组合: {method}:{algo}；可选: {available}", 4)
    platform_method, fixed = ALGO_MAP[key]
    return platform_method, dict(fixed)


def build_params(
    args: argparse.Namespace, method: str, algo: str, dataset: Path | None
) -> tuple[str, dict[str, Any]]:
    """CLI 参数 → 平台 params。（平台的 dataset_id/data_url/data_dir 三者取其一）"""
    platform_method, params = resolve_step(method, algo)
    algo = algo.lower()

    scalar = {
        "rate": args.rate,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.learning_rate,
        "opset": args.opset,
        "image_size": args.image_size,
        "temperature": args.temperature,
        "alpha": args.alpha,
        "width_mult": args.width_mult,
        "lora_rank": args.lora_rank,
        "hf_max_len": args.hf_max_len,
        "dpo_beta": args.dpo_beta,
        "stage": args.stage,
        "data": args.data,
        "model_kind": args.model_kind,
        "imgsz": args.imgsz,
        "dynamic_axes": args.dynamic_axes,
    }
    params.update({k: v for k, v in scalar.items() if v is not None})
    if args.input_shape:
        params["input_shape"] = parse_shape(args.input_shape)
    elif args.image_size and platform_method in IMAGE_INPUT_METHODS:
        # 让导出的 ONNX/IR 输入尺寸与 --image-size 一致（否则平台默认 224，
        # 与后续 --image-size 的评估数据不兼容）
        params["input_shape"] = [1, 3, args.image_size, args.image_size]

    if dataset is not None:
        data_ref = dataset
        if platform_method == "llm_finetune" and dataset.is_file():
            # 平台的微调数据解析器按目录扫描（csv/xlsx/jsonl…），传文件会解析不到
            data_ref = dataset.parent
            ctx_hint = f"finetune 数据集按目录解析：{data_ref}"
        if platform_method in CALIBRATION_METHODS:
            params["calib_data_dir"] = str(data_ref)
        else:
            params["data_dir"] = str(data_ref)
        if platform_method == "llm_finetune" and dataset.is_file():
            params["_data_note"] = ctx_hint

    if platform_method == "llm_quantize":
        params.setdefault("bits", args.bits if args.bits is not None else 4)
        params.setdefault("group_size", args.group_size if args.group_size is not None else 128)
        params.setdefault("sym", "true")
    if platform_method == "quantize_onnx_int4":
        params.setdefault("block_size", args.group_size if args.group_size else 32)
        params.setdefault("bits", args.bits if args.bits is not None else 4)
    if platform_method in CALIBRATION_METHODS and args.max_samples is not None:
        params["calibration_samples"] = args.max_samples
    if platform_method == "openvino_int8" and args.precision:
        params["precision"] = args.precision
    if platform_method == "export_gguf":
        outtype = (args.outtype or "f16").strip()
        if outtype not in GGUF_OUTTYPES:
            raise JobError(
                f"不支持的 GGUF outtype: {outtype}；可选: {', '.join(sorted(GGUF_OUTTYPES))}", 2
            )
        params["outtype"] = outtype
    if platform_method == "prune" and algo in DATA_DRIVEN_PRUNE and dataset is None:
        raise JobError(f"prune:{algo} 需要 --dataset-path（按类图片目录）", 3)
    if platform_method == "qat" and dataset is None:
        raise JobError("qat 需要 --dataset-path（按类图片目录，用于 QAT 微调/校准）", 3)

    eval_path = Path(args.eval_dataset_path) if args.eval_dataset_path else dataset
    if eval_path is not None and eval_path.exists():
        if eval_path.is_dir():
            has_images = any(p.suffix.lower() in IMAGE_SUFFIXES for p in eval_path.rglob("*"))
            first_text = next((p for p in sorted(eval_path.rglob("*.txt"))), None)
            if has_images:
                params["eval"] = {
                    "type": "classification",
                    "data_dir": str(eval_path),
                    "image_size": args.image_size or 224,
                }
            elif first_text is not None:
                params["eval"] = {"type": "perplexity", "text_file": str(first_text)}
        elif eval_path.suffix.lower() == ".txt":
            params["eval"] = {"type": "perplexity", "text_file": str(eval_path)}

    params["_cli"] = {
        k: v for k, v in vars(args).items()
        if k in {
            "method", "algo", "epochs", "batch_size", "learning_rate", "rate", "bits",
            "group_size", "precision", "outtype", "temperature", "alpha", "width_mult",
            "lora_rank", "dpo_beta", "hf_max_len", "max_samples", "input_shape",
            "image_size", "opset", "seed", "stage", "device", "model_kind", "imgsz",
            "dynamic_axes",
        } and v is not None
    }
    return platform_method, params


# --------------------------------------------------------------------------- #
# 5. 本地执行引擎（进程内调用平台作业函数，无 DB / Redis）
# --------------------------------------------------------------------------- #
# 记录当前镜像/环境缺失的平台方法（只影响对应方法，不影响其它能力）
_MISSING_JOB_FUNCTIONS: list[str] = []


def _job_functions() -> dict[str, Callable[..., Any]]:
    """返回「平台方法名 → 作业函数」映射。

    基础镜像可能早于 CLI（缺新方法）：用 ``getattr`` 过滤，缺失的方法在真正被请求时
    才报「镜像能力不足」，不影响其它方法。
    """
    from app.jobs import compression as engine

    wanted = {
        "export_onnx": "export_onnx",
        "quantize_onnx": "quantize_onnx",
        "quantize_onnx_int4": "quantize_onnx_int4",
        "quantize_onnx_static": "quantize_onnx_static",
        "openvino_int8": "openvino_int8",
        "qat": "qat",
        "sparsify": "sparsify",
        "export_gguf": "export_gguf",
        "prune": "prune",
        "distill": "distill",
        "llm_quantize": "llm_quantize",
        "llm_finetune": "llm_finetune",
    }
    available: dict[str, Callable[..., Any]] = {}
    missing: list[str] = []
    for name in wanted:
        function = getattr(engine, name, None)
        if callable(function):
            available[name] = function
        else:
            missing.append(name)
    _MISSING_JOB_FUNCTIONS.clear()
    _MISSING_JOB_FUNCTIONS.extend(missing)
    return available


def _resolve_artifact(path: Path) -> Path:
    from app.jobs.compression import resolve_artifact_file

    return resolve_artifact_file(path)


def _size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if path.is_dir():
        return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    return 0


def _cli_supplied(argv: list[str]) -> set[str]:
    """从原始 argv 里提取用户显式给出的参数名（去掉 --，用于标注输入来源）。"""
    return {item.split("=", 1)[0].lstrip("-") for item in argv if item.startswith("--")}


def _arg_source(args: argparse.Namespace, name: str) -> str:
    """输入来源：cli（命令行显式给出）/ env（环境变量）/ default（代码默认）。"""
    if name in (getattr(args, "_supplied", None) or set()):
        return "cli"
    env_key = "COMPRESS_JOB_" + name.upper().replace("-", "_")
    if os.environ.get(env_key) or (name == "hf-cache-dir" and os.environ.get("HF_HOME")):
        return "env"
    return "default"


def _input_row(
    args: argparse.Namespace,
    name: str,
    value: Any,
    *,
    required: bool = False,
    note: str = "",
    resolved: str | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "name": "--" + name,
        "required": required,
        "provided": value not in (None, ""),
        "source": _arg_source(args, name),
        "value": str(value) if value not in (None, "") else None,
    }
    if resolved is not None:
        row["resolved"] = resolved
    if note:
        row["note"] = note
    return row


def _input_manifest(
    args: argparse.Namespace,
    dataset: Path | None = None,
    *,
    stage: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """列出本次作业的**全部输入**：命令行参数名、来源（cli/env/default）、解析结果。

    用于 `--dry-run` 与 `result.json.inputs`，避免出现「文档里没写、代码里默认」的隐藏输入。
    """
    stage = stage or {}
    rows = [
        _input_row(args, "base-model-path", args.base_model_path, required=True,
                   resolved=stage.get("base_model_path"),
                   note=stage.get("base_model_note", "")),
        _input_row(args, "model-dir", args.model_dir,
                   note="--base-model-path 为相对路径时的基准目录"),
        _input_row(args, "external-data-path", args.external_data_path,
                   required=stage.get("external_data_required") == "true",
                   resolved=stage.get("external_data_path"),
                   note=stage.get("external_data_note", "")),
        _input_row(args, "allow-implicit-external-data", args.allow_implicit_external_data,
                   note="开启时允许外部权重按同目录隐式解析"),
        _input_row(args, "arch-file", args.arch_file, resolved=stage.get("arch_file"),
                   note=stage.get("arch_note", "")),
        _input_row(args, "dataset-path", args.dataset_path,
                   required=stage.get("dataset_required") == "true",
                   resolved=str(dataset) if dataset else None),
        _input_row(args, "eval-dataset-path", args.eval_dataset_path,
                   note="缺省复用 --dataset-path"),
        _input_row(args, "output-dir", args.output_dir, required=True,
                   resolved=stage.get("output_dir")),
        _input_row(args, "llama-cpp-path", args.llama_cpp_path,
                   note="GGUF 导出使用的离线工具目录"),
        _input_row(args, "hf-cache-dir", args.hf_cache_dir,
                   note="HF/ModelScope 下载缓存（离线复用）"),
    ]
    if args.mode == "api":
        rows += [
            _input_row(args, "api-base", args.api_base, required=True),
            _input_row(args, "api-token", args.api_token, required=True),
            _input_row(args, "platform-model-id", args.platform_model_id, required=True),
        ]
    return rows


_EXTERNAL_DATA_CHECK_MAX_BYTES = 64 * 1024 * 1024


def _onnx_external_locations(model_path: Path) -> list[str]:
    """返回 ONNX 图里声明为外置存储的权重文件 location（不读取权重数据）。

    超过 `_EXTERNAL_DATA_CHECK_MAX_BYTES` 的 ONNX 一定是权重内联的大文件，
    跳过解析以免把整份图读进内存。
    """
    if model_path.suffix.lower() != ".onnx":
        return []
    if model_path.stat().st_size > _EXTERNAL_DATA_CHECK_MAX_BYTES:
        return []
    try:
        import onnx

        proto = onnx.load(str(model_path), load_external_data=False)
    except Exception:  # noqa: BLE001 - 解析失败留给引擎报错
        return []
    locations: list[str] = []
    for tensor in proto.graph.initializer:
        if tensor.data_location != onnx.TensorProto.EXTERNAL:
            continue
        for item in tensor.external_data:
            if item.key == "location" and item.value not in locations:
                locations.append(item.value)
    return locations


def _inline_external_data(
    model_path: Path,
    external: Path,
    staging: Path,
    ctx: JobContext,
) -> Path:
    """按 `--external-data-path` 指定的地址读入外部权重，另存为单文件暂存副本。

    支持两种写法：给 `.onnx.data` 文件本身，或给它所在目录。解析用
    `onnx.external_data_helper.load_external_data_for_model`，所以不论图里
    `location` 写的是什么相对名都能对上。
    """
    import onnx
    from onnx.external_data_helper import load_external_data_for_model

    external = external.expanduser()
    if external.is_file():
        base_dir = external.parent
        locations = _onnx_external_locations(model_path)
        if locations and Path(locations[0]).name != external.name:
            ctx.log(
                "staging",
                f"--external-data-path 给的是 {external.name}，图里声明的是 {locations[0]}；"
                f"按所在目录 {base_dir} 解析",
                level="warning",
            )
    else:
        _require(external.is_dir(),
                 f"--external-data-path 既不是文件也不是目录: {external}", 3)
        base_dir = external

    proto = onnx.load(str(model_path), load_external_data=False)
    try:
        load_external_data_for_model(proto, str(base_dir))
    except Exception as exc:  # noqa: BLE001 - 缺文件/名字对不上都会到这里
        raise JobError(
            f"无法从 {base_dir} 载入外部权重（{type(exc).__name__}: {exc}）；"
            "请确认 --external-data-path 指向 .onnx.data 文件或其所在目录", 3,
        ) from exc

    staging.mkdir(parents=True, exist_ok=True)
    staged = staging / model_path.name
    onnx.save_model(proto, str(staged), save_as_external_data=False)
    ctx.log("staging",
            f"外部权重已内联为单文件：{staged}（源目录 {base_dir}，"
            f"{staged.stat().st_size} B）")
    return staged


def _resolve_input_path(value: str | None, args: argparse.Namespace) -> Path | None:
    """把输入路径解析成绝对路径：相对路径以 `--model-dir` 为基准（若有）。"""
    if not value:
        return None
    path = Path(value).expanduser()
    if args.model_dir and not path.is_absolute():
        path = Path(args.model_dir).expanduser() / path
    return path


def _prepare_model(
    args: argparse.Namespace,
    ctx: JobContext,
    staging: Path,
    *,
    dry_run: bool = False,
) -> tuple[Path, dict[str, str]]:
    """校验/解析模型输入，必要时暂存成自包含副本；返回 (模型路径, 输入解析说明)。

    规则（输入全部显式，不做隐式猜测）：
      * `--model-dir` 给出时，`--base-model-path` 可只给文件名/相对路径；
      * ONNX 若声明了外部权重，则**必须**显式给 `--external-data-path`（文件或目录），
        否则退出码 3；只有显式打开 `--allow-implicit-external-data` 才回退到同目录旧行为；
      * `.pt/.pth` 需要 arch.py 时，把模型与 arch 一起暂存到 `<output-dir>/_stage/`。
    """
    stage: dict[str, str] = {}
    _require(bool(args.base_model_path),
             "--base-model-path 必填（或设 COMPRESS_JOB_BASE_MODEL_PATH）", 3)

    raw = _resolve_input_path(args.base_model_path, args)
    assert raw is not None  # _require 已保证非空
    _require(raw.exists(), f"模型路径不存在: {raw}", 3)
    stage["base_model_path"] = str(raw)
    if args.model_dir and not Path(args.base_model_path).expanduser().is_absolute():
        stage["base_model_note"] = f"按 --model-dir 解析为 {raw}"

    arch = _resolve_input_path(args.arch_file, args)
    if arch is not None:
        _require(arch.is_file(), f"--arch-file 不存在: {arch}", 3)
        stage["arch_file"] = str(arch)

    if raw.is_dir():  # HuggingFace 模型目录
        stage["base_model_note"] = (stage.get("base_model_note", "") + "（目录：HF 模型）").strip()
        return raw, stage

    if raw.suffix.lower() in {".pt", ".pth"}:
        default_arch = raw.parent / "arch.py"
        if arch is None and default_arch.exists():
            arch = default_arch
            stage["arch_file"] = str(arch)
            stage["arch_note"] = "未显式给 --arch-file，按同目录 arch.py 约定解析"
        if arch is not None and not default_arch.exists():
            if dry_run:
                stage["staged_model"] = str(staging / raw.name) + "（dry-run 未写入）"
                return raw, stage
            staging.mkdir(parents=True, exist_ok=True)
            staged_model = staging / raw.name
            staged_arch = staging / "arch.py"
            if not staged_model.exists():
                staged_model.write_bytes(raw.read_bytes())
                ctx.log("staging", f"模型暂存到 {staged_model}")
            staged_arch.write_bytes(arch.read_bytes())
            ctx.log("staging", f"arch.py 暂存到 {staged_arch}")
            stage["staged_model"] = str(staged_model)
            return staged_model, stage
        return raw, stage

    # ---------- ONNX：外部权重必须显式声明 ----------
    locations = _onnx_external_locations(raw)
    if not locations:
        stage["external_data_note"] = f"权重内联（单文件 {raw.stat().st_size} B），无需外部权重"
        return raw, stage

    stage["external_data_required"] = "true"
    if args.external_data_path:
        external = _resolve_input_path(args.external_data_path, args)
        assert external is not None
        _require(external.exists(), f"--external-data-path 不存在: {external}", 3)
        stage["external_data_path"] = str(external)
        if dry_run:
            stage["external_data_note"] = (
                f"dry-run：将从 {external} 内联外部权重（图内声明 {len(locations)} 个："
                f"{', '.join(locations[:3])}）"
            )
            return raw, stage
        return _inline_external_data(raw, external, staging, ctx), stage

    if args.allow_implicit_external_data:
        stage["external_data_note"] = "已显式开启 --allow-implicit-external-data：按同目录隐式解析"
        ctx.log("staging", "外部权重按隐式同目录解析（--allow-implicit-external-data）",
                level="warning")
        return raw, stage

    sibling = raw.parent / f"{raw.name}.data"
    raise JobError(
        "该 ONNX 使用外部权重（external data），但未显式给出权重地址：请加 "
        f"--external-data-path <文件或目录>（图内声明的 location: {', '.join(locations[:3])}）。"
        f"同目录的 {sibling} 不会被默认采用；若确需沿用旧行为，请显式加 "
        "--allow-implicit-external-data",
        3,
    )


def _apply_input_env(args: argparse.Namespace, ctx: JobContext) -> None:
    """把「路径型输入」同步到引擎认识的进程环境变量，并写日志（避免隐式环境依赖）。"""
    if args.llama_cpp_path:
        os.environ["LLAMA_CPP_PATH"] = str(Path(args.llama_cpp_path).expanduser())
        ctx.log("env", f"LLAMA_CPP_PATH={os.environ['LLAMA_CPP_PATH']}")
    if args.hf_cache_dir:
        cache = str(Path(args.hf_cache_dir).expanduser())
        os.environ["HF_HOME"] = cache
        os.environ.setdefault("HF_HUB_CACHE", cache)
        ctx.log("env", f"HF_HOME={cache}")


def _progress_callback(
    ctx: JobContext,
    step: str,
    tracker: ProgressTracker | None = None,
) -> Callable[[float, str], None]:
    def progress(percent: float, message: str = "") -> None:
        ctx.log("progress", message, step=step, percent=round(float(percent), 1))
        if tracker is not None:
            tracker.update(percent, message)

    return progress




def _detect_image_size(artifact: Path) -> int | None:
    """从 ONNX / OpenVINO IR 产物推断图片边长（评估用），推断不出返回 None。"""
    try:
        target = _resolve_artifact(artifact)
        if target.is_dir() and (target / "model.xml").exists():
            import openvino as ov

            model = ov.Core().read_model(str(target / "model.xml"))
            dims = list(next(iter(model.inputs)).partial_shape)
            if len(dims) == 4 and not dims[2].is_dynamic and not dims[3].is_dynamic:
                return int(dims[3].get_length())
            return None
        if target.suffix.lower() == ".onnx":
            import onnxruntime as ort

            session = ort.InferenceSession(str(target), providers=["CPUExecutionProvider"])
            shape = session.get_inputs()[0].shape
            if len(shape) == 4 and isinstance(shape[-1], int) and shape[-1] > 0:
                return int(shape[-1])
    except Exception:  # noqa: BLE001 - 探测失败就沿用用户/默认值
        return None
    return None


def _align_eval_image_size(
    args: argparse.Namespace, ctx: JobContext, artifact: Path | None, eval_config: dict[str, Any]
) -> dict[str, Any]:
    """评估图片尺寸对齐产物输入形状，避免 224/32 之类的维度不匹配。"""
    if artifact is None or eval_config.get("type") != "classification":
        return eval_config
    detected = _detect_image_size(artifact)
    if not detected:
        return eval_config
    if args.image_size is None:
        ctx.log("eval_align", f"按产物输入形状设置评估 image_size={detected}")
        return {**eval_config, "image_size": detected}
    if int(eval_config.get("image_size", 224)) != detected:
        # 产物输入是静态尺寸时以其为准：用错误尺寸评估只会报维度错误
        ctx.log(
            "eval_align",
            f"产物输入为 {detected}px，覆盖 --image-size={eval_config.get('image_size')}；"
            "如需按其它尺寸评估请先导出动态输入形状的产物",
            level="warning",
        )
        return {**eval_config, "image_size": detected}
    return eval_config


def run_local(args: argparse.Namespace, ctx: JobContext) -> JobResult:
    """执行压缩（支持 `--method=pipeline` 串联）并写出结果文件。"""
    _ensure_backend_on_path()
    started = time.time()
    output_dir = Path(args.output_dir).expanduser().resolve()
    artifact_dir = output_dir / "artifact"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    ctx.log_file = output_dir / "logs.jsonl"
    progress = ProgressTracker(ctx, output_dir, interval=max(1, int(args.progress_interval)),
                               enabled=bool(args.progress_file))
    ctx.progress = progress
    if progress.enabled:
        ctx.log("progress_file", f"进度文件: {progress.json_path} / {progress.log_path}")

    dataset = Path(args.dataset_path).expanduser() if args.dataset_path else None
    if dataset is not None:
        _require(dataset.exists(), f"数据集路径不存在: {dataset}", 3)

    _apply_input_env(args, ctx)
    model_path, input_stage = _prepare_model(args, ctx, output_dir / "_stage")
    input_stage["output_dir"] = str(output_dir)
    ctx.log("start", f"job={ctx.job_id} tenant={ctx.tenant_id}", model=str(model_path),
            dataset=str(dataset) if dataset else None, output_dir=str(output_dir))
    ctx.inputs = _input_manifest(args, dataset, stage=input_stage)
    ctx.log("inputs", "本次作业的全部输入（参数 / 来源 / 解析结果）", inputs=ctx.inputs)

    if args.method == "pipeline":
        raw_steps = [s.strip() for s in (args.pipeline or "").split(",") if s.strip()]
        _require(bool(raw_steps),
                 '--method=pipeline 需要 --pipeline="step[:algo],step[:algo]"，'
                 '例如 "prune:group_fisher,quantize:onnx_dynamic_int8"', 2)
        steps: list[tuple[str, str | None]] = []
        for item in raw_steps:
            step_method, _, step_algo = item.partition(":")
            steps.append((step_method.strip(), step_algo.strip() or None))
    else:
        steps = [(args.method, args.algo)]

    functions = _job_functions()
    model = ModelShim(file_path=str(model_path), name=model_path.name,
                      framework="huggingface" if model_path.is_dir() else "pytorch",
                      size_bytes=_size(model_path))
    step_reports: list[dict[str, Any]] = []
    merged_metrics: dict[str, Any] = {}
    current_artifact: Path | None = None
    progress.steps_total = len(steps)

    for index, (step_method, step_algo) in enumerate(steps, start=1):
        platform_method, params = build_params(args, step_method, step_algo, dataset)
        note = params.pop("_data_note", None)
        if note:
            ctx.log("data", note)
        ctx.log("step_start", f"[{index}/{len(steps)}] {step_method}:{step_algo} → {platform_method}",
                params={k: v for k, v in params.items() if not k.startswith("_")})
        step_dir = artifact_dir if len(steps) == 1 else artifact_dir / f"step{index}"
        step_dir.mkdir(parents=True, exist_ok=True)
        progress.start_step(index, len(steps), f"{step_method}:{step_algo} → {platform_method}")

        function = functions.get(platform_method)
        _require(function is not None,
                 f"镜像/环境中的平台代码未提供方法 {platform_method}"
                 f"（缺失: {', '.join(_MISSING_JOB_FUNCTIONS) or '无'}）；"
                 "请使用较新的压缩镜像（deploy/compress/Dockerfile.cpu）或更新平台代码",
                 7)
        try:
            artifact, metrics = function(
                model, params, step_dir, _progress_callback(ctx, f"{index}", progress)
            )
        except JobError:
            progress.fail(f"作业失败: {step_method}:{step_algo}")
            raise
        except ModuleNotFoundError as exc:
            progress.fail(f"缺少依赖: {exc}")
            raise JobError(f"缺少依赖: {exc}（请使用带 LLM/量化扩展的镜像）", 7) from exc
        except Exception as exc:  # noqa: BLE001 - 统一归类为执行失败
            progress.fail(f"作业失败: {type(exc).__name__}: {exc}")
            raise JobError(f"{type(exc).__name__}: {exc}", 5) from exc

        artifact_path = Path(artifact)
        resolved = _resolve_artifact(artifact_path)
        merged_metrics.update(metrics or {})
        current_artifact = artifact_path
        progress.finish("succeeded", artifact=str(artifact_path),
                        message=f"{step_method}:{step_algo} 完成")
        step_reports.append({
            "index": index,
            "method": step_method,
            "algo": step_algo,
            "platform_method": platform_method,
            "params": {k: v for k, v in params.items() if not str(k).startswith("_")},
            "artifact": str(artifact_path),
            "resolved": str(resolved),
            "artifact_bytes": _size(artifact_path),
            "metrics": metrics or {},
        })
        ctx.log("step_done", f"[{index}/{len(steps)}] {platform_method} 完成",
                artifact=str(artifact_path), bytes=_size(artifact_path))
        progress.update(100, f"{step_method}:{step_algo} 完成")

        model = ModelShim(file_path=str(resolved), name=resolved.name,
                          framework="huggingface" if resolved.is_dir() else "pytorch",
                          size_bytes=_size(resolved))

    try:
        result = _finalize(args, ctx, started, output_dir, model_path, dataset,
                           model, current_artifact, step_reports, merged_metrics)
    except Exception as exc:  # noqa: BLE001 - 评测/落盘失败也要把进度收尾
        progress.fail(f"收尾失败: {type(exc).__name__}: {exc}")
        raise
    progress.finish("succeeded", artifact=str(current_artifact) if current_artifact else None,
                    message="作业完成")
    return result


def _finalize(
    args: argparse.Namespace,
    ctx: JobContext,
    started: float,
    output_dir: Path,
    model_path: Path,
    dataset: Path | None,
    model: ModelShim,
    current_artifact: Path | None,
    step_reports: list[dict[str, Any]],
    merged_metrics: dict[str, Any],
) -> JobResult:
    """可选评估 + 写出 result.json / metrics.json。"""
    eval_config = merged_metrics.pop("eval", None)
    if eval_config is None and (dataset is not None or args.eval_dataset_path):
        # 只要给了数据集（训练/评估任一）就尝试评测；评估集缺省复用 --dataset-path
        last_method, last_algo = step_reports[-1]["method"], step_reports[-1]["algo"]
        _, last_params = build_params(args, last_method, last_algo or "", dataset)
        eval_config = last_params.get("eval")
    if eval_config and current_artifact is not None:
        try:
            from app.jobs.compression import run_evaluation

            eval_config = _align_eval_image_size(args, ctx, current_artifact, eval_config)
            ctx.log("eval_start", f"评测: {eval_config.get('type')}")
            merged_metrics["eval"] = run_evaluation(
                model, _resolve_artifact(current_artifact), output_dir / "_eval",
                eval_config, _progress_callback(ctx, "eval"),
            )
            ctx.log("eval_done", "评测完成", metrics=merged_metrics["eval"])
        except Exception as exc:  # noqa: BLE001 - 评测失败不影响压缩产物
            merged_metrics["eval_error"] = f"{type(exc).__name__}: {exc}"[:300]
            ctx.log("eval_error", merged_metrics["eval_error"], level="warning")

    duration = round(time.time() - started, 3)
    result = JobResult(
        status="succeeded",
        method=args.method,
        algo=args.algo,
        params={"steps": [dict(s["params"]) for s in step_reports]},
        metrics=merged_metrics,
        artifact=str(current_artifact) if current_artifact else None,
        artifact_bytes=_size(current_artifact) if current_artifact else 0,
        duration_s=duration,
        steps=step_reports,
    )
    report = {
        "status": result.status,
        "job_id": ctx.job_id,
        "tenant_id": ctx.tenant_id,
        "trace_id": ctx.trace_id,
        "method": args.method,
        "algo": args.algo,
        "base_model_path": str(model_path),
        "dataset_path": str(dataset) if dataset else None,
        "output_dir": str(output_dir),
        "artifact": result.artifact,
        "artifact_resolved": step_reports[-1]["resolved"] if step_reports else None,
        "artifact_bytes": result.artifact_bytes,
        "duration_s": duration,
        "inputs": ctx.inputs,
        "metrics": merged_metrics,
        "steps": step_reports,
    }
    (output_dir / "result.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(merged_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    ctx.log("done", f"完成，用时 {duration}s", artifact=result.artifact,
            artifact_bytes=result.artifact_bytes)
    return result


# --------------------------------------------------------------------------- #
# 6. API 模式（把作业提交到已部署的压缩平台，适合本机无 GPU/无依赖时）
# --------------------------------------------------------------------------- #
def run_api(args: argparse.Namespace, ctx: JobContext) -> JobResult:
    import urllib.error
    import urllib.parse
    import urllib.request

    _ensure_backend_on_path()

    base = (args.api_base or os.environ.get("COMPRESSION_API_BASE") or "http://127.0.0.1:8001").rstrip("/")
    if not base.endswith("/api/v1"):
        base += "/api/v1"
    token = args.api_token or os.environ.get("COMPRESSION_TOKEN")
    email = os.environ.get("COMPRESSION_EMAIL")
    password = os.environ.get("COMPRESSION_PASSWORD")

    def call(method: str, path: str, body: dict[str, Any] | None = None, form: bool = False) -> Any:
        headers: dict[str, str] = {}
        data = None
        if body is not None:
            data = (urllib.parse.urlencode(body) if form else json.dumps(body)).encode()
            headers["Content-Type"] = ("application/x-www-form-urlencoded" if form else "application/json")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        request = urllib.request.Request(f"{base}{path}", data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.load(response) if response.length != 0 else {}
        except urllib.error.HTTPError as exc:
            raise JobError(f"平台 API {exc.code}: {exc.read().decode()[:300]}", 5) from exc
        except urllib.error.URLError as exc:
            raise JobError(f"无法连接平台 {base}: {exc.reason}", 5) from exc

    if not token:
        _require(bool(email and password), "API 模式需要 --api-token 或 COMPRESSION_EMAIL/PASSWORD", 3)
        token = call("POST", "/login/access-token",
                     {"username": email, "password": password}, form=True)["access_token"]

    model_ref = args.platform_model_id or args.base_model_path
    _require(bool(model_ref), "API 模式需要 --platform-model-id（平台模型 ID/名称）", 3)
    platform_method, params = build_params(args, args.method, args.algo or "", None)
    task = call("POST", "/compressions/", {
        "model_id": model_ref, "method": platform_method,
        "params": {k: v for k, v in params.items() if not str(k).startswith("_")},
    })
    task_id = task.get("id")
    _require(bool(task_id), f"提交失败: {task}", 5)
    ctx.log("submitted", f"task_id={task_id}", platform_method=platform_method)

    deadline = time.time() + (args.timeout or 24 * 3600)
    while True:
        current = call("GET", f"/compressions/{task_id}")
        status = current.get("status")
        ctx.log("progress", f"{status} {current.get('progress')}%", task_id=task_id)
        if status in {"succeeded", "failed", "canceled"}:
            break
        _require(time.time() < deadline, f"等待超时（task {task_id}）", 6)
        time.sleep(5)
    if status != "succeeded":
        raise JobError(f"任务 {task_id} {status}: {current.get('error_message')}", 5)

    metrics = current.get("metrics") or {}
    return JobResult(
        status="succeeded", method=args.method, algo=args.algo,
        params={"platform_method": platform_method, "task_id": task_id},
        metrics=metrics, artifact=f"platform://compressions/{task_id}",
        artifact_bytes=int(current.get("output_size_bytes") or 0),
        duration_s=0.0,  # API 模式不统计本地耗时
        steps=[{"platform_method": platform_method, "task_id": task_id, "metrics": metrics}],
    )


# --------------------------------------------------------------------------- #
# 7. 服务模式（可选）：POST /compress 接收同样的参数（JSON），异步执行
# --------------------------------------------------------------------------- #
def _argv_from_payload(payload: dict[str, Any]) -> list[str]:
    """把 JSON 请求体转成 CLI 参数（键 → --kebab-case，布尔 true → 开关）。"""
    argv: list[str] = []
    for key, value in payload.items():
        flag = "--" + str(key).replace("_", "-")
        if isinstance(value, bool):
            if value:
                argv.append(flag)
        elif isinstance(value, (list, tuple)):
            argv += [flag, ",".join(str(v) for v in value)]
        elif value is not None:
            argv += [flag, str(value)]
    return argv


def serve(args: argparse.Namespace) -> int:
    # `--serve` 的可选依赖在模块级做守卫式导入：注解（Request/JSONResponse）必须
    # 能在模块命名空间中解析，否则 FastAPI 会把它们当成 query 参数
    # （`from __future__ import annotations` 会把注解变成字符串，函数内导入的名称
    #  在路由注册时解析不到）。
    if _SERVE_IMPORT_ERROR is not None:  # pragma: no cover - optional extra
        raise JobError(
            f"--serve 需要 fastapi + uvicorn（pip install fastapi uvicorn）：{_SERVE_IMPORT_ERROR}",
            7,
        )

    parser = build_parser()
    app = FastAPI(title="compress-job", version="1.0.0")
    registry: dict[str, dict[str, Any]] = {}

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        import importlib.util

        return {
            "status": "ok",
            "jobs": len(registry),
            "torch": importlib.util.find_spec("torch") is not None,
            "openvino": importlib.util.find_spec("openvino") is not None,
            "nncf": importlib.util.find_spec("nncf") is not None,
            "transformers": importlib.util.find_spec("transformers") is not None,
            "cuda": _cuda_available(),
        }

    @app.post("/compress")
    async def compress(request: Request) -> JSONResponse:
        """请求体：`{"job_id":…, "tenant_id":…, "trace_id":…, "args": {CLI 参数}}`，
        也接受「扁平」写法（CLI 参数直接放在顶层）。"""
        import asyncio

        try:
            payload = await request.json()
        except Exception as exc:  # noqa: BLE001 - invalid JSON body
            raise HTTPException(status_code=422, detail=f"invalid JSON body: {exc}") from exc
        if not isinstance(payload, dict):
            raise HTTPException(status_code=422, detail="body must be a JSON object")

        passthrough = {"args", "job_id", "tenant_id", "trace_id"}
        flat = {key: value for key, value in payload.items() if key not in passthrough}
        nested = payload.get("args") if isinstance(payload.get("args"), dict) else {}
        merged: dict[str, Any] = {**flat, **nested}
        for source, target in (("job_id", "job-id"), ("tenant_id", "tenant-id"),
                               ("trace_id", "trace-id")):
            if payload.get(source):
                merged.setdefault(target, payload[source])
        merged.setdefault("job-id", f"svc-{int(time.time())}")
        merged.setdefault("tenant-id", "local")
        merged.setdefault("trace-id", "")
        merged.setdefault("json-logs", False)

        try:
            request_args = parser.parse_args(_argv_from_payload(merged))
        except SystemExit as exc:  # argparse 报错会退出进程
            raise HTTPException(status_code=422,
                                detail=f"invalid arguments (argparse exit {exc.code})") from exc
        request_args.serve = False
        request_args._supplied = _cli_supplied(_argv_from_payload(merged))
        _bootstrap_settings(Path(request_args.output_dir))
        ctx = JobContext(job_id=request_args.job_id, tenant_id=request_args.tenant_id,
                         trace_id=request_args.trace_id, json_logs=False)

        def _run() -> dict[str, Any]:
            result = run_local(request_args, ctx)
            return {
                "status": result.status, "job_id": ctx.job_id, "artifact": result.artifact,
                "artifact_bytes": result.artifact_bytes, "metrics": result.metrics,
                "duration_s": result.duration_s, "output_dir": str(Path(request_args.output_dir)),
            }

        try:
            loop = asyncio.get_running_loop()
            report = await loop.run_in_executor(None, _run)
        except JobError as exc:
            registry[ctx.job_id] = {"status": "failed", "error": str(exc), "code": exc.code}
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        registry[ctx.job_id] = report
        return JSONResponse(report)

    @app.get("/jobs/{job_id}")
    def job(job_id: str) -> dict[str, Any]:
        if job_id not in registry:
            raise HTTPException(status_code=404, detail="job not found")
        return registry[job_id]

    print(f"[compress-job] serving on http://{args.host}:{args.port}  (POST /compress)", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001 - torch absent
        return False


# --------------------------------------------------------------------------- #
# 8. 摘要与入口
# --------------------------------------------------------------------------- #
_HIGHLIGHT_METRICS = (
    "precision", "bits", "group_size", "outtype", "tool", "model_kind",
    "total_filters", "pruned_filters", "params_after", "forward_check",
    "head_layers_adjusted", "onnx_exported", "size_reduction_pct", "ratio",
    "trainable_params", "preference_pairs", "dpo_ready", "samples",
    "initial_loss", "final_loss", "perplexity", "accuracy", "baseline_accuracy",
)


def _finish_progress(ctx: JobContext, status: str, message: str = "") -> None:
    """让进度文件收尾（失败/中断路径也留下 100% 的快照与状态）。"""
    tracker = getattr(ctx, "progress", None)
    if tracker is not None:
        try:
            tracker.fail(message, status=status)
        except Exception:  # noqa: BLE001 - 进度收尾失败不影响退出码
            pass


def _print_summary(
    result: JobResult,
    stream: Any = sys.stdout,
    output_dir: Path | None = None,
) -> None:
    def human(size: int) -> str:
        value = float(size)
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if value < 1024 or unit == "TB":
                return f"{value:.1f}{unit}" if unit != "B" else f"{int(value)}B"
            value /= 1024
        return f"{value:.1f}TB"

    print("-" * 68, file=stream)
    print(f"status     : {result.status}", file=stream)
    print(f"method     : {result.method}{':' + result.algo if result.algo else ''}", file=stream)
    print(f"artifact   : {result.artifact}", file=stream)
    print(f"artifact 大小: {human(result.artifact_bytes)}", file=stream)
    print(f"耗时        : {result.duration_s}s", file=stream)
    highlights = {k: v for k, v in result.metrics.items()
                  if k in _HIGHLIGHT_METRICS and v is not None}
    if highlights:
        print("指标        :", file=stream)
        for key, value in highlights.items():
            print(f"  {key:<20} {value}", file=stream)
    if "eval" in result.metrics:
        print(f"评测        : {result.metrics['eval']}", file=stream)
    if "eval_error" in result.metrics:
        print(f"评测失败    : {result.metrics['eval_error']}", file=stream)
    if output_dir is not None:
        print(f"输出目录    : {output_dir}", file=stream)
        present = [name for name in ("artifact", "result.json", "metrics.json", "logs.jsonl",
                                     "progress.json", "progress.log")
                   if (output_dir / name).exists()]
        print(f"  已写出    : {'  '.join(present) or '（无）'}", file=stream)
    print("-" * 68, file=stream)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    # 记录哪些参数是命令行显式给出的（用于 inputs 清单里的来源标注）
    args._supplied = _cli_supplied(list(argv if argv is not None else sys.argv[1:]))

    if args.serve:
        try:
            return serve(args)
        except JobError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return exc.code

    _bootstrap_settings(Path(args.output_dir))
    ctx = JobContext(job_id=args.job_id, tenant_id=args.tenant_id, trace_id=args.trace_id,
                     json_logs=bool(args.json_logs))
    seed_everything(args.seed)

    try:
        _require(bool(args.method), "--method 必填（quantize|prune|distill|finetune|export|pipeline）", 2)
        dataset = Path(args.dataset_path).expanduser() if args.dataset_path else None

        if args.dry_run:
            steps = ([(m.strip(), (a.strip() or None)) for m, _, a in
                      (item.partition(":") for item in (args.pipeline or "").split(","))]
                     if args.method == "pipeline" else [(args.method, args.algo)])
            preview = []
            for step_method, step_algo in steps:
                if not step_method:
                    continue
                platform_method, params = build_params(args, step_method, step_algo, dataset)
                preview.append({
                    "cli_step": f"{step_method}:{step_algo or ''}".rstrip(":"),
                    "platform_method": platform_method,
                    "params": {k: v for k, v in params.items() if not str(k).startswith("_")},
                })
            # 输入解析：模型存在时连外部权重等要求一起校验（含暂存计划，但不写盘）
            stage: dict[str, str] = {"output_dir": str(Path(args.output_dir).expanduser().resolve())}
            raw_model = Path(args.base_model_path).expanduser() if args.base_model_path else None
            if raw_model is not None and args.model_dir and not raw_model.is_absolute():
                raw_model = Path(args.model_dir).expanduser() / raw_model
            if raw_model is not None and raw_model.exists():
                try:
                    _, stage = _prepare_model(args, ctx, Path(args.output_dir) / "_stage",
                                              dry_run=True)
                    stage["output_dir"] = str(Path(args.output_dir).expanduser().resolve())
                except JobError as exc:
                    print(json.dumps({
                        "dry_run": True, "job_id": ctx.job_id,
                        "inputs": _input_manifest(args, dataset, stage=stage),
                        "input_error": {"message": str(exc), "exit_code": exc.code},
                    }, ensure_ascii=False, indent=2))
                    return exc.code
            else:
                stage["base_model_note"] = "dry-run：模型路径不存在或未给出，跳过外部权重等校验"
            print(json.dumps({
                "dry_run": True, "job_id": ctx.job_id, "tenant_id": ctx.tenant_id,
                "output_dir": str(Path(args.output_dir).expanduser().resolve()),
                "inputs": _input_manifest(args, dataset, stage=stage),
                "steps": preview,
            }, ensure_ascii=False, indent=2))
            return 0

        alarm = None
        if args.timeout and hasattr(signal, "SIGALRM"):
            def _on_alarm(_signum: int, _frame: Any) -> None:
                raise JobError(f"作业超时（>{args.timeout:.0f}s）", 6)

            signal.signal(signal.SIGALRM, _on_alarm)
            alarm = int(args.timeout)
            signal.alarm(alarm)
        try:
            result = run_api(args, ctx) if args.mode == "api" else run_local(args, ctx)
        finally:
            if alarm is not None:
                signal.alarm(0)
    except JobError as exc:
        _finish_progress(ctx, "failed", str(exc))
        ctx.log("error", str(exc), level="error", exit_code=exc.code)
        print(f"error: {exc}", file=sys.stderr)
        return exc.code
    except KeyboardInterrupt:  # pragma: no cover - interactive cancel
        _finish_progress(ctx, "interrupted", "用户中断")
        ctx.log("interrupted", "用户中断", level="warning")
        return 130

    _print_summary(result, stream=sys.stderr if args.json_logs else sys.stdout,
                   output_dir=Path(args.output_dir).expanduser().resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())




