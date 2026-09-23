#!/usr/bin/env python3
"""压缩代码包自检：确认「引擎实现体在包里」且 `compress_job.py` 能直接跑。

用法（在代码包根目录、或本仓库根目录执行均可）::

    python scripts/check_compress_package.py                    # 布局 + 语法 + 引擎导入 + CLI 干跑
    python scripts/check_compress_package.py --run model.onnx   # 额外真跑一次量化作业
    python scripts/check_compress_package.py --run model.pt --algo onnx_dynamic_int8 \\
        --arch-file path/to/arch.py

退出码：0 全部通过；1 有检查项失败；2 缺少运行依赖（torch/onnxruntime 等）无法做引擎级检查。
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# 代码包必须包含的文件（缺一个就意味着「只能看文档，跑不起来」）
REQUIRED_FILES: tuple[str, ...] = (
    "compress_job.py",                    # 单文件 CLI（壳）
    "pyproject.toml",                     # 依赖声明（镜像构建用）
    "uv.lock",                            # 依赖锁定，保证可复现构建
    "backend/pyproject.toml",
    "backend/app/__init__.py",
    "backend/app/models.py",              # CompressionMethod 枚举
    "backend/app/core/config.py",
    "backend/app/core/db.py",
    "backend/app/jobs/__init__.py",
    "backend/app/jobs/compression.py",     # ★ 压缩引擎实现体（量化/剪枝/蒸馏…）
    "backend/app/jobs/storage.py",
    "deploy/compress/Dockerfile.cpu",
    "deploy/compress/Dockerfile.gpu",
    "deploy/compress/Dockerfile.fast",
    "docs/compression-job.md",
    "docs/compression-report.md",
    "examples/compression-demo-welding.md",
)

# 引擎必须暴露的入口（对应 CLI 的 --method/--algo 全部分支）
ENGINE_CALLABLES: tuple[str, ...] = (
    "export_onnx",
    "quantize_onnx",
    "quantize_onnx_int4",
    "quantize_onnx_static",
    "openvino_int8",
    "prune",
    "sparsify",
    "distill",
    "qat",
    "llm_finetune",
    "llm_quantize",
    "export_gguf",
    "resolve_artifact_file",
    "run_evaluation",
    "pick_artifact",
    "run_benchmark",
    "rerun_metrics",
)

# 平台方法枚举必须与 CLI 的 ALGO_MAP 对齐
ENUM_MEMBERS: tuple[str, ...] = (
    "export_onnx",
    "quantize_onnx",
    "quantize_onnx_int4",
    "quantize_onnx_static",
    "openvino_int8",
    "prune",
    "sparsify",
    "distill",
    "qat",
    "llm_finetune",
    "export_gguf",
    "llm_quantize",
)

# 引擎导入探针（子进程执行，避免污染本进程环境）
ENGINE_PROBE = """
import json, os, sys
root = os.environ["CPK_ROOT"]
# 与 compress_job.py 的 _ensure_backend_on_path 完全一致的查找顺序；
# 只插入 backend/（绝不能把包根放进 sys.path：根目录若含 compression.py 会遮蔽标准库同名包）
probe = os.path.join(root, "backend")
if os.path.isfile(os.path.join(probe, "app", "jobs", "compression.py")):
    sys.path.insert(0, probe)
else:
    print(json.dumps({"error": f"缺少 {probe}/app/jobs/compression.py（压缩引擎未随包交付）"}))
    raise SystemExit(1)
os.environ.setdefault("SECRET_KEY", "compress-package-check")
os.environ.setdefault("PROJECT_NAME", "compress-package-check")
os.environ.setdefault("DATABASE_URL", "postgresql://check:check@127.0.0.1:5432/check")
os.environ.setdefault("FIRST_SUPERUSER", "check@example.com")
os.environ.setdefault("FIRST_SUPERUSER_PASSWORD", "check")
os.environ.setdefault("MODEL_STORAGE_PATH", os.environ["CPK_TMP"])
try:
    from app.jobs import compression as engine
    from app.models import CompressionMethod
except Exception as exc:  # pragma: no cover - 依赖缺失/代码缺失都会走到这里
    print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))
    raise SystemExit(1)
names = json.loads(os.environ["CPK_ENGINE_NAMES"])
members = json.loads(os.environ["CPK_ENUM_MEMBERS"])
print(json.dumps({
    "missing_engine": [n for n in names if not callable(getattr(engine, n, None))],
    "missing_enum": [m for m in members if not hasattr(CompressionMethod, m)],
    "module_file": getattr(engine, "__file__", ""),
}))
"""


def ok(msg: str) -> None:
    print(f"  [OK] {msg}")


def warn(msg: str) -> None:
    print(f"  [!!] {msg}")


def bad(msg: str) -> None:
    print(f"  [FAIL] {msg}")


def check_layout(root: Path) -> int:
    print("[1/4] 文件布局")
    missing = [rel for rel in REQUIRED_FILES if not (root / rel).is_file()]
    if missing:
        for rel in missing:
            bad(f"缺少 {rel}")
        return 1
    ok(f"{len(REQUIRED_FILES)} 个关键文件齐全（含压缩引擎 backend/app/jobs/compression.py）")
    engine = root / "backend/app/jobs/compression.py"
    ok(f"引擎代码：{engine.relative_to(root)}（{engine.stat().st_size / 1024:.0f} KB）")
    return 0


def check_syntax(root: Path) -> int:
    print("[2/4] Python 语法")
    targets = [root / "compress_job.py", *sorted((root / "backend/app").rglob("*.py"))]
    targets = [p for p in targets if "__pycache__" not in p.parts]
    failed: list[str] = []
    for path in targets:
        try:
            ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as exc:  # pragma: no cover
            failed.append(f"{path.relative_to(root)}: {exc}")
    if failed:
        for item in failed:
            bad(item)
        return 1
    ok(f"{len(targets)} 个 .py 文件语法正确")
    return 0


def check_engine(root: Path, python: str) -> int:
    print("[3/4] 引擎导入（量化/剪枝等实现体可加载）")
    with tempfile.TemporaryDirectory(prefix="cpk-") as tmp:
        env = dict(os.environ)
        env.update(
            CPK_ROOT=str(root),
            CPK_TMP=tmp,
            CPK_ENGINE_NAMES=json.dumps(list(ENGINE_CALLABLES)),
            CPK_ENUM_MEMBERS=json.dumps(list(ENUM_MEMBERS)),
        )
        proc = subprocess.run(
            [python, "-c", ENGINE_PROBE], capture_output=True, text=True, env=env, cwd=tmp
        )
    if proc.returncode != 0:
        detail = (proc.stdout + proc.stderr).strip().splitlines()
        bad(f"导入 app.jobs.compression 失败：{detail[-1] if detail else '未知错误'}")
        warn("通常是本机缺少 torch/onnxruntime 等依赖；请用环境包镜像，或先 uv sync --no-dev --package app")
        return 2
    info = json.loads(proc.stdout.strip().splitlines()[-1])
    if info["missing_engine"]:
        bad(f"引擎缺少入口：{', '.join(info['missing_engine'])}")
        return 1
    if info["missing_enum"]:
        bad(f"CompressionMethod 缺少成员：{', '.join(info['missing_enum'])}")
        return 1
    ok(f"{len(ENGINE_CALLABLES)} 个引擎入口全部存在（文件：{info['module_file']}）")
    ok(f"CompressionMethod {len(ENUM_MEMBERS)} 个方法枚举齐全")
    return 0


def check_cli(root: Path, python: str) -> int:
    print("[4/4] CLI 可运行（--help / --dry-run）")
    cli = str(root / "compress_job.py")
    help_proc = subprocess.run([python, cli, "--help"], capture_output=True, text=True)
    if help_proc.returncode != 0:
        bad(f"compress_job.py --help 退出码 {help_proc.returncode}")
        return 1
    ok("--help 正常（参数解析器就绪）")
    text = help_proc.stdout
    if "--output-dir" not in text:
        bad("--help 里缺少 --output-dir（输出目录参数必须叫 --output-dir）")
        return 1
    for stray in ("--out ", "--outdir", "--output-path", "--out-dir"):
        if stray in text:
            bad(f"出现非规范输出参数 {stray!r}；输出目录统一用 --output-dir")
            return 1
    ok("输出目录参数唯一且规范：--output-dir")
    with tempfile.TemporaryDirectory(prefix="cpk-out-") as out:
        dry = subprocess.run(
            [
                python, cli, "--dry-run", "--json-logs",
                "--job-id=selfcheck", "--base-model-path=/nonexistent.pt",
                "--output-dir", out,
                "--method=quantize", "--algo=onnx_dynamic_int8",
            ],
            capture_output=True,
            text=True,
        )
    if dry.returncode != 0:
        bad(f"--dry-run 退出码 {dry.returncode}：{(dry.stdout + dry.stderr).strip()[-300:]}")
        return 1
    ok("--dry-run 通过（参数映射与产物路径规划正确）")
    return 0


def run_real_job(
    root: Path,
    python: str,
    model: Path,
    algo: str,
    arch_file: str | None,
    external_data: str | None = None,
) -> int:
    print(f"[额外] 真跑一次压缩作业：{model.name} → --algo={algo}")
    with tempfile.TemporaryDirectory(prefix="cpk-real-") as out:
        cmd = [
            python, str(root / "compress_job.py"), "--json-logs",
            "--job-id=selfcheck", "--base-model-path", str(model),
            "--output-dir", out, "--method=quantize", f"--algo={algo}",
        ]
        if arch_file:
            cmd += ["--arch-file", arch_file]
        if external_data:
            cmd += ["--external-data-path", external_data]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            bad(f"作业失败（退出码 {proc.returncode}）：{(proc.stdout + proc.stderr).strip()[-500:]}")
            if proc.returncode == 3 and "外部权重" in (proc.stdout + proc.stderr):
                warn("该模型是外部权重 ONNX：请加 --external-data-path <文件或目录> 再自检")
            return 1
        artifact_dir = Path(out) / "artifact"
        produced = sorted(p for p in artifact_dir.rglob("*") if p.is_file()) if artifact_dir.is_dir() else []
        if not produced:
            bad(f"未产出 artifact：{artifact_dir}")
            return 1
        smallest = min(produced, key=lambda p: p.stat().st_size)
        ok(f"产出 {len(produced)} 个文件，主产物 {smallest.name}（{smallest.stat().st_size} B）")
        # 进度文件：必须在 output-dir 下、每 10% 一行、快照到 100%
        progress_json = Path(out) / "progress.json"
        progress_log = Path(out) / "progress.log"
        if not progress_json.is_file() or not progress_log.is_file():
            bad("缺少 progress.json / progress.log（--progress-file 默认应在 --output-dir 下生成）")
            return 1
        snapshot = json.loads(progress_json.read_text(encoding="utf-8"))
        lines = [ln for ln in progress_log.read_text(encoding="utf-8").splitlines() if ln.strip()]
        milestones = [int(ln.split()[1].rstrip("%")) for ln in lines if "%" in ln]
        expected = list(range(0, 101, int(snapshot.get("interval", 10))))
        if not set(expected).issubset(set(milestones)):
            bad(f"progress.log 缺少里程碑：应为 {expected}，实际 {sorted(set(milestones))}")
            return 1
        if snapshot.get("status") != "succeeded" or int(snapshot.get("percent", 0)) != 100:
            bad(f"progress.json 收尾不正确：status={snapshot.get('status')} percent={snapshot.get('percent')}")
            return 1
        ok(f"进度文件正常：progress.log {len(lines)} 行（含 0–100% 每档）、progress.json 收尾 100%")
        result = Path(out) / "result.json"
        if result.is_file():
            data = json.loads(result.read_text(encoding="utf-8"))
            metrics = data.get("metrics", data)
            ok(f"result.json：{json.dumps(metrics, ensure_ascii=False)[:220]}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="压缩代码包自检（布局 / 语法 / 引擎 / CLI）")
    parser.add_argument("--root", default=".", help="代码包根目录（默认当前目录）")
    parser.add_argument("--python", default=sys.executable, help="用于检查的解释器")
    parser.add_argument("--run", metavar="MODEL", help="额外用该模型真跑一次量化作业")
    parser.add_argument("--algo", default="onnx_dynamic_int8", help="--run 时使用的算法")
    parser.add_argument("--arch-file", default=None, help="--run 时的架构文件（自定义模型需要）")
    parser.add_argument("--external-data-path", default=None,
                        help="--run 时外部权重 ONNX 的 .data 文件/目录")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    print(f"压缩代码包自检：{root}")
    print(f"解释器：{args.python}\n")

    codes = [
        check_layout(root),
        check_syntax(root),
        check_engine(root, args.python),
        check_cli(root, args.python),
    ]
    if args.run:
        model = Path(args.run).resolve()
        if not model.is_file():
            bad(f"模型不存在：{model}")
            codes.append(1)
        else:
            codes.append(run_real_job(root, args.python, model, args.algo, args.arch_file,
                                      args.external_data_path))

    print()
    if any(code == 1 for code in codes):
        print("结论：自检失败 —— 代码包不完整或环境不满足，请查看上面的 FAIL 项。")
        return 1
    if any(code == 2 for code in codes):
        print("结论：代码包完整；本机缺少推理/训练依赖，未做引擎级验证（镜像内可直接运行）。")
        return 0
    print("结论：自检通过 —— 代码包自带压缩引擎，`python compress_job.py …` 可直接运行。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
