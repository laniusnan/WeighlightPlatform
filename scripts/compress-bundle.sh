#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# 一键打包「模型压缩」交付物（环境与代码分别成包）：
#
#   环境包：dist/compress-job-cpu-<日期>.tar.gz      # 由 scripts/compress-image.sh save 生成
#   代码包：dist/compress-code-<日期>.zip            # 本脚本 --bundle-only：代码/文档/demo，几百 KB
#   全量包：dist/compress-full-<日期>.zip            # 本脚本默认：代码包内容 + 镜像 tar.gz
#
#   scripts/compress-bundle.sh                 # 若已有镜像 tar 则复用，否则自动 docker save
#   scripts/compress-bundle.sh --with-image    # 先构建镜像（快路径）再打包
#   scripts/compress-bundle.sh --bundle-only   # 只打代码与文档（不含镜像，小体积）
#
# 交付物结构（解压后）：
#   compress-code-<日期>/  或  compress-full-<日期>/
#     ├── README.md                      # 交付说明（镜像 import + 运行示例）
#     ├── compress_job.py                # 单文件压缩 CLI
#     ├── deploy/compress/…              # Dockerfile/示例/结果
#     ├── docs/compression-job.md        # 命令行与参数说明
#     ├── docs/compression-report.md     # 实验报告
#     ├── examples/compression-demo-welding.md
#     ├── scripts/{compress-image.sh,compress-bundle.sh}
#     ├── SHA256SUMS                     # 逐文件校验和
#     └── image/compress-job-cpu-<日期>.tar.gz(+.sha256)   # 仅全量包
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATE="$(date +%Y%m%d)"
DIST="$REPO_ROOT/dist"
IMAGE="${CPU_IMAGE:-compress-job:cpu}"
WITH_IMAGE=0
BUNDLE_ONLY=0

for arg in "$@"; do
  case "$arg" in
    --with-image) WITH_IMAGE=1 ;;
    --bundle-only) BUNDLE_ONLY=1 ;;
    *) echo "unknown option: $arg"; exit 2 ;;
  esac
done

# 环境包与代码包分开命名：--bundle-only 不带镜像（几百 KB）
if [ "$BUNDLE_ONLY" = "1" ]; then
  NAME="compress-code-$DATE"
else
  NAME="compress-full-$DATE"
fi
STAGE="$DIST/$NAME"

echo "==> 准备暂存目录 $STAGE"
rm -rf "$STAGE"
mkdir -p "$STAGE"/{deploy/compress/examples,examples,docs,scripts,image}

copy() { cp -r "$1" "$2"; }

# 代码与脚本
copy compress_job.py "$STAGE/compress_job.py"
copy deploy/compress/Dockerfile.cpu "$STAGE/deploy/compress/"
copy deploy/compress/Dockerfile.gpu "$STAGE/deploy/compress/"
copy deploy/compress/Dockerfile.fast "$STAGE/deploy/compress/"
copy deploy/compress/README.md "$STAGE/deploy/compress/"
copy deploy/compress/llama-cpp "$STAGE/deploy/compress/llama-cpp"
copy deploy/compress/examples/welding_eval.py "$STAGE/deploy/compress/examples/"
copy deploy/compress/examples/welding_demo.sh "$STAGE/deploy/compress/examples/"
copy deploy/compress/examples/results "$STAGE/deploy/compress/examples/results"
copy scripts/compress-image.sh "$STAGE/scripts/compress-image.sh"
copy scripts/compress-bundle.sh "$STAGE/scripts/compress-bundle.sh"
copy scripts/check_compress_package.py "$STAGE/scripts/check_compress_package.py"

# 依赖声明与构建文件（Dockerfile.cpu 需要根 pyproject + uv.lock + backend/pyproject）
copy pyproject.toml "$STAGE/pyproject.toml"
copy uv.lock "$STAGE/uv.lock"
copy .dockerignore "$STAGE/.dockerignore"

# 后端源码：★ 压缩引擎实现体（量化/剪枝/蒸馏/微调/导出）在这里，
# 没有它 compress_job.py 只是个空壳（会报「找不到 app/jobs/compression.py」）
mkdir -p "$STAGE/backend"
cp -r backend/app "$STAGE/backend/app"
for f in pyproject.toml requirements.txt requirements-gpu.txt Dockerfile Dockerfile.cpu Dockerfile.gpu alembic.ini README.md; do
  if [ -f "backend/$f" ]; then cp "backend/$f" "$STAGE/backend/$f"; fi
done
copy backend/scripts "$STAGE/backend/scripts" 2>/dev/null || true

# 瘦身：前端构建产物、缓存、本地上传的模型都不属于代码交付件
rm -rf "$STAGE/backend/app/frontend" "$STAGE/backend/app/email-templates" "$STAGE/backend/models-storage"
find "$STAGE" -type d -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true
find "$STAGE" -type d \( -name '.pytest_cache' -o -name '.mypy_cache' -o -name '.ruff_cache' \) -prune -exec rm -rf {} + 2>/dev/null || true
find "$STAGE" -type f -name '*.pyc' -delete 2>/dev/null || true

# 文档
copy docs/compression-job.md "$STAGE/docs/"
copy docs/compression-report.md "$STAGE/docs/"
copy docs/compression-cli.md "$STAGE/docs/" 2>/dev/null || true
copy docs/compression-env.md "$STAGE/docs/" 2>/dev/null || true
copy examples/compression-demo-welding.md "$STAGE/examples/"
copy examples/compression-demo.md "$STAGE/examples/" 2>/dev/null || true

# 镜像
if [ "$WITH_IMAGE" = "1" ]; then
  echo "==> 构建镜像（快路径，复用平台镜像）"
  docker build -f deploy/compress/Dockerfile.fast -t "$IMAGE" \
    --build-arg BASE_IMAGE="${FAST_BASE:-backend:cpu-ship}" .
fi
if [ "$BUNDLE_ONLY" = "0" ]; then
  EXISTING="$(ls -1 "$DIST"/compress-job-cpu-*.tar.gz 2>/dev/null | tail -1 || true)"
  if [ -n "$EXISTING" ]; then
    echo "==> 复用已导出的镜像 tar：$EXISTING"
    cp -f "$EXISTING" "$STAGE/image/"
    [ -f "$EXISTING.sha256" ] && cp -f "$EXISTING.sha256" "$STAGE/image/"
  elif docker image inspect "$IMAGE" >/dev/null 2>&1; then
    TAR="$STAGE/image/compress-job-cpu-$DATE.tar.gz"
    echo "==> 导出镜像 -> $TAR（数 GB，需几分钟）"
    docker save "$IMAGE" | gzip -1 > "$TAR"
    ( cd "$(dirname "$TAR")" && sha256sum "$(basename "$TAR")" > "$(basename "$TAR").sha256" )
  else
    echo "!! 未找到镜像 $IMAGE 且无已导出的 tar：跳过镜像（可先跑 scripts/compress-image.sh build / save）"
  fi
fi

# 交付说明
cat > "$STAGE/README.md" <<'MD'
# 模型压缩交付包（mlcompress）

> 交付分两个包，可分别传输：
> - **环境包**：`compress-job-cpu-<日期>.tar.gz`（约 1.7 GB）——Docker 镜像，含全部 Python 依赖与运行时；
> - **代码包**：`compress-code-<日期>.zip`——本包：**压缩引擎源码 + CLI + Dockerfile + 文档 + demo + 实测结果**（不含镜像，约 1 MB）。
>
> 全量包 `compress-full-<日期>.zip` = 代码包内容 + `image/` 里的环境包，一次拿齐。

## 包内结构（代码包 = 可运行的最小源码树）

```
compress-code-<日期>/
├── BUILD_INFO.txt                 # ★ 版本指纹：构建时间/git/关键文件 sha256/核对方法
├── compress_job.py               # 单文件作业 CLI（壳）
├── backend/app/jobs/compression.py  # ★ 压缩引擎实现体：量化/剪枝/稀疏化/蒸馏/QAT/微调/GGUF…
├── backend/app/{models.py,core/…}   # 引擎依赖的枚举、配置、DB 适配层
├── backend/pyproject.toml        # 后端依赖声明
├── pyproject.toml + uv.lock      # 依赖锁定（复现构建）
├── deploy/compress/*.Dockerfile  # 作业镜像定义
├── scripts/check_compress_package.py  # 代码包自检（布局/引擎/CLI/进度文件）
├── docs/ · examples/ · deploy/compress/examples/results/   # 文档、demo、实测结果
├── SHA256SUMS                     # 逐文件校验和
└── BUILD_INFO.txt                 # 版本指纹（见上）
```

## 0. 先核对版本（同名包可能有新旧两版，务必先做）

```bash
cat BUILD_INFO.txt                                   # 构建时间 / git / 关键文件 sha256
sha256sum -c SHA256SUMS                              # 包内逐文件校验
grep -c _prepare_quant_input backend/app/jobs/compression.py   # ≥1 = 含 ONNX 自动降级链（2026-09-22 起）
python scripts/check_compress_package.py             # 自检：布局/引擎/CLI/进度文件
```

对照 zip 自身的校验值：`compress-code-<日期>.zip.sha256`（与 zip 同目录）。**只换 `compress_job.py` 不够**——
修复在引擎 `backend/app/jobs/compression.py` 里，请整包替换。

## 1. 导入镜像（环境包）

```bash
sha256sum -c image/compress-job-cpu-*.tar.gz.sha256   # 可选：校验
gzip -dc image/compress-job-cpu-*.tar.gz | docker load      # 得到 compress-job:cpu
docker run --rm compress-job:cpu --help                     # 自检
```

## 2. 代码包自检与三种运行方式

**自检（先跑这个，确认包完整、引擎在、CLI 能跑）**：

```bash
python scripts/check_compress_package.py                 # 布局 + 语法 + 引擎导入 + --dry-run
python scripts/check_compress_package.py --run model.onnx  # 额外真跑一次量化作业
```

① **宿主机直接跑**（不用 Docker；需自备 Python 依赖，与镜像同源）：

```bash
pip install uv && uv sync --no-dev --package app           # 依赖按 uv.lock 锁定版本安装
python compress_job.py --job-id=1001 --base-model-path=/models/model.pt \
  --output-dir=/out/1001 --method=quantize --algo=onnx_dynamic_int8
```

② **在代码包里构建镜像**（`Dockerfile.cpu` 只用本包内容，可离线复现）：

```bash
docker build -f deploy/compress/Dockerfile.cpu -t compress-job:cpu .
```

③ **直接用环境包镜像**（推荐，零依赖安装）：见下面第 3 节。

## 3. 压缩模型（命令行直调，与训练脚本同风格）

```bash
docker run --rm -v /data/models:/models -v /data/out:/out compress-job:cpu \
  --job-id={{job.id}} --tenant-id={{job.tenant_id}} --trace-id={{job.trace_id}} \
  --base-model-path={{inputs.base_model_path}} \
  --output-dir={{inputs.output_path}} \
  --method=quantize --algo=onnx_dynamic_int8
```

**所有输入都要显式给出**（旁挂文件不会被猜出来）：

```bash
# ONNX 外部权重：必须给文件或目录（缺失即退出码 3）
  --base-model-path=/models/model.onnx --external-data-path=/models/model.onnx.data
# 只给文件名/相对路径时，用 --model-dir 指明基准目录
  --model-dir=/models --base-model-path=model.onnx --external-data-path=model.onnx.data --arch-file=arch.py
# 兼容旧行为（显式开启后，外部权重才按“与 .onnx 同目录”解析）
  --allow-implicit-external-data
# 其它路径型输入：--dataset-path / --eval-dataset-path / --arch-file /
#                 --llama-cpp-path / --hf-cache-dir / --output-dir
```

核对一次运行到底用了哪些输入：`--dry-run` 会打印 **inputs 清单**（参数名 / 来源 cli·env·default /
解析结果 / 是否必填），每次作业的 `result.json` 里也有同样的 `inputs`。

### 输出目录与日志/进度（统一规则）

* **输出目录参数只有一个名字：`--output-dir`**（等价环境变量 `COMPRESS_JOB_OUTPUT_DIR`；调度器模板
  `--output-dir={{inputs.output_path}}`）。所有产物、报告、日志、进度都写在这个目录下：

```
<output-dir>/
├── artifact/       压缩产物（model.onnx / model_quantized.onnx / IR 目录 / *.gguf …）
├── result.json     作业总报告（含 inputs 清单、metrics、steps、artifact_bytes）
├── metrics.json    指标
├── logs.jsonl      结构化日志（**与 stdout 内容一致**，逐条 JSON Lines）
├── progress.json   进度快照（最新的 status/step/percent/message/elapsed_s/artifact）
├── progress.log    进度里程碑（**每 10% 一行**，可 tail -f）
└── _stage/         暂存副本（内联后的单文件 ONNX、arch.py 等，跑完可删）
```

* **日志是“两条腿”**：`JobContext.log()` 既 `print` 到 stdout（默认 JSON Lines，
  `--no-json-logs` 时转人类可读；末页摘要表走 stderr），又追加写 `<output-dir>/logs.jsonl`。
  所以容器日志采集与事后排查都能拿到同一份内容。
* **进度**：`--progress-interval`（默认 10，可改）保证每跨过一个档位就写一行 `progress.log` +
  一条 `progress_milestone` 日志；`progress.json` 是随时可读的最新快照（原子替换写入）。
  失败/中断也会落盘（`status: failed/interrupted` + 原因），不会留下没有结束状态的进度。
  `--no-progress-file` 可关闭。

结果：`{{inputs.output_path}}/artifact/…`（压缩产物）、`result.json`（指标）、`logs.jsonl`（日志）。

## 4. 文档

| 文档 | 内容 |
|------|------|
| `docs/compression-job.md` | 命令行**每个参数**的含义/取值/作用、方法矩阵、退出码、产物结构、排错表 |
| `docs/compression-report.md` | **实验报告**（焊接模型：压缩率 62.60%、精度损失 0.06%） |
| `examples/compression-demo-welding.md` | **demo 文档**：逐步操作与实测结论 |
| `deploy/compress/README.md` | 交付包说明、镜像构建/分发、能力速查 |

## 5. 更新已有镜像里的代码（可选）

镜像内代码快照在 `/app/compress_job.py`、`/app/examples/`；把新版代码热更新进容器：

```bash
docker cp compress_job.py            <container>:/app/compress_job.py
docker cp backend/app/jobs/compression.py <container>:/app/backend/app/jobs/compression.py
docker cp backend/app/models.py      <container>:/app/backend/app/models.py
```

## 6. 复现 demo（需自备焊接模型包）

```bash
chmod +x deploy/compress/examples/welding_demo.sh
deploy/compress/examples/welding_demo.sh <焊接模型包目录> <输出目录> compress-job:cpu
```

（另附本次实测原始结果：`deploy/compress/examples/results/`，含评测 JSON 与压缩产物 INT8 ONNX。）
MD

echo "==> 生成校验和"
( cd "$STAGE" && find . -type f ! -name SHA256SUMS -exec sha256sum {} + | sort -k2 > SHA256SUMS )

echo "==> 生成 BUILD_INFO.txt（版本指纹，避免同名新旧包混淆）"
{
  echo "包类型      : $NAME"
  echo "构建时间    : $(date +%Y-%m-%dT%H:%M:%S%z)"
  GIT_SHA="$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
  GIT_BRANCH="$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
  GIT_DIRTY="$(git -C "$REPO_ROOT" status --porcelain 2>/dev/null | head -1 >/dev/null && \
              { [ -n "$(git -C "$REPO_ROOT" status --porcelain 2>/dev/null)" ] && echo dirty || echo clean; } || echo unknown)"
  echo "git         : $GIT_BRANCH@$GIT_SHA ($GIT_DIRTY)"
  echo "关键文件 sha256："
  ( cd "$STAGE" && for f in compress_job.py backend/app/jobs/compression.py backend/app/models.py \
      scripts/check_compress_package.py deploy/compress/examples/welding_eval.py; do
      [ -f "$f" ] && printf '  %-46s %s\n' "$f" "$(sha256sum "$f" | cut -d' ' -f1)"
    done )
  echo "核对方法："
  echo "  1) sha256sum -c SHA256SUMS                     # 包内逐文件校验"
  echo "  2) grep -c _prepare_quant_input backend/app/jobs/compression.py   # ≥1 表示含 ONNX 自动降级链（2026-09-22 之后）"
  echo "  3) python scripts/check_compress_package.py    # 布局/引擎/CLI/进度文件自检"
} > "$STAGE/BUILD_INFO.txt"
cat "$STAGE/BUILD_INFO.txt"

echo "==> 自检：代码包是否自带引擎、compress_job.py 能否直接跑"
CHECK_PY="$REPO_ROOT/.venv/bin/python"
if [ ! -x "$CHECK_PY" ]; then CHECK_PY="$(command -v python3 || true)"; fi
if [ -n "$CHECK_PY" ]; then
  # 本机缺 torch 等依赖时返回 2（只做静态检查），不算打包失败
  if ! "$CHECK_PY" scripts/check_compress_package.py --root "$STAGE"; then
    echo "!! 自检未通过：请修复上面的 FAIL 项后再交付（打包继续，但代码包可能不可直接运行）"
  fi
else
  echo "!! 未找到 python 解释器，跳过自检"
fi

echo "==> 打包 zip"
make_zip() {  # make_zip <srcdir> <outzip> <topname>
  local src="$1" out="$2" top="$3"
  if command -v zip >/dev/null 2>&1; then
    ( cd "$(dirname "$src")" && zip -qr "$out" "$top" )
    return
  fi
  # 无 zip 时用 Python 标准库（保留顶层目录名与相对路径）；
  # 已压缩文件（.tar.gz/.gz/.zip）用 ZIP_STORED，避免无意义地二次压缩大文件
  python3 - "$src" "$out" <<'PY'
import os, sys, zipfile
src, out = sys.argv[1], sys.argv[2]
base = os.path.dirname(os.path.abspath(src))
precompressed = (".gz", ".tgz", ".zip", ".xz", ".bz2", ".whl")
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
    for root, _dirs, files in os.walk(src):
        for name in files:
            full = os.path.join(root, name)
            method = zipfile.ZIP_STORED if name.endswith(precompressed) else zipfile.ZIP_DEFLATED
            archive.write(full, os.path.relpath(full, base), compress_type=method)
print(f"wrote {out}")
PY
}
make_zip "$STAGE" "$DIST/$NAME.zip" "$NAME"
( cd "$DIST" && sha256sum "$NAME.zip" > "$NAME.zip.sha256" )
echo
echo "交付物（环境与代码分两个包）："
echo "  本包：$DIST/$NAME.zip ($(du -h "$DIST/$NAME.zip" | cut -f1))"
echo "        sha256: $(cut -d' ' -f1 "$DIST/$NAME.zip.sha256")"
if [ "$BUNDLE_ONLY" = "0" ] && [ -d "$STAGE/image" ] && [ -n "$(ls -A "$STAGE/image" 2>/dev/null)" ]; then
  echo "  环境包：$DIST/compress-job-cpu-$DATE.tar.gz ($(du -h "$DIST/compress-job-cpu-$DATE.tar.gz" 2>/dev/null | cut -f1))"
  echo "          ↑ 已同时收进全量包的 image/ 目录"
fi
echo "内容概览："
( cd "$STAGE" && find . -maxdepth 2 -type f | sort | head -30 )
