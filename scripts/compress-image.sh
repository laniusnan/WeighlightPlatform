#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# 构建压缩作业镜像并导出成可离线分发的 tar 包。
#
#   scripts/compress-image.sh build          # 构建 compress-job:cpu
#   scripts/compress-image.sh build-fast     # 基于已有平台镜像秒级构建
#   scripts/compress-image.sh build-gpu      # 构建 compress-job:gpu（CUDA 12.6）
#   scripts/compress-image.sh save           # docker save -> dist/compress-job-cpu-<ver>.tar.gz
#   scripts/compress-image.sh load <file>    # 在目标机器上导入
#   scripts/compress-image.sh smoke          # 用镜像跑一次内置 --help/--dry-run 自检
#   scripts/compress-image.sh all            # build + save + smoke
#
# 约定：镜像内入口为 /app/compress_job.py，工作目录 /app；
#       运行作业时把「模型目录」「输出目录」挂载进去即可。
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VERSION="$(date +%Y%m%d)"
CPU_IMAGE="${CPU_IMAGE:-compress-job:cpu}"
GPU_IMAGE="${GPU_IMAGE:-compress-job:gpu}"
FAST_BASE="${FAST_BASE:-backend:cpu-ship}"
DIST_DIR="${DIST_DIR:-$REPO_ROOT/dist}"

usage() {
  grep '^#' "$0" | sed 's/^# \{0,1\}//'
  exit 1
}

build_cpu() {
  echo "==> 构建 $CPU_IMAGE（独立依赖安装，约 5-15 分钟）"
  docker build -f deploy/compress/Dockerfile.cpu -t "$CPU_IMAGE" .
}

build_fast() {
  echo "==> 构建 $CPU_IMAGE（复用基础镜像 $FAST_BASE，秒级）"
  docker build -f deploy/compress/Dockerfile.fast -t "$CPU_IMAGE" --build-arg BASE_IMAGE="$FAST_BASE" .
}

build_gpu() {
  echo "==> 构建 $GPU_IMAGE（CUDA 12.6 wheels）"
  docker build -f deploy/compress/Dockerfile.gpu -t "$GPU_IMAGE" .
}

save_cpu() {
  mkdir -p "$DIST_DIR"
  local out="$DIST_DIR/compress-job-cpu-${VERSION}.tar.gz"
  echo "==> docker save $CPU_IMAGE -> $out"
  docker save "$CPU_IMAGE" | gzip -1 > "$out"
  ( cd "$DIST_DIR" && sha256sum "$(basename "$out")" > "$(basename "$out").sha256" )
  echo "==> 完成：$out（$(du -h "$out" | cut -f1)）"
}

save_gpu() {
  mkdir -p "$DIST_DIR"
  local out="$DIST_DIR/compress-job-gpu-${VERSION}.tar.gz"
  echo "==> docker save $GPU_IMAGE -> $out"
  docker save "$GPU_IMAGE" | gzip -1 > "$out"
  ( cd "$DIST_DIR" && sha256sum "$(basename "$out")" > "$(basename "$out").sha256" )
  echo "==> 完成：$out（$(du -h "$out" | cut -f1)）"
}

load_image() {
  local file="$1"
  echo "==> docker load < $file"
  gzip -dc "$file" | docker load
}

smoke() {
  echo "==> 镜像自检：$CPU_IMAGE"
  docker run --rm "$CPU_IMAGE" --help | head -12
  echo "==> dry-run 参数解析自检"
  docker run --rm "$CPU_IMAGE" --dry-run --job-id=smoke --base-model-path=/tmp/none.pt \
    --output-dir=/tmp/smoke --method=quantize --algo=onnx_dynamic_int8 --json-logs
  echo "==> 自检通过"
}

case "${1:-}" in
  build) build_cpu ;;
  build-fast) build_fast ;;
  build-gpu) build_gpu ;;
  save) save_cpu ;;
  save-gpu) save_gpu ;;
  load) [ $# -ge 2 ] || usage; load_image "$2" ;;
  smoke) smoke ;;
  all) build_cpu; save_cpu; smoke ;;
  fast-all) build_fast; save_cpu; smoke ;;
  *) usage ;;
esac
