#!/usr/bin/env bash
# ============================================================================
# 本地构建 CPU 镜像 → 传到服务器 → 直接部署（不在服务器上构建）
#
# 为什么：服务器上 `uv sync` 需要从 PyPI 拉 2GB+ 依赖（torch/openvino/...），
# 出网抖动时（hatchling 下载 30s 超时）会反复构建失败。本地网络好，构建完成
# 后把镜像整体搬运过去即可；服务器只做 `docker load` + `compose up -d --no-build`。
#
# 用法：
#   bash scripts/cpu-image-ship.sh                      # 默认 root@10.110.1.36
#   bash scripts/cpu-image-ship.sh --host root@10.110.1.36 --no-up
#   bash scripts/cpu-image-ship.sh --skip-build         # 复用已有 backend:cpu-ship
#   bash scripts/cpu-image-ship.sh --keep-archive       # 保留本地 tar
#
# 前置：本地与服务器同为 linux/amd64；服务器上存在 /opt/app/compose.cpu.yml
# 与 .env.cpu（沿用 cpu-deploy.sh 的约定）。
# ============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

HOST="${SHIP_HOST:-root@10.110.1.36}"
REMOTE_DIR="${SHIP_REMOTE_DIR:-/opt/app}"
IMAGE="${SHIP_IMAGE:-backend:latest}"
SHIP_TAG="${SHIP_TAG:-backend:cpu-ship}"
ARCHIVE_DIR="${SHIP_ARCHIVE_DIR:-/home/lanius/refs}"
WORKER_CONTAINERS="${SHIP_WORKER_CONTAINERS:-app-worker-1 app-backend-1}"
DO_BUILD=1
DO_UP=1
KEEP_ARCHIVE=0

say() { printf '\033[1;36m[ship]\033[0m %s\n' "$*"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST="$2"; shift 2 ;;
    --remote-dir) REMOTE_DIR="$2"; shift 2 ;;
    --image) IMAGE="$2"; shift 2 ;;
    --skip-build) DO_BUILD=0; shift ;;
    --no-up) DO_UP=0; shift ;;
    --keep-archive) KEEP_ARCHIVE=1; shift ;;
    -h|--help)
      echo "usage: $0 [--host user@server] [--remote-dir /opt/app] [--image backend:latest]"
      echo "          [--skip-build] [--no-up] [--keep-archive]"
      exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

command -v docker >/dev/null || { echo "docker not found" >&2; exit 1; }
mkdir -p "$ARCHIVE_DIR"

# --- 1. build locally -------------------------------------------------------
if [[ "$DO_BUILD" == "1" ]]; then
  say "building $SHIP_TAG from backend/Dockerfile.cpu (cache makes this quick)"
  docker build -f backend/Dockerfile.cpu -t "$SHIP_TAG" .
else
  docker image inspect "$SHIP_TAG" >/dev/null 2>&1 || {
    echo "image $SHIP_TAG not found locally (drop --skip-build)" >&2; exit 1; }
  say "reusing existing $SHIP_TAG"
fi

IMAGE_ID="$(docker image inspect "$SHIP_TAG" --format '{{.Id}}')"
say "built image $IMAGE_ID ($(docker image inspect "$SHIP_TAG" --format '{{.Size}}') bytes)"

# --- 2. save + compress -----------------------------------------------------
USE_ZSTD=0
if command -v zstd >/dev/null 2>&1 && ssh "$HOST" 'command -v zstd' >/dev/null 2>&1; then
  USE_ZSTD=1
fi
if [[ "$USE_ZSTD" == "1" ]]; then
  ARCHIVE="$ARCHIVE_DIR/${SHIP_TAG//[:\/]/_}.tar.zst"
  say "saving with zstd -> $ARCHIVE"
  docker save "$SHIP_TAG" | zstd -3 -T0 -q -f -o "$ARCHIVE"
else
  ARCHIVE="$ARCHIVE_DIR/${SHIP_TAG//[:\/]/_}.tar.gz"
  say "saving with gzip -> $ARCHIVE"
  docker save "$SHIP_TAG" | gzip -1 > "$ARCHIVE"
fi
say "archive size: $(du -h "$ARCHIVE" | cut -f1)"


# --- 3. transfer (resumable) ------------------------------------------------
REMOTE_ARCHIVE="$REMOTE_DIR/$(basename "$ARCHIVE")"
say "rsync -> $HOST:$REMOTE_ARCHIVE"
rsync -avP --partial --inplace "$ARCHIVE" "$HOST:$REMOTE_ARCHIVE"

# --- 4. load + tag + deploy on the server -----------------------------------
say "loading on $HOST and switching $IMAGE (previous image kept as $IMAGE-prev)"
ssh "$HOST" bash -s <<REMOTE
set -euo pipefail
cd '$REMOTE_DIR'

# Keep a rollback tag pointing at whatever is running right now.
prev=""
for c in $WORKER_CONTAINERS; do
  if docker inspect "\$c" >/dev/null 2>&1; then
    prev="\$(docker inspect --format '{{.Image}}' "\$c")"
    break
  fi
done
if [ -n "\$prev" ]; then
  docker tag "\$prev" "$IMAGE-prev" 2>/dev/null || true
  echo "[ship] rollback tag $IMAGE-prev -> \$prev"
fi

if [[ "$USE_ZSTD" == "1" ]]; then
  zstd -dc "$REMOTE_ARCHIVE" | docker load
else
  gunzip -c "$REMOTE_ARCHIVE" | docker load
fi
docker tag "$SHIP_TAG" "$IMAGE"
echo "[ship] $IMAGE -> \$(docker image inspect "$IMAGE" --format '{{.Id}}')"
REMOTE

if [[ "$DO_UP" == "1" ]]; then
  say "recreating containers with the shipped image (no build on the server)"
  ssh "$HOST" "cd '$REMOTE_DIR' && docker compose -f compose.cpu.yml --env-file .env.cpu up -d --no-build"
fi

# --- 5. verify --------------------------------------------------------------
say "verifying"
ssh "$HOST" bash -s <<REMOTE
set -euo pipefail
docker ps --format '{{.Names}} {{.Status}}' | grep -E 'app-backend|app-worker' || true
for c in $WORKER_CONTAINERS; do
  docker inspect "\$c" >/dev/null 2>&1 || continue
  echo "\$c image=\$(docker inspect --format '{{.Config.Image}}' "\$c") id=\$(docker inspect --format '{{.Image}}' "\$c" | cut -c8-19)"
done
REMOTE

say "shipped image id: ${IMAGE_ID:7:12}"
if [[ "$KEEP_ARCHIVE" != "1" ]]; then
  rm -f "$ARCHIVE"
  ssh "$HOST" "rm -f '$REMOTE_ARCHIVE'"
  say "archives removed (use --keep-archive to keep them)"
else
  say "kept $ARCHIVE and $HOST:$REMOTE_ARCHIVE"
fi
say "done"
