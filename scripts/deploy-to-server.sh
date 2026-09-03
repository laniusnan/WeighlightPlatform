#!/usr/bin/env bash
# ============================================================================
# 一键部署：AI Model Lightweight Platform → CPU-only 服务器
#
# 用法：
#   ./scripts/deploy-to-server.sh [用户@]服务器IP [远端目录]
#   ./scripts/deploy-to-server.sh root@1.2.3.4
#   ./scripts/deploy-to-server.sh root@1.2.3.4 /opt/app
#   export DEPLOY_HOST=root@1.2.3.4 && ./scripts/deploy-to-server.sh
#
# 流程：
#   1) rsync 同步代码（排除 .venv/node_modules/models-storage/前端构建产物等）
#   2) 检查远端 .env.cpu；缺失时从模板生成并提示填写密钥后重跑
#   3) 远端后台构建（scripts/cpu-deploy.sh background，setsid+nohup，断开 SSH 不中断）
#   4) 轮询远端 cpu-build.log 直到构建完成
#   5) 启动服务（db / redis / backend / worker）并打印访问地址
#
# 前置：本机与服务器已配置 SSH 免密登录；服务器装有 Docker + Compose v2。
# ============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST="${1:-${DEPLOY_HOST:-}}"
REMOTE_DIR="${2:-/opt/app}"

say() { printf '\033[1;36m>>\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m!!\033[0m %s\n' "$*" >&2; exit 1; }

if [ -z "$HOST" ]; then
  cat <<'EOF'
用法:
  ./scripts/deploy-to-server.sh [用户@]服务器IP [远端目录(/opt/app)]
  export DEPLOY_HOST=root@1.2.3.4 && ./scripts/deploy-to-server.sh
EOF
  exit 1
fi

EXCLUDES=(
  --exclude=.git
  --exclude=.venv
  --exclude=node_modules
  --exclude=backend/models-storage
  --exclude=backend/app/frontend
  --exclude=backend/.venv
  --exclude=playwright-report
  --exclude=test-results
  --exclude=cpu-build.log
  --exclude=.env
  --exclude='*.log'
)

# 1/5 同步代码
say "1/5 同步代码  ->  ${HOST}:${REMOTE_DIR}"
ssh -o BatchMode=yes "$HOST" "mkdir -p '$REMOTE_DIR'" \
  || die "无法 SSH 到 $HOST（请先配置免密：ssh-copy-id $HOST）"
rsync -az "${EXCLUDES[@]}" "$ROOT_DIR/" "$HOST:$REMOTE_DIR/"

# 2/5 检查远端 .env.cpu
say "2/5 检查远端环境变量 .env.cpu"
if ssh -o BatchMode=yes "$HOST" "test -s '$REMOTE_DIR/.env.cpu'"; then
  say ".env.cpu 已存在，继续。"
else
  ssh -o BatchMode=yes "$HOST" "cd '$REMOTE_DIR' && cp .env.cpu.example .env.cpu"
  cat <<EOF

!! 远端已从模板生成 .env.cpu，请先登录服务器填写三个密钥后重跑本脚本：

    ssh $HOST
    vi $REMOTE_DIR/.env.cpu
      SECRET_KEY=...                  # openssl rand -hex 32
      POSTGRES_PASSWORD=...           # openssl rand -hex 32
      FIRST_SUPERUSER_PASSWORD=...    # openssl rand -hex 32
      FIRST_SUPERUSER=admin@example.com
      FRONTEND_HOST=http://服务器IP:8001
EOF
  exit 1
fi

# 3/5 远端后台构建
say "3/5 远端后台构建（断开本机 SSH 也不中断；日志 $REMOTE_DIR/cpu-build.log）"
ssh -o BatchMode=yes "$HOST" \
  "cd '$REMOTE_DIR' && bash scripts/cpu-deploy.sh background" \
  || die "远端构建未能启动，请手动 ssh 检查"

# 4/5 轮询构建结果（最多 30 分钟）
say "4/5 轮询构建结果（每 20s 一次，最长 30 分钟）"
BUILT=0
for i in $(seq 1 90); do
  sleep 20
  out="$(ssh -o BatchMode=yes "$HOST" \
    "cd '$REMOTE_DIR' && bash scripts/cpu-deploy.sh status 2>&1" || true)"
  if printf '%s' "$out" | grep -q '构建完成'; then
    BUILT=1
    say "镜像构建完成 ✔"
    break
  elif printf '%s' "$out" | grep -q '构建进行中'; then
    printf '    已等待约 %d 分钟，仍在构建…\n' "$((i * 20 / 60))"
    continue
  else
    say "远端没有在构建（可能失败），最近日志："
    ssh -o BatchMode=yes "$HOST" "tail -n 20 '$REMOTE_DIR/cpu-build.log'" || true
    exit 1
  fi
done
if [ "$BUILT" -ne 1 ]; then
  die "等待超时（30 分钟）。请 ssh $HOST 后查看 $REMOTE_DIR/cpu-build.log"
fi

# 5/5 启动服务
say "5/5 启动服务（db / redis / backend / worker）"
ssh -o BatchMode=yes "$HOST" \
  "cd '$REMOTE_DIR' && bash scripts/cpu-deploy.sh up"

HOST_IP="${HOST#*@}"
cat <<EOF

部署完成 ✔
  平台前端     http://$HOST_IP:8001
  Swagger      http://$HOST_IP:8001/docs
  超级管理员    见 $REMOTE_DIR/.env.cpu 的 FIRST_SUPERUSER / FIRST_SUPERUSER_PASSWORD
日志/状态      ssh $HOST "bash $REMOTE_DIR/scripts/cpu-deploy.sh logs"
EOF
