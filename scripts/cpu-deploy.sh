#!/usr/bin/env bash
# ============================================================================
# CPU 单机部署助手（docker compose build / up）
#
# 为什么需要它：
#   `docker compose ... build` 里的 uv 会并行下载/解压 torch、openvino 等
#   数十个包（合计约 2GB+），在 VS Code 集成终端里跑可能把 WSL/编辑器拖到
#   崩溃。本脚本支持：
#     - build：在「独立系统终端 / SSH」里前台执行（推荐）
#     - background：后台执行（setsid + nohup），关掉 VS Code / 断开 SSH 也不中断
#     - logs / status：查看进度与完成状态
#     - up / down：启动 / 停止服务
#
# 用法示例：
#   ./scripts/cpu-deploy.sh background
#   ./scripts/cpu-deploy.sh logs        # Ctrl+C 只退出查看，不影响构建
#   ./scripts/cpu-deploy.sh status
#   ./scripts/cpu-deploy.sh up
# ============================================================================
set -euo pipefail

# 项目根目录（脚本位于 <root>/scripts/）
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Compose 命令：优先 docker compose(v2)，否则回退 docker-compose(v1)
if docker compose version >/dev/null 2>&1; then
  COMPOSE=(docker compose -f compose.cpu.yml --env-file .env.cpu)
else
  COMPOSE=(docker-compose -f compose.cpu.yml --env-file .env.cpu)
fi

LOG_FILE="$ROOT_DIR/cpu-build.log"
BUILD_MARKER='build:'
PROC_PATTERN='compose.cpu.yml --env-file .env.cpu build'

say() { printf '\033[1;36m[%s]\033[0m %s\n' "$(date +%H:%M:%S)" "$*"; }

case "${1:-}" in
  build)
    say "前台构建（请在独立终端执行，勿在 VS Code 集成终端跑重型构建）"
    exec "${COMPOSE[@]}" build
    ;;

  background)
    if pgrep -f "$PROC_PATTERN" >/dev/null 2>&1; then
      say "构建已在运行中。查看进度： ./scripts/cpu-deploy.sh logs"
      exit 0
    fi
    say "后台构建启动：日志 $LOG_FILE"
    : > "$LOG_FILE"
    # setsid: 脱离当前会话；nohup: 忽略 SIGHUP。两者配合后，
    # 关闭 VS Code / 终端 / 断开 SSH 都不会中断构建。
    setsid nohup "${COMPOSE[@]}" build >>"$LOG_FILE" 2>&1 </dev/null &
    disown || true
    sleep 1
    say "已启动（PID $!）。完成/失败后文件会追加标记，用 status 命令查询。"
    ;;

  logs)
    tail -f "$LOG_FILE"
    ;;

  status)
    if pgrep -f "$PROC_PATTERN" >/dev/null 2>&1; then
      say "构建进行中…（最新日志）"
      tail -5 "$LOG_FILE" || true
    elif grep -qE 'Successfully (built|tagged)|Built backend|backend:latest Built' "$LOG_FILE" 2>/dev/null; then
      say "构建完成 ✔  镜像 backend:latest 可用 → ./scripts/cpu-deploy.sh up"
    else
      say "当前无构建进程；若中途失败请看日志尾部："
      tail -8 "$LOG_FILE" 2>/dev/null || true
    fi
    ;;

  up)
    say "启动服务：db / redis / backend / worker"
    exec "${COMPOSE[@]}" up -d
    ;;

  down)
    say "停止服务（数据卷保留）"
    exec "${COMPOSE[@]}" down
    ;;

  *)
    cat <<'EOF'
用法: ./scripts/cpu-deploy.sh <命令>

  build       前台构建 —— 推荐在独立终端（Windows Terminal / SSH / WSL 原生终端）里执行
  background  后台构建 —— VS Code 或 SSH 断开也不中断；日志见 ./cpu-build.log
  logs        实时查看构建日志（Ctrl+C 不影响后台构建）
  status      查询构建是否完成
  up          构建完成后启动全部服务
  down        停止服务（数据卷保留）

前置要求：先按 deployment-cpu.md 生成并填写 .env.cpu。
EOF
    ;;
esac
