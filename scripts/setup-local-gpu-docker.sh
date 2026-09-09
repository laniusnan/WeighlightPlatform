#!/usr/bin/env bash
# 本机 GPU docker 准备 v2：带重试 + 国内镜像回退
set -euo pipefail

PLUGIN_DIR="/usr/local/lib/docker/cli-plugins"
mkdir -p "$PLUGIN_DIR"

install_compose_plugin() {
  echo ">> [compose] downloading docker compose v2 plugin (retry+resume) ..."
  if curl -fL --retry 8 --retry-all-errors -C - \
      "https://github.com/docker/compose/releases/latest/download/docker-compose-linux-x86_64" \
      -o "$PLUGIN_DIR/docker-compose"; then
    chmod +x "$PLUGIN_DIR/docker-compose"
    "$PLUGIN_DIR/docker-compose" version
    return 0
  fi
  echo ">> [compose] GitHub 失败，尝试 docker-ce 镜像仓库安装 ..."
  curl -fsSL https://mirrors.aliyun.com/docker-ce/linux/ubuntu/gpg \
    | gpg --dearmor -o /usr/share/keyrings/docker-ce.gpg
  echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-ce.gpg] https://mirrors.aliyun.com/docker-ce/linux/ubuntu jammy stable" \
    > /etc/apt/sources.list.d/docker-ce.list
  apt-get update
  apt-get install -y docker-compose-plugin
  docker compose version
}

if ! docker compose version >/dev/null 2>&1; then
  install_compose_plugin
fi

if ! command -v nvidia-ctk >/dev/null 2>&1; then
  echo ">> installing nvidia-container-toolkit ..."
  curl -fsSL --retry 5 https://nvidia.github.io/libnvidia-container/gpgkey \
    | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL --retry 5 https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    > /etc/apt/sources.list.d/nvidia-container-toolkit.list
  apt-get update
  apt-get install -y nvidia-container-toolkit
fi

nvidia-ctk runtime configure --runtime=docker
systemctl restart docker
sleep 3

echo "== verify =="
docker compose version
docker info | grep -A2 Runtimes
