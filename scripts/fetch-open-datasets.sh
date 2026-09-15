#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Fetch open datasets (CIFAR-10/100, MNIST, Fashion-MNIST) onto the platform
# host and convert them into the flat per-class image folders that data-driven
# pruning, QAT, distillation and classification evaluation consume.
#
# Run once on the server (Docker required - the conversion happens inside the
# platform backend image). Afterwards every split shows up under
# /datasets-local/<name> inside the containers and can be registered by
# reference (no copy) through the model page → 数据集 → 从服务器导入, i.e.
# POST /api/v1/datasets/import-local {"server_path": ..., "mode": "reference"}.
#
#   bash scripts/fetch-open-datasets.sh                       # everything
#   bash scripts/fetch-open-datasets.sh --only cifar10
#   DATASET_ROOT=/data/Dataset bash scripts/fetch-open-datasets.sh --only mnist
#
# Downloads are resumable (curl -C -) and skipped when already present, so the
# script is safe to re-run after a network hiccup.
# ---------------------------------------------------------------------------
set -euo pipefail

ROOT="${DATASET_ROOT:-/home/zhangxuefei/Dataset}"
IMAGE="${CONVERT_IMAGE:-}"
ONLY="${ONLY:-cifar10,cifar100,mnist,fashionmnist}"
MIRROR="${MIRROR:-}"
# hf  : HuggingFace parquet through a fast mirror (default; the original
#       CIFAR/MNIST archives can be unreachable-slow from some networks)
# archive: original tar.gz/idx archives from their upstream hosts
SOURCE="${SOURCE:-hf}"
HF_BASE="${HF_BASE:-${HF_ENDPOINT:-https://hf-mirror.com}}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root) ROOT="$2"; shift 2 ;;
    --image) IMAGE="$2"; shift 2 ;;
    --only) ONLY="$2"; shift 2 ;;
    --mirror) MIRROR="$2"; shift 2 ;;
    --source) SOURCE="$2"; shift 2 ;;
    --hf-base) HF_BASE="$2"; shift 2 ;;
    -h|--help)
      echo "usage: $0 [--root DIR] [--image IMAGE] [--source hf|archive]"
      echo "          [--only cifar10,cifar100,mnist,fashionmnist] [--mirror URL] [--hf-base URL]"
      exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

RAW="$ROOT/_raw"
TOOLS="$ROOT/_tools"
mkdir -p "$RAW" "$TOOLS"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cp "$SCRIPT_DIR/convert_open_datasets.py" "$TOOLS/"
cp "$SCRIPT_DIR/convert_hf_parquet.py" "$TOOLS/"

if [[ -z "$IMAGE" ]]; then
  # Reuse whatever image the platform's own worker/API containers run, so the
  # converter always has polars + Pillow available (no guessing).
  for container in app-worker-1 app-backend-1 mlplatform-gpu-local-worker-1 \
                   mlplatform-gpu-local-backend-1 backend worker; do
    if docker inspect "$container" >/dev/null 2>&1; then
      IMAGE="$(docker inspect "$container" --format '{{.Config.Image}}')"
      echo "[fetch] using image of running container $container: $IMAGE"
      break
    fi
  done
fi
IMAGE="${IMAGE:-backend:latest}"

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  for candidate in backend:latest backend-gpu:latest mlplatform-cpu:latest mlplatform-gpu:latest; do
    if docker image inspect "$candidate" >/dev/null 2>&1; then
      echo "[fetch] image $IMAGE not found, using $candidate for conversion"
      IMAGE="$candidate"
      break
    fi
  done
fi
docker image inspect "$IMAGE" >/dev/null 2>&1 || {
  echo "[fetch] no usable container image found (looked for $IMAGE); aborting" >&2
  exit 1
}
docker run --rm "$IMAGE" python -c "import polars, PIL" >/dev/null 2>&1 || {
  echo "[fetch] image $IMAGE lacks polars/Pillow; pass --image <platform image>" >&2
  exit 1
}

fetch() { # fetch <url> <destination>
  local url="$1" dest="$2"
  if [[ -s "$dest" ]]; then
    echo "[fetch] cached: $(basename "$dest")"
    return 0
  fi
  mkdir -p "$(dirname "$dest")"
  echo "[fetch] $url"
  curl -fL -C - --retry 8 --retry-delay 5 --retry-connrefused \
       --connect-timeout 20 --speed-time 60 --speed-limit 1024 \
       -o "$dest" "$url"
}

if [[ "$SOURCE" == "hf" ]]; then
  # HuggingFace parquet route: fast through a mirror, read with polars inside
  # the platform image (no extra dependency).
  declare -A HF_REPO=(
    [cifar10]="datasets/uoft-cs/cifar10/resolve/main/plain_text"
    [cifar100]="datasets/uoft-cs/cifar100/resolve/main/cifar100"
    [mnist]="datasets/ylecun/mnist/resolve/main/mnist"
    [fashionmnist]="datasets/zalando-datasets/fashion_mnist/resolve/main/fashion_mnist"
  )
  for key in cifar10 cifar100 mnist fashionmnist; do
    [[ "$ONLY" == *"$key"* ]] || continue
    for split in train test; do
      fetch "$HF_BASE/${HF_REPO[$key]}/$split-00000-of-00001.parquet" \
            "$RAW/hf/$key/$split.parquet"
    done
  done
fi

if [[ "$SOURCE" != "hf" && "$ONLY" == *cifar10* ]]; then
  base="${MIRROR:-https://www.cs.toronto.edu/~kriz}"
  fetch "$base/cifar-10-python.tar.gz" "$RAW/cifar-10-python.tar.gz"
  tar -xzf "$RAW/cifar-10-python.tar.gz" -C "$RAW"
fi

if [[ "$SOURCE" != "hf" && "$ONLY" == *cifar100* ]]; then
  base="${MIRROR:-https://www.cs.toronto.edu/~kriz}"
  fetch "$base/cifar-100-python.tar.gz" "$RAW/cifar-100-python.tar.gz"
  tar -xzf "$RAW/cifar-100-python.tar.gz" -C "$RAW"
fi

IDX_FILES=(
  train-images-idx3-ubyte
  train-labels-idx1-ubyte
  t10k-images-idx3-ubyte
  t10k-labels-idx1-ubyte
)

if [[ "$SOURCE" != "hf" && "$ONLY" == *mnist* ]]; then
  base="${MIRROR:-https://ossci-datasets.s3.amazonaws.com/mnist}"
  for name in "${IDX_FILES[@]}"; do
    fetch "$base/$name.gz" "$RAW/mnist/$name.gz"
  done
  for name in "${IDX_FILES[@]}"; do
    gunzip -kf "$RAW/mnist/$name.gz"
  done
fi

if [[ "$SOURCE" != "hf" && "$ONLY" == *fashionmnist* ]]; then
  base="${MIRROR:-http://fashion-mnist.s3-website.eu-central-1.amazonaws.com}"
  for name in "${IDX_FILES[@]}"; do
    fetch "$base/$name.gz" "$RAW/fashion/$name.gz"
  done
  for name in "${IDX_FILES[@]}"; do
    gunzip -kf "$RAW/fashion/$name.gz"
  done
fi

CONVERTER="convert_open_datasets.py"
[[ "$SOURCE" == "hf" ]] && CONVERTER="convert_hf_parquet.py"

echo "[fetch] converting inside $IMAGE via $CONVERTER (a minute per dataset)"
docker run --rm -v "$ROOT":/work -w /work "$IMAGE" \
  python "/work/_tools/$CONVERTER" --root /work --only "$ONLY"

echo
echo "[fetch] datasets prepared under $ROOT (container path /datasets-local):"
for dir in "$ROOT"/*/; do
  name="$(basename "$dir")"
  case "$name" in _*) continue ;; esac
  classes="$(find "$dir" -mindepth 1 -maxdepth 1 -type d | wc -l)"
  images="$(find "$dir" -name '*.png' | wc -l)"
  printf '  %-24s classes=%-4s images=%s\n' "$name" "$classes" "$images"
done

cat <<'HINT'

[fetch] register them (reference mode copies nothing):
  UI : 模型页 → 数据集 → 从服务器导入 → 选择 CIFAR10-train ...
  API: POST /api/v1/datasets/import-local
       {"server_path": "/datasets-local/CIFAR10-train", "mode": "reference"}
HINT
