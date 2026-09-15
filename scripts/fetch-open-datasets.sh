#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Fetch open datasets onto the platform host and convert them into the layouts
# the platform consumes:
#
#   images  CIFAR-10/100, MNIST, Fashion-MNIST -> per-class PNG folders
#           (data-driven pruning, QAT, distillation, classification eval)
#   text    WikiText-2 raw   -> <split>.txt   (real-corpus perplexity)
#           Alpaca / GSM8K   -> *.jsonl       (LoRA SFT)
#           HH-RLHF          -> preferences.jsonl (prompt/chosen/rejected, DPO)
#
# Run once on the server (Docker required - the conversion happens inside the
# platform backend image). Afterwards every split shows up under
# /datasets-local/<name> inside the containers and can be registered by
# reference (no copy) through the model page → 数据集 → 从服务器导入, i.e.
# POST /api/v1/datasets/import-local {"server_path": ..., "mode": "reference"}.
#
#   bash scripts/fetch-open-datasets.sh                              # all
#   bash scripts/fetch-open-datasets.sh --only cifar10,mnist
#   bash scripts/fetch-open-datasets.sh --only wikitext2,alpaca
#   DATASET_ROOT=/data/Dataset bash scripts/fetch-open-datasets.sh --only gsm8k
#
# Downloads are resumable (curl -C -) and skipped when already present, so the
# script is safe to re-run after a network hiccup.
# ---------------------------------------------------------------------------
set -euo pipefail

ROOT="${DATASET_ROOT:-/home/zhangxuefei/Dataset}"
IMAGE="${CONVERT_IMAGE:-}"
ONLY="${ONLY:-cifar10,cifar100,mnist,fashionmnist,wikitext2,alpaca}"
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
      echo "          [--only images:cifar10,cifar100,mnist,fashionmnist"
      echo "                 text:wikitext2,alpaca,gsm8k,hhrlhf]"
      echo "          [--mirror URL] [--hf-base URL]"
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
cp "$SCRIPT_DIR/convert_hf_text.py" "$TOOLS/"

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

# Image-classification datasets (per-class PNG folders).
IMAGE_KEYS=(cifar10 cifar100 mnist fashionmnist)
# Text datasets (corpus / instructions / preference pairs).
TEXT_KEYS=(wikitext2 alpaca gsm8k hhrlhf)

fetch_text_dataset() { # fetch_text_dataset <key>
  local key="$1" split
  case "$key" in
    wikitext2)
      # Real-corpus perplexity evaluation (WikiText-2 raw).
      for split in train validation test; do
        fetch "$HF_BASE/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/$split-00000-of-00001.parquet" \
              "$RAW/hf/wikitext2/$split.parquet"
      done ;;
    alpaca)
      # Instruction tuning (LoRA SFT).
      fetch "$HF_BASE/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet" \
            "$RAW/hf/alpaca/train.parquet" ;;
    gsm8k)
      # Grade-school math reasoning (SFT / evals).
      for split in train test; do
        fetch "$HF_BASE/datasets/openai/gsm8k/resolve/main/main/$split-00000-of-00001.parquet" \
              "$RAW/hf/gsm8k/$split.parquet"
      done ;;
    hhrlhf)
      # Human preference pairs (DPO) - upstream ships gzipped JSONL.
      for split in train test; do
        fetch "$HF_BASE/datasets/Anthropic/hh-rlhf/resolve/main/helpful-base/$split.jsonl.gz" \
              "$RAW/hf/hhrlhf/$split.jsonl.gz"
      done ;;
  esac
}

if [[ "$SOURCE" == "hf" ]]; then
  # HuggingFace route: fast through a mirror, read with polars inside the
  # platform image (no extra dependency).
  declare -A HF_REPO=(
    [cifar10]="datasets/uoft-cs/cifar10/resolve/main/plain_text"
    [cifar100]="datasets/uoft-cs/cifar100/resolve/main/cifar100"
    [mnist]="datasets/ylecun/mnist/resolve/main/mnist"
    [fashionmnist]="datasets/zalando-datasets/fashion_mnist/resolve/main/fashion_mnist"
  )
  for key in "${IMAGE_KEYS[@]}"; do
    [[ "$ONLY" == *"$key"* ]] || continue
    for split in train test; do
      fetch "$HF_BASE/${HF_REPO[$key]}/$split-00000-of-00001.parquet" \
            "$RAW/hf/$key/$split.parquet"
    done
  done
  for key in "${TEXT_KEYS[@]}"; do
    [[ "$ONLY" == *"$key"* ]] || continue
    fetch_text_dataset "$key"
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

# Split the requested keys into the image / text converters.
IMAGE_ONLY=""
for key in "${IMAGE_KEYS[@]}"; do
  [[ "$ONLY" == *"$key"* ]] && IMAGE_ONLY="${IMAGE_ONLY:+$IMAGE_ONLY,}$key"
done
TEXT_ONLY=""
for key in "${TEXT_KEYS[@]}"; do
  [[ "$ONLY" == *"$key"* ]] && TEXT_ONLY="${TEXT_ONLY:+$TEXT_ONLY,}$key"
done

run_converter() { # run_converter <script> <comma-separated keys>
  [[ -n "$2" ]] || return 0
  echo "[fetch] converting ({$2}) inside $IMAGE via $1"
  docker run --rm -v "$ROOT":/work -w /work "$IMAGE" \
    python "/work/_tools/$1" --root /work --only "$2"
}

if [[ "$SOURCE" == "hf" ]]; then
  run_converter convert_hf_parquet.py "$IMAGE_ONLY"
  run_converter convert_hf_text.py "$TEXT_ONLY"
else
  run_converter "$CONVERTER" "$IMAGE_ONLY"
fi

echo
echo "[fetch] datasets prepared under $ROOT (container path /datasets-local):"
for dir in "$ROOT"/*/; do
  name="$(basename "$dir")"
  case "$name" in _*) continue ;; esac
  classes="$(find "$dir" -mindepth 1 -maxdepth 1 -type d | wc -l)"
  images="$(find "$dir" -name '*.png' | wc -l)"
  files="$(find "$dir" -maxdepth 1 -type f | wc -l)"
  size="$(du -sh "$dir" | cut -f1)"
  printf '  %-24s classes=%-4s images=%-7s files=%-4s size=%s\n' \
    "$name" "$classes" "$images" "$files" "$size"
done

cat <<'HINT'

[fetch] register them (reference mode copies nothing):
  UI : 模型页 → 数据集 → 从服务器导入 → 选择 CIFAR10-train / wikitext2-raw-test ...
  API: POST /api/v1/datasets/import-local
       {"server_path": "/datasets-local/CIFAR10-train", "mode": "reference"}

[fetch] downstream usage:
  prune(group_fisher/dmcp/autoslim) / qat / distill  : dataset_id = CIFAR10-train
  classification eval                                : dataset_id = CIFAR10-test
  perplexity eval (real corpus)                      : dataset_id = wikitext2-raw-test
  llm_finetune (LoRA SFT)                            : dataset_id = alpaca-train / gsm8k-train
  llm_dpo (preference pairs)                         : dataset_id = hh-rlhf-train
HINT
