#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Run every CIFAR-10 compression method end-to-end on the server and collect
# metrics for docs/compression-eval-report.md.  Real execution only - every
# number in the report comes from the JSON files this script writes.
#
#   bash run_cifar_tasks.sh            # all
#   bash run_cifar_tasks.sh 03 05      # selected ids
# ---------------------------------------------------------------------------
set -uo pipefail

REPO="${REPO:-/home/lanius/Workspace/full-stack-fastapi}"
WORK="${WORK:-/home/lanius/compression-eval}"
DATA="${DATA:-/home/lanius/Dataset}"
IMAGE="${IMAGE:-compress-job:cpu}"
MODEL_DIR="$WORK/models/CifarCNN10"
TRAIN="$DATA/CIFAR10-train"
TEST="$DATA/CIFAR10-test"
ARCH="$MODEL_DIR/arch.py"
OUT="$WORK/out"
LOG="$WORK/logs"
mkdir -p "$OUT" "$LOG"

# CLI overlay: use the repository's current compress_job.py inside the image
OVERLAY=(-v "$REPO/compress_job.py:/app/compress_job.py:ro")
MOUNTS=(--user "$(id -u):$(id -g)" -v "$DATA:/datasets:ro" -v "$WORK:/work" "${OVERLAY[@]}")
COMMON=(--json-logs --image-size 32 --eval-dataset-path "/datasets/CIFAR10-test")

cli() {  # cli <job-id> <out-name> <args...>
  local job="$1" name="$2"; shift 2
  docker run --rm "${MOUNTS[@]}" "$IMAGE" --job-id="$job" --output-dir="/work/out/$name" "$@" \
    > "$LOG/$name.log" 2>&1
  echo "$?" > "$OUT/$name.exit"
}

bench() {  # bench <out-name> [arch]
  local name="$1" arch="${2:-}"
  local extra=()
  [ -n "$arch" ] && extra=(--arch "$arch")
  docker run --rm "${MOUNTS[@]}" --entrypoint python "$IMAGE" /work/scripts/bench.py \
    --artifact "/work/out/$name/artifact" --data /datasets/CIFAR10-test --image-size 32 \
    --samples 2000 --out "/work/out/$name/bench.json" --label "$name" "${extra[@]}" \
    > "$LOG/$name.bench.log" 2>&1
  echo "$?" > "$OUT/$name.bench.exit"
}

want() { [ $# -eq 0 ] && return 0; for id in "$@"; do [ "$id" = "$RUN_ID" ] && return 0; done; return 1; }

# ---------------------------------------------------------------- baseline ---
if [ $# -eq 0 ]; then
  echo "== baseline: pytorch checkpoint (CifarCNN10, trained on server) =="
  bench baseline-pt "$ARCH"
fi

for RUN_ID in "$@"; do
  case "$RUN_ID" in
    03) echo "== 03 static INT8 (calibration: CIFAR10-train, 512 samples) =="
        cli eval-static-int8 03-static-int8 \
          --method=quantize --algo=onnx_static_int8 \
          --base-model-path=/work/out/01-export/artifact/model.onnx \
          --dataset-path=/datasets/CIFAR10-train --max-samples 512 "${COMMON[@]}"
        bench 03-static-int8 ;;
    04) echo "== 04 OpenVINO INT8 (calibration: CIFAR10-train, 512 samples) =="
        cli eval-openvino-int8 04-openvino-int8 \
          --method=quantize --algo=openvino_int8 --precision int8 \
          --base-model-path=/work/out/01-export/artifact/model.onnx \
          --dataset-path=/datasets/CIFAR10-train --max-samples 512 "${COMMON[@]}"
        bench 04-openvino-int8 ;;
    05) echo "== 05 structured L2 pruning (rate=0.3, no data) =="
        cli eval-prune-l2 05-prune-l2 \
          --method=prune --algo=l2 --rate=0.3 \
          --base-model-path=/work/models/CifarCNN10/model.pt --arch-file=/work/models/CifarCNN10/arch.py \
          "${COMMON[@]}"
        bench 05-prune-l2 ;;
    06) echo "== 06 GroupFisher pruning (rate=0.3, calibration 256 imgs) =="
        cli eval-prune-gf 06-prune-groupfisher \
          --method=prune --algo=group_fisher --rate=0.3 \
          --base-model-path=/work/models/CifarCNN10/model.pt --arch-file=/work/models/CifarCNN10/arch.py \
          --dataset-path=/datasets/CIFAR10-train --max-samples 256 "${COMMON[@]}"
        bench 06-prune-groupfisher ;;
    07) echo "== 07 knowledge distillation (logits, 3 epochs) =="
        cli eval-distill 07-distill-logits \
          --method=distill --algo=logits --epochs=3 --batch-size=128 --learning-rate=0.001 \
          --base-model-path=/work/models/CifarCNN10/model.pt --arch-file=/work/models/CifarCNN10/arch.py \
          --dataset-path=/datasets/CIFAR10-train "${COMMON[@]}"
        bench 07-distill-logits ;;
    08) echo "== 08 dynamic INT8 re-run (for a single consolidated table) =="
        cli eval-dynamic-int8 02-dynamic-int8 \
          --method=quantize --algo=onnx_dynamic_int8 \
          --base-model-path=/work/out/01-export/artifact/model.onnx "${COMMON[@]}"
        bench 02-dynamic-int8 ;;
    *) echo "unknown task id: $RUN_ID" >&2 ;;
  esac
done
echo "== done =="
