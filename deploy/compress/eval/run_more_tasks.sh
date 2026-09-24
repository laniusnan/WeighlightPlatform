#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Second CIFAR-10 wave: missing baselines/benches + extra methods.
#   bash run_more_tasks.sh
# Everything appends to the same out/ tree as run_cifar_tasks.sh so the report
# can read one directory per task.
# ---------------------------------------------------------------------------
set -uo pipefail

REPO="${REPO:-/home/lanius/Workspace/full-stack-fastapi}"
WORK="${WORK:-/home/lanius/compression-eval}"
DATA="${DATA:-/home/lanius/Dataset}"
IMAGE="${IMAGE:-compress-job:cpu}"
OUT="$WORK/out"
LOG="$WORK/logs"
mkdir -p "$OUT" "$LOG"

MOUNTS=(--user "$(id -u):$(id -g)" -v "$DATA:/datasets:ro" -v "$WORK:/work"
        -v "$REPO/compress_job.py:/app/compress_job.py:ro")
COMMON=(--json-logs --image-size 32 --eval-dataset-path "/datasets/CIFAR10-test")

cli() {  # cli <name> <args...>
  local name="$1"; shift
  docker run --rm "${MOUNTS[@]}" "$IMAGE" --job-id="$name" --output-dir="/work/out/$name" "$@" \
    > "$LOG/$name.log" 2>&1
  echo "$?" > "$OUT/$name.exit"
}

bench() {  # bench <name> [artifact-path] [arch]
  local name="$1"
  local artifact="${2:-/work/out/$name/artifact}"
  local arch="${3:-}"
  local extra=()
  [ -n "$arch" ] && extra=(--arch "$arch")
  docker run --rm "${MOUNTS[@]}" --entrypoint python "$IMAGE" /work/scripts/bench.py \
    --artifact "$artifact" --data /datasets/CIFAR10-test --image-size 32 \
    --samples 2000 --out "/work/out/$name/bench.json" --label "$name" "${extra[@]}" \
    > "$LOG/$name.bench.log" 2>&1
  echo "$?" > "$OUT/$name.bench.exit"
}

echo "== baseline: fp32 ONNX (01-export) =="
bench 01-export /work/out/01-export/artifact/model.onnx

echo "== baseline: pytorch checkpoint (CifarCNN10) =="
bench baseline-pt /work/models/CifarCNN10/model.pt /work/arch/CifarCNN10-arch.py

echo "== 04 OpenVINO INT8 bench (re-run with static-batch support) =="
bench 04-openvino-int8

echo "== 09 L2 pruning, gentle rate 0.1 =="
cli 09-prune-l2-r01 --method=prune --algo=l2 --rate=0.1 \
  --base-model-path=/work/models/CifarCNN10/model.pt --arch-file=/work/models/CifarCNN10/arch.py \
  "${COMMON[@]}"
bench 09-prune-l2-r01

echo "== 10 pipeline: L2 prune(0.3) -> dynamic INT8 =="
cli 10-prune-then-quant --method=pipeline --pipeline "prune:l2,quantize:onnx_dynamic_int8" --rate=0.3 \
  --base-model-path=/work/models/CifarCNN10/model.pt --arch-file=/work/models/CifarCNN10/arch.py \
  "${COMMON[@]}"
bench 10-prune-then-quant

echo "== 11 unstructured sparsify (rate 0.3) =="
cli 11-sparsify --method=prune --algo=sparsify --rate=0.3 \
  --base-model-path=/work/models/CifarCNN10/model.pt --arch-file=/work/models/CifarCNN10/arch.py \
  "${COMMON[@]}"
bench 11-sparsify

echo "== 13 distillation, 10 epochs (best-effort convergence) =="
cli 13-distill-10ep --method=distill --algo=logits --epochs=10 --batch-size=128 --learning-rate=0.001 \
  --base-model-path=/work/models/CifarCNN10/model.pt --arch-file=/work/models/CifarCNN10/arch.py \
  --dataset-path=/datasets/CIFAR10-train "${COMMON[@]}"
bench 13-distill-10ep

echo "== done =="
