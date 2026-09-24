#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# QAT long run (requested): 10 epochs on the FULL CIFAR10-train (50k images).
# A 10-epoch run on the 4k subset follows as a guaranteed fallback; the first
# attempt on 50k was OOM-killed earlier (exit 137), so failures stay recorded.
#   bash run_qat_long.sh
# ---------------------------------------------------------------------------
set -uo pipefail

WORK="${WORK:-/home/lanius/compression-eval}"
DATA="${DATA:-/home/lanius/Dataset}"
IMAGE="${IMAGE:-compress-job:cpu}"
REPO="${REPO:-/home/lanius/Workspace/full-stack-fastapi}"
OUT="$WORK/out"
LOG="$WORK/logs"

run_qat() {  # run_qat <task> <data-dir> <epochs>
  local task="$1" data_dir="$2" epochs="$3"
  echo "== $task: $epochs epochs on $data_dir =="
  docker run --rm -v "$DATA:/datasets:ro" -v "$WORK:/work" \
    -v "$REPO/compress_job.py:/app/compress_job.py:ro" "$IMAGE" \
    --job-id="$task" --method=qat --algo=fx_int8 --epochs="$epochs" --batch-size=64 \
    --learning-rate=0.0005 \
    --base-model-path=/work/models/CifarCNN10/model.pt --arch-file=/work/models/CifarCNN10/arch.py \
    --dataset-path="$data_dir" --image-size 32 \
    --eval-dataset-path=/datasets/CIFAR10-test --output-dir="/work/out/$task" --json-logs \
    > "$LOG/$task.log" 2>&1
  echo "$?" > "$OUT/$task.exit"
  docker run --rm -v "$DATA:/datasets:ro" -v "$WORK:/work" --entrypoint python "$IMAGE" \
    /work/scripts/bench.py --artifact "/work/out/$task/artifact" \
    --data /datasets/CIFAR10-test --image-size 32 --samples 2000 \
    --out "/work/out/$task/bench.json" --label "$task" > "$LOG/$task.bench.log" 2>&1
  echo "$?" > "$OUT/$task.bench.exit"
}

# Primary target: full data, 10 epochs (may be OOM-killed on a 16 GB host).
run_qat 12-qat-10ep-full /datasets/CIFAR10-train 10
# Fallback / comparison: subset, 10 epochs.
run_qat 12-qat-10ep-4k /work/datasets-derived/CIFAR10-train-4k 10

python3 "$WORK/scripts/collect_results.py" --out "$OUT/summary.json" > "$LOG/collect_qat_long.log" 2>&1
echo "== done =="
