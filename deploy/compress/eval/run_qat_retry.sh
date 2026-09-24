#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# QAT retry: the first attempt was OOM-killed (exit 137) and additionally lost
# the FX-QAT path because `--user 1000` has no passwd entry inside the image
# ("getpwuid(): uid not found: 1000" → fell back to fine-tune + PTQ).
#
# Fix: run as root (passwd entry exists) and bound memory with a 4k-image
# symlink subset of CIFAR10-train (documented as derived data).
# ---------------------------------------------------------------------------
set -uo pipefail

WORK="${WORK:-/home/lanius/compression-eval}"
DATA="${DATA:-/home/lanius/Dataset}"
IMAGE="${IMAGE:-compress-job:cpu}"
REPO="${REPO:-/home/lanius/Workspace/full-stack-fastapi}"
DERIVED="$WORK/datasets-derived"
OUT="$WORK/out"
LOG="$WORK/logs"

# ---- 4k subset (400 images per class, symlinks only) ----
SUBSET="$DERIVED/CIFAR10-train-4k"
if [ ! -d "$SUBSET" ]; then
  for class_dir in "$DATA"/CIFAR10-train/*/; do
    class="$(basename "$class_dir")"
    mkdir -p "$SUBSET/$class"
    find "$class_dir" -maxdepth 1 -type f | sort | head -400 | while read -r image; do
      ln -sf "$image" "$SUBSET/$class/$(basename "$image")"
    done
  done
fi
echo "subset files: $(find "$SUBSET" -type f | wc -l)"

echo "== 12 QAT retry (root, 4k subset) =="
docker run --rm -v "$DATA:/datasets:ro" -v "$WORK:/work" \
  -v "$REPO/compress_job.py:/app/compress_job.py:ro" "$IMAGE" \
  --job-id=12-qat --method=qat --algo=fx_int8 --epochs=1 --batch-size=64 --learning-rate=0.0005 \
  --base-model-path=/work/models/CifarCNN10/model.pt --arch-file=/work/models/CifarCNN10/arch.py \
  --dataset-path=/work/datasets-derived/CIFAR10-train-4k --image-size 32 \
  --eval-dataset-path=/datasets/CIFAR10-test --output-dir=/work/out/12-qat --json-logs \
  > "$LOG/12-qat.log" 2>&1
echo "$?" > "$OUT/12-qat.exit"

docker run --rm -v "$DATA:/datasets:ro" -v "$WORK:/work" --entrypoint python "$IMAGE" \
  /work/scripts/bench.py --artifact /work/out/12-qat/artifact \
  --data /datasets/CIFAR10-test --image-size 32 --samples 2000 \
  --out /work/out/12-qat/bench.json --label 12-qat > "$LOG/12-qat.bench.log" 2>&1
echo "$?" > "$OUT/12-qat.bench.exit"

python3 "$WORK/scripts/collect_results.py" --out "$OUT/summary.json" > "$LOG/collect_qat2.log" 2>&1
echo "== done =="
