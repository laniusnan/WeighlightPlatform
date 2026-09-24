#!/usr/bin/env bash
# QAT retry #2: the symlink subset broke inside the container (absolute targets
# pointed at /home/lanius/Dataset, which is mounted at /datasets there), so the
# 4k subset is materialised with real copies instead. Runs as root because the
# FX-QAT path needs a passwd entry (uid 1000 does not exist in the image).
set -uo pipefail

WORK="${WORK:-/home/lanius/compression-eval}"
DATA="${DATA:-/home/lanius/Dataset}"
IMAGE="${IMAGE:-compress-job:cpu}"
REPO="${REPO:-/home/lanius/Workspace/full-stack-fastapi}"
SUBSET="$WORK/datasets-derived/CIFAR10-train-4k"
OUT="$WORK/out"
LOG="$WORK/logs"

rm -rf "$SUBSET"
for class_dir in "$DATA"/CIFAR10-train/*/; do
  class="$(basename "$class_dir")"
  mkdir -p "$SUBSET/$class"
  find "$class_dir" -maxdepth 1 -type f | sort | head -400 | xargs -r -I{} cp {} "$SUBSET/$class/"
done
echo "subset files: $(find "$SUBSET" -type f | wc -l)  size: $(du -sh "$SUBSET" | cut -f1)"

echo "== 12 QAT (root, real 4k subset) =="
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

python3 "$WORK/scripts/collect_results.py" --out "$OUT/summary.json" > "$LOG/collect_qat3.log" 2>&1
python3 "$WORK/scripts/build_method_cards.py" --summary "$OUT/summary.json" \
  --out "$REPO/frontend/src/components/lab/method-cards.json" >> "$LOG/collect_qat3.log" 2>&1
echo "== done =="
