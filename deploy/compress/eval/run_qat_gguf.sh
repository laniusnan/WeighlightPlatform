#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# QAT (quantization-aware training) + GGUF export — the two methods that were
# previously not runnable from the CLI / missing offline tooling.
#   bash run_qat_gguf.sh
#   - QAT:  --method=qat (added to the CLI, maps to engine's FX QAT + QDQ export)
#   - GGUF: --method=export --algo=gguf with the on-host llama.cpp copy
#           (/home/lanius/llama.cpp/convert_hf_to_gguf.py) mounted read-only.
# ---------------------------------------------------------------------------
set -uo pipefail

REPO="${REPO:-/home/lanius/Workspace/full-stack-fastapi}"
WORK="${WORK:-/home/lanius/compression-eval}"
DATA="${DATA:-/home/lanius/Dataset}"
MODELS="${MODELS:-/home/lanius/models-local}"
LLAMACPP="${LLAMACPP:-/home/lanius/llama.cpp}"
IMAGE="${IMAGE:-compress-job:cpu}"
OUT="$WORK/out"
LOG="$WORK/logs"

MOUNTS=(--user "$(id -u):$(id -g)" -v "$MODELS:/models:ro" -v "$DATA:/datasets:ro" -v "$WORK:/work"
        -v "$REPO/compress_job.py:/app/compress_job.py:ro")

echo "== 12 QAT (fx_int8, 1 epoch on CIFAR10-train) =="
docker run --rm "${MOUNTS[@]}" "$IMAGE" \
  --job-id=12-qat --method=qat --algo=fx_int8 --epochs=1 --batch-size=64 --learning-rate=0.0005 \
  --base-model-path=/work/models/CifarCNN10/model.pt --arch-file=/work/models/CifarCNN10/arch.py \
  --dataset-path=/datasets/CIFAR10-train --image-size 32 \
  --eval-dataset-path=/datasets/CIFAR10-test --output-dir=/work/out/12-qat --json-logs \
  > "$LOG/12-qat.log" 2>&1
echo "$?" > "$OUT/12-qat.exit"

echo "== 22 GGUF export (Qwen2.5-0.5B -> q4_K_M, offline llama.cpp) =="
docker run --rm "${MOUNTS[@]}" -v "$LLAMACPP:/llamacpp:ro" "$IMAGE" \
  --job-id=22-qwen-gguf --method=export --algo=gguf --outtype=q4_K_M \
  --base-model-path=/models/Qwen2.5-0.5B --llama-cpp-path=/llamacpp \
  --output-dir=/work/out/22-qwen-gguf --json-logs \
  > "$LOG/22-qwen-gguf.log" 2>&1
echo "$?" > "$OUT/22-qwen-gguf.exit"

echo "== benches =="
docker run --rm "${MOUNTS[@]}" --entrypoint python "$IMAGE" /work/scripts/bench.py \
  --artifact /work/out/12-qat/artifact --data /datasets/CIFAR10-test --image-size 32 \
  --samples 2000 --out /work/out/12-qat/bench.json --label 12-qat \
  > "$LOG/12-qat.bench.log" 2>&1
echo "$?" > "$OUT/12-qat.bench.exit"

python3 "$WORK/scripts/collect_results.py" --out "$OUT/summary.json" > "$LOG/collect_qat.log" 2>&1
echo "== done =="
