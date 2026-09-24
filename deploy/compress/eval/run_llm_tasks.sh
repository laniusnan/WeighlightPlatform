#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# LLM tasks on the server-local Qwen2.5-0.5B:
#   20  NNCF INT4 weight compression (OpenVINO IR)  — CPU, no calibration data
#   21  LoRA SFT on the server-local alpaca split  — GPU (CUDA required)
#   22  decode latency / throughput / perplexity for baseline vs INT4
# Results land in the same out/<task>/ tree used by the CIFAR runs.
# ---------------------------------------------------------------------------
set -uo pipefail

REPO="${REPO:-/home/lanius/Workspace/full-stack-fastapi}"
WORK="${WORK:-/home/lanius/compression-eval}"
DATA="${DATA:-/home/lanius/Dataset}"
MODELS="${MODELS:-/home/lanius/models-local}"
CPU_IMAGE="${CPU_IMAGE:-compress-job:cpu}"
GPU_IMAGE="${GPU_IMAGE:-b121980fea3c}"
OUT="$WORK/out"
LOG="$WORK/logs"
mkdir -p "$OUT" "$LOG"

CPU_MOUNTS=(--user "$(id -u):$(id -g)" -v "$MODELS:/models:ro" -v "$DATA:/datasets:ro" -v "$WORK:/work"
            -v "$REPO/compress_job.py:/app/compress_job.py:ro")
GPU_MOUNTS=(--gpus all --user "$(id -u):$(id -g)" -v "$REPO:/workspace:ro" -v "$MODELS:/models:ro"
            -v "$DATA:/datasets:ro" -v "$WORK:/work" -v "$HOME/.cache/huggingface:/root/.cache/huggingface")

echo "== 20 NNCF INT4 (Qwen2.5-0.5B, CPU) =="
docker run --rm "${CPU_MOUNTS[@]}" "$CPU_IMAGE" \
  --job-id=20-qwen-int4 --method=quantize --algo=nncf_int4 --bits=4 --group-size=128 \
  --base-model-path=/models/Qwen2.5-0.5B --output-dir=/work/out/20-qwen-int4 \
  --eval-dataset-path=/work/datasets-derived/alpaca-eval.txt --json-logs \
  > "$LOG/20-qwen-int4.log" 2>&1
echo "$?" > "$OUT/20-qwen-int4.exit"

echo "== 22 LLM bench: baseline (HF fp32, CPU) =="
docker run --rm "${CPU_MOUNTS[@]}" --entrypoint python "$CPU_IMAGE" /work/scripts/bench_llm.py \
  --side baseline --model /models/Qwen2.5-0.5B --corpus /work/datasets-derived/alpaca-eval.txt \
  --out /work/out/20-qwen-int4/bench_baseline.json > "$LOG/20-qwen-baseline.log" 2>&1
echo "$?" > "$OUT/20-qwen-baseline.exit"

echo "== 22 LLM bench: INT4 artifact (OpenVINO IR, CPU) =="
docker run --rm "${CPU_MOUNTS[@]}" --entrypoint python "$CPU_IMAGE" /work/scripts/bench_llm.py \
  --side artifact --model /work/out/20-qwen-int4/artifact --corpus /work/datasets-derived/alpaca-eval.txt \
  --out /work/out/20-qwen-int4/bench_artifact.json > "$LOG/20-qwen-artifact.log" 2>&1
echo "$?" > "$OUT/20-qwen-artifact.exit"

echo "== 21 LoRA SFT (Qwen2.5-0.5B + alpaca-32, GPU) =="
docker run --rm "${GPU_MOUNTS[@]}" --entrypoint python "$GPU_IMAGE" /workspace/compress_job.py \
  --job-id=21-qwen-lora --method=finetune --algo=lora \
  --base-model-path=/models/Qwen2.5-0.5B --dataset-path=/datasets/alpaca-train-32 \
  --output-dir=/work/out/21-qwen-lora --epochs=1 --lora-rank=8 --hf-max-len=128 \
  --learning-rate=0.0003 --device=cuda --json-logs > "$LOG/21-qwen-lora.log" 2>&1
echo "$?" > "$OUT/21-qwen-lora.exit"

echo "== done =="
