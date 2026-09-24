#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Continuation after the memory-starved first pass: LLM benchmarks + LoRA.
# (task 20 NNCF INT4 already finished: out/20-qwen-int4)
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

CPU_MOUNTS=(--user "$(id -u):$(id -g)" -v "$MODELS:/models:ro" -v "$DATA:/datasets:ro" -v "$WORK:/work"
            -v "$REPO/compress_job.py:/app/compress_job.py:ro")
GPU_MOUNTS=(--gpus all --user "$(id -u):$(id -g)" -v "$REPO:/workspace:ro" -v "$MODELS:/models:ro"
            -v "$DATA:/datasets:ro" -v "$WORK:/work")

echo "== LoRA SFT (Qwen2.5-0.5B + alpaca-32, GPU) =="
docker run --rm "${GPU_MOUNTS[@]}" --entrypoint python "$GPU_IMAGE" /workspace/compress_job.py \
  --job-id=21-qwen-lora --method=finetune --algo=lora \
  --base-model-path=/models/Qwen2.5-0.5B --dataset-path=/datasets/alpaca-train-32 \
  --output-dir=/work/out/21-qwen-lora --epochs=1 --lora-rank=8 --hf-max-len=128 \
  --learning-rate=0.0003 --device=cuda --json-logs > "$LOG/21-qwen-lora.log" 2>&1
echo "$?" > "$OUT/21-qwen-lora.exit"

echo "== LLM bench: baseline (HF fp32 CPU, decode + PPL) =="
docker run --rm "${CPU_MOUNTS[@]}" --entrypoint python "$CPU_IMAGE" /work/scripts/bench_llm.py \
  --side baseline --model /models/Qwen2.5-0.5B --corpus /work/datasets-derived/alpaca-eval.txt \
  --max-new-tokens 24 --out /work/out/20-qwen-int4/bench_baseline.json \
  > "$LOG/20-qwen-baseline.log" 2>&1
echo "$?" > "$OUT/20-qwen-baseline.exit"

echo "== LLM bench: INT4 artifact (OpenVINO IR, decode + PPL) =="
docker run --rm "${CPU_MOUNTS[@]}" --entrypoint python "$CPU_IMAGE" /work/scripts/bench_llm.py \
  --side artifact --model /work/out/20-qwen-int4/artifact --corpus /work/datasets-derived/alpaca-eval.txt \
  --max-new-tokens 24 --out /work/out/20-qwen-int4/bench_artifact.json \
  > "$LOG/20-qwen-artifact.log" 2>&1
echo "$?" > "$OUT/20-qwen-artifact.exit"

echo "== collect + method cards =="
python3 "$WORK/scripts/collect_results.py" --out "$OUT/summary.json" > "$LOG/collect_final.log" 2>&1
python3 "$WORK/scripts/build_method_cards.py" --summary "$OUT/summary.json" \
  --out "$REPO/frontend/src/components/lab/method-cards.json" >> "$LOG/collect_final.log" 2>&1
echo "== done =="
