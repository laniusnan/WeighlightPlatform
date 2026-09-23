#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# 焊接 Transformer 模型压缩 demo：导出 → INT8 量化 → 精度验收
#
#   deploy/compress/examples/welding_demo.sh <模型包目录> [输出目录] [镜像名]
#
# 例：
#   unzip welding-transformer-package-20260622T0620Z.zip -d /data/welding
#   deploy/compress/examples/welding_demo.sh \
#     /data/welding/welding-transformer-package-20260622T0620Z /data/welding-out compress-job:cpu
#
# 判据：压缩率 > 60% 且 精度损失 < 3%（脚本最后自动判定并打印）
# 详细说明见 examples/compression-demo-welding.md
# ---------------------------------------------------------------------------
set -euo pipefail

PKG="${1:?用法: welding_demo.sh <模型包目录> [输出目录] [镜像名]}"
OUT="${2:-$PWD/welding-out}"
IMAGE="${3:-compress-job:cpu}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

mkdir -p "$OUT"
VIEW=(-v "$PKG:/pkg:ro" -v "$OUT:/out" -v "$REPO_ROOT:/workspace:ro")

run() {  # run <作业参数...>
  docker run --rm "${VIEW[@]}" "$IMAGE" "$@"
}

summarize() {  # summarize <标签> <输出目录>：打印目录内容 / progress.json / progress.log / result.json
  local label="$1" dir="$2"
  echo
  echo "==> $label 输出目录：$dir"
  ( cd "$dir" && ls -1 | sed 's/^/    /' )
  docker run --rm --entrypoint python -i -v "$dir:/out:ro" "$IMAGE" - <<'PY'
import json
import pathlib

root = pathlib.Path("/out")
progress = root / "progress.json"
if progress.is_file():
    d = json.loads(progress.read_text(encoding="utf-8"))
    print("  progress.json : status={} percent={}% step={}/{} elapsed={}s".format(
        d.get("status"), d.get("percent"), d.get("step"), d.get("steps_total"), d.get("elapsed_s")))
    print("                  artifact={}".format(d.get("artifact")))
log = root / "progress.log"
if log.is_file():
    lines = [x for x in log.read_text(encoding="utf-8").splitlines() if x.strip()]
    print("  progress.log  : {} 行（档位由 --progress-interval 决定，默认每 10% 一行）".format(len(lines)))
    for x in lines[-3:]:
        print("      " + x)
result = root / "result.json"
if result.is_file():
    d = json.loads(result.read_text(encoding="utf-8"))
    print("  result.json   : status={} method={}/{} artifact_bytes={}".format(
        d.get("status"), d.get("method"), d.get("algo"), d.get("artifact_bytes")))
    metrics = {k: v for k, v in (d.get("metrics") or {}).items()
               if isinstance(v, (int, float, str, bool))}
    if metrics:
        print("  metrics       : " + json.dumps(metrics, ensure_ascii=False))
PY
}

echo "==> [1/3] PyTorch -> FP32 ONNX"
run --job-id=weld-01 --tenant-id=welding --trace-id=demo-1 \
  --base-model-path=/pkg/model/welding_transformer.pt \
  --arch-file=/pkg/scripts/welding_transformer.py \
  --output-dir=/out/w01-fp32-onnx \
  --method=export --algo=onnx --input-shape=1x7 --model-kind=tabular --opset=17 \
  --no-json-logs

echo "==> [2/3] FP32 ONNX -> INT8 ONNX（权重动态量化）"
run --job-id=weld-02 --tenant-id=welding --trace-id=demo-2 \
  --base-model-path=/out/w01-fp32-onnx/artifact/model.onnx \
  --output-dir=/out/w02-int8-onnx \
  --method=quantize --algo=onnx_dynamic_int8 \
  --no-json-logs

echo "==> [3/3] 精度/延迟验收（与训练一致的测试集切分）"
# 镜像 ENTRYPOINT 是 compress_job.py，跑评测脚本需显式指定 entrypoint
docker run --rm --entrypoint python "${VIEW[@]}" "$IMAGE" \
  /app/examples/welding_eval.py \
  --csv  /pkg/data/training/ds_41631c12a3_welding_training.csv \
  --config /pkg/model/welding_transformer_config.json \
  --arch /pkg/scripts/welding_transformer.py \
  --baseline /pkg/model/welding_transformer.pt \
  --candidate /out/w02-int8-onnx/artifact/model_quantized.onnx \
  --out /out/welding-eval.json --latency-samples 2000

echo
echo "产物："
ls -la "$OUT/w02-int8-onnx/artifact/"
echo "报告：$OUT/w02-int8-onnx/result.json, $OUT/welding-eval.json"

# 输出目录规则：所有产物/报告/日志/进度都在 --output-dir 下
summarize "步骤 1（export:onnx）" "$OUT/w01-fp32-onnx"
summarize "步骤 2（quantize:onnx_dynamic_int8）" "$OUT/w02-int8-onnx"
echo
echo "日志与进度（两条腿）："
echo "  * stdout 与 <输出目录>/logs.jsonl 内容一致（--json-logs 只管 stdout 格式）"
echo "  * <输出目录>/progress.log 每 10% 一行（--progress-interval 可改粒度），可 tail -f"
echo "  * <输出目录>/progress.json 为最新快照（失败/中断也会落盘 status=failed/interrupted）"
echo "  * 输出目录参数统一为 --output-dir（无 --out/--outdir 别名）"
