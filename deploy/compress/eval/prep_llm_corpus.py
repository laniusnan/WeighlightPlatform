#!/usr/bin/env python3
"""Build the perplexity corpus used for the LLM evaluation.

There is **no** wikitext2 on this server (checked during the asset scan), so the
corpus is derived from the server-local alpaca split: instruction + input + output
of N rows concatenated into plain text. The derivation is recorded in
docs/compression-eval-report.md so nobody mistakes it for wikitext.

    python prep_llm_corpus.py --source /datasets/alpaca-train-32/alpaca.jsonl \
        --out /work/datasets-derived/alpaca-eval.txt --rows 32
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--rows", type=int, default=32)
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.source).read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = rows[: args.rows]
    text = "\n".join(
        " ".join(str(row.get(key, "") or "").strip() for key in ("instruction", "input", "output")).strip()
        for row in selected
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(json.dumps({"source": args.source, "rows": len(selected), "out": str(out),
                      "bytes": out.stat().st_size, "chars": len(text)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
