#!/usr/bin/env python3
"""Convert HuggingFace *text* parquet datasets into platform-ready corpora.

Companion of ``convert_hf_parquet.py`` (images). The platform consumes text
directly from a dataset directory, so each split becomes its own importable
folder:

    wikitext-2-raw-v1/{train,validation,test} -> <root>/wikitext2-raw-<split>/<split>.txt
    alpaca (instruction/input/output)        -> <root>/alpaca-<split>/alpaca.jsonl
    gsm8k  (question/answer)                 -> <root>/gsm8k-<split>/gsm8k.jsonl
    hh-rlhf (chosen/rejected dialogues)      -> <root>/hh-rlhf-<split>/preferences.jsonl

Downstream usage:

* ``wikitext2-raw-test`` -> ``llm_quantize``/``export_gguf`` task with
  ``eval = {"type": "perplexity", "dataset_id": ...}`` (real-corpus PPPL);
* ``alpaca-*`` / ``gsm8k-*`` -> ``llm_finetune`` (LoRA SFT) ``dataset_id``;
* ``hh-rlhf-*`` -> ``llm_dpo`` ``dataset_id`` (prompt/chosen/rejected rows).

Run inside the platform image (polars is already there)::

    docker run --rm -v /home/zhangxuefei/Dataset:/work -w /work \\
        backend:cpu python /work/_tools/convert_hf_text.py --root /work

Conversion is idempotent: a split whose output file already exists (and is not
empty) is skipped.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import polars as pl

# key -> (output prefix, kind, splits)
DATASETS: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "wikitext2": ("wikitext2-raw", "corpus", ("train", "validation", "test")),
    "alpaca": ("alpaca", "instructions", ("train",)),
    "gsm8k": ("gsm8k", "instructions", ("train", "test")),
    "hhrlhf": ("hh-rlhf", "preferences", ("train", "test")),
}


def log(message: str) -> None:
    print(f"[text] {message}", flush=True)


def _parquet_path(raw: Path, key: str, split: str) -> Path:
    """Locate ``<raw>/<key>/<split>.parquet`` (mirrors the fetch script)."""
    return raw / key / f"{split}.parquet"


def _source_path(raw: Path, key: str, split: str) -> Path | None:
    """Return the downloaded source file for a split (parquet or jsonl.gz)."""
    for name in (f"{split}.parquet", f"{split}.jsonl.gz", f"{split}.jsonl"):
        candidate = raw / key / name
        if candidate.is_file():
            return candidate
    return None


def _first_existing(frame: pl.DataFrame, names: tuple[str, ...]) -> str | None:
    for name in names:
        if name in frame.columns:
            return name
    return None


def _split_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _iter_records(path: Path) -> Iterator[dict[str, Any]]:
    """Yield row dicts from a parquet file or a (optionally gzipped) JSONL file."""
    if path.name.endswith((".jsonl", ".jsonl.gz", ".json.gz")):
        opener = gzip.open if path.name.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as handle:  # type: ignore[operator]
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if isinstance(record, dict):
                    yield record
        return
    frame = pl.read_parquet(path)
    yield from frame.iter_rows(named=True)


def convert_corpus(path: Path, out_file: Path, text_col: str) -> dict[str, Any]:
    """wikitext-style ``{"text": ...}`` rows -> a single ``.txt`` file."""
    frame = pl.read_parquet(path, columns=[text_col])
    lines = [_split_text(value) for value in frame[text_col].to_list()]
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("\n".join(lines), encoding="utf-8")
    return {"rows": len(lines), "bytes": out_file.stat().st_size}


def convert_instructions(path: Path, out_file: Path) -> dict[str, Any]:
    """instruction/input/output (+question/answer) rows -> SFT jsonl."""
    frame = pl.read_parquet(path)
    instruction_col = _first_existing(frame, ("instruction", "question", "prompt"))
    output_col = _first_existing(frame, ("output", "answer", "response", "completion"))
    if instruction_col is None or output_col is None:
        raise SystemExit(f"{path}: cannot find instruction/output columns in {frame.columns}")
    input_col = _first_existing(frame, ("input",))

    rows: list[dict[str, str]] = []
    for row in frame.iter_rows(named=True):
        instruction = _split_text(row.get(instruction_col)).strip()
        answer = _split_text(row.get(output_col)).strip()
        if not instruction or not answer:
            continue
        extra = _split_text(row.get(input_col)).strip() if input_col else ""
        rows.append({"instruction": instruction, "input": extra, "output": answer})

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"rows": len(rows), "bytes": out_file.stat().st_size}


def convert_preferences(path: Path, out_file: Path) -> dict[str, Any]:
    """hh-rlhf ``chosen``/``rejected`` dialogues -> prompt/chosen/rejected rows.

    Accepts both parquet and gzipped JSONL (``train.jsonl.gz``), which is what
    the upstream ``Anthropic/hh-rlhf`` repo ships.
    """
    rows: list[dict[str, str]] = []
    marker = "Assistant:"
    for row in _iter_records(path):
        chosen = _split_text(row.get("chosen"))
        rejected = _split_text(row.get("rejected"))
        cut = chosen.rfind(marker)
        if cut == -1:
            continue
        prompt = chosen[: cut + len(marker)].strip()
        good = chosen[cut + len(marker) :].strip()
        bad_cut = rejected.rfind(marker)
        bad = rejected[bad_cut + len(marker) :].strip() if bad_cut != -1 else ""
        if not (prompt and good and bad):
            continue
        rows.append({"prompt": prompt, "chosen": good, "rejected": bad})

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"rows": len(rows), "bytes": out_file.stat().st_size}


def main() -> int:
    parser = argparse.ArgumentParser(description="HF text parquet -> platform corpora")
    parser.add_argument("--root", default="/work", help="dataset root (host mount)")
    parser.add_argument(
        "--only",
        default="wikitext2,alpaca",
        help="comma separated subset of wikitext2,alpaca,gsm8k,hhrlhf",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    raw = root / "_raw" / "hf"
    wanted = [item.strip().lower() for item in args.only.split(",") if item.strip()]

    summaries: list[dict[str, Any]] = []
    missing: list[str] = []

    for key in wanted:
        if key not in DATASETS:
            log(f"unknown dataset key: {key}")
            continue
        prefix, kind, splits = DATASETS[key]
        for split in splits:
            source = _source_path(raw, key, split)
            if source is None:
                missing.append(f"{key}/{split}")
                continue
            if kind == "corpus":
                out_file = root / f"{prefix}-{split}" / f"{split}.txt"
            elif kind == "instructions":
                out_file = root / f"{prefix}-{split}" / f"{prefix}.jsonl"
            else:
                out_file = root / f"{prefix}-{split}" / "preferences.jsonl"

            if out_file.is_file() and out_file.stat().st_size > 0:
                log(f"{prefix}-{split}: already converted, skipping")
                summaries.append({"dataset": f"{prefix}-{split}", "skipped": True})
                continue

            started = time.time()
            if kind == "corpus":
                frame = pl.read_parquet(source)
                text_col = _first_existing(frame, ("text", "content", "document"))
                if text_col is None:
                    raise SystemExit(f"{source}: no text column in {frame.columns}")
                stats = convert_corpus(source, out_file, text_col)
            elif kind == "instructions":
                stats = convert_instructions(source, out_file)
            else:
                stats = convert_preferences(source, out_file)
            stats.update(
                {
                    "dataset": f"{prefix}-{split}",
                    "kind": kind,
                    "source": source.name,
                    "seconds": round(time.time() - started, 1),
                }
            )
            summaries.append(stats)
            log(f"{prefix}-{split}: {stats}")

    if not summaries:
        log("nothing to do; missing inputs: " + (", ".join(missing) or "none"))
        return 1

    meta_dir = root / "_meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    for summary in summaries:
        name = summary["dataset"]
        (meta_dir / f"{name}.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    if missing:
        log("skipped (not downloaded yet): " + ", ".join(missing))
    log(json.dumps(summaries, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
