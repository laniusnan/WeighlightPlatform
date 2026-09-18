"""``finetune`` subcommand: LoRA SFT and DPO preference tuning.

================  ==============================  ==================================================
--method          platform params                 notes
================  ==============================  ==================================================
lora              llm_finetune, stage=sft         HF models need CUDA; CPU runs the micro prototype
qlora             llm_finetune, stage=sft         alias of ``lora`` (INT4 base is a GPU-phase item)
dpo               llm_finetune, stage=dpo         prompt/chosen/rejected pairs, L = -log σ(β·Δ)
================  ==============================  ==================================================

Data sources (first match wins): ``--data`` inline text, ``--dataset`` platform
dataset (jsonl / csv / xlsx / txt), ``--model-repo`` HuggingFace dataset id.
"""
from __future__ import annotations

import argparse
from typing import Any

from . import common

METHODS: dict[str, str] = {
    "lora": "LoRA SFT (instruction pairs -> adapter + merged weights)",
    "qlora": "alias of lora (QLoRA/INT4 base belongs to the GPU phase)",
    "dpo": "DPO preference tuning (prompt/chosen/rejected)",
}

PARAMS: list[tuple[str, type, str]] = [
    ("dataset", str, "platform dataset id/name (jsonl/csv/xlsx/txt)"),
    ("data", str, "inline training text ('question => answer' per line)"),
    ("model-repo", str, "HuggingFace dataset id to pull pairs from"),
    ("epochs", int, "epochs (SFT default 3, DPO default 1)"),
    ("lr", float, "learning rate (SFT 3e-3, DPO 5e-5)"),
    ("lora-rank", int, "LoRA rank (default 8)"),
    ("hf-max-len", int, "HF tokenizer max length (default 128)"),
    ("dpo-beta", float, "DPO beta (default 0.1)"),
    ("context", int, "micro-LM context window for external corpora (default 128)"),
    ("max-new", int, "max new tokens during the post-training sample (default 32)"),
]


def register(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "finetune",
        help="fine-tune a model (LoRA SFT / DPO)",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="model id / name / name-prefix / UUID")
    parser.add_argument("--method", required=True, choices=sorted(METHODS), help="tuning method")
    for flag, cast, helptext in PARAMS:
        parser.add_argument(f"--{flag}", dest=flag.replace("-", "_"), type=cast, default=None, help=helptext)
    common.add_global_flags(parser)
    parser.set_defaults(handler=run)


def build_params(args: argparse.Namespace) -> dict[str, Any]:
    client = common.ApiClient()
    stage = "dpo" if args.method == "dpo" else "sft"
    defaults = {"epochs": 1 if stage == "dpo" else 3, "lr": 5e-5 if stage == "dpo" else 3e-3}

    params: dict[str, Any] = {
        "stage": stage,
        "epochs": common.resolve_int("epochs", args.epochs, defaults["epochs"]),
        "lr": common.resolve_float("lr", args.lr, defaults["lr"]),
        "lora_rank": common.resolve_int("lora-rank", args.lora_rank, 8),
    }
    if args.hf_max_len is not None:
        params["hf_max_len"] = common.resolve_int("hf-max-len", args.hf_max_len, 128)
    if args.context is not None:
        params["context"] = common.resolve_int("context", args.context, 128)
    if args.max_new is not None:
        params["max_new"] = common.resolve_int("max-new", args.max_new, 32)
    if stage == "dpo":
        params["dpo_beta"] = common.resolve_float("dpo-beta", args.dpo_beta, 0.1)
    if args.model_repo:
        params["model_repo"] = args.model_repo

    sources = [bool(args.dataset), bool(args.data), bool(args.model_repo)]
    if sum(sources) == 0:
        raise common.fail(
            "finetune needs training data: pass --dataset, --data or --model-repo", common.EXIT_USAGE
        )
    if args.dataset:
        params["dataset_id"] = common.resolve_dataset(client, args.dataset)["id"]
    if args.data:
        params["data"] = args.data
    return common.merged_params(args, params)


def run(args: argparse.Namespace) -> dict[str, Any]:
    return common.run_submit_flow(args, args.model, "llm_finetune", build_params(args))
