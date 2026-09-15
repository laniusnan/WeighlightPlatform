#!/usr/bin/env python3
"""Convert HuggingFace *parquet* image-classification datasets to per-class
image folders (the layout this platform's jobs consume).

Why parquet: the original CIFAR/MNIST archives live on overseas hosts that are
unreachable at usable speed from some networks, while HuggingFace mirrors are
fast. The platform image already ships polars (native parquet reader) + Pillow,
so no extra dependency is needed::

    <root>/_raw/hf/cifar10/train.parquet
        -> <root>/CIFAR10-train/<class>/*.png
    <root>/_raw/hf/cifar10/test.parquet
        -> <root>/CIFAR10-test/<class>/*.png

Supported datasets (auto-detected ``img``/``image`` and ``label``/``fine_label``
columns): cifar10, cifar100, mnist, fashionmnist.

Run inside the platform image (the host only needs Docker)::

    docker run --rm -v /home/zhangxuefei/Dataset:/work -w /work \\
        backend:cpu python /work/_tools/convert_hf_parquet.py --root /work

Conversion is idempotent: a split whose folder already holds the expected
number of PNGs is skipped.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import polars as pl
from PIL import Image

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
FASHION_CLASSES = [
    "t-shirt", "trouser", "pullover", "dress", "coat",
    "sandal", "shirt", "sneaker", "bag", "ankle-boot",
]
CIFAR100_CLASSES = [
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle",
    "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel",
    "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock",
    "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster",
    "house", "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion",
    "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain", "mouse",
    "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear",
    "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine",
    "possum", "rabbit", "raccoon", "ray", "road", "rocket", "rose", "sea",
    "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake", "spider",
    "squirrel", "streetcar", "sunflower", "sweet_pepper", "table", "tank",
    "telephone", "television", "tiger", "tractor", "train", "trout", "tulip",
    "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm",
]

# dataset key -> (parquet dir under _raw/hf, output prefix, class names, expected)
DATASETS: dict[str, tuple[str, str, list[str], dict[str, int]]] = {
    "cifar10": ("cifar10", "CIFAR10", CIFAR10_CLASSES, {"train": 50000, "test": 10000}),
    "cifar100": ("cifar100", "CIFAR100", CIFAR100_CLASSES, {"train": 50000, "test": 10000}),
    "mnist": ("mnist", "MNIST", [str(i) for i in range(10)], {"train": 60000, "test": 10000}),
    "fashionmnist": ("fashionmnist", "FashionMNIST", FASHION_CLASSES, {"train": 60000, "test": 10000}),
}


def log(message: str) -> None:
    print(f"[parquet] {message}", flush=True)


def split_done(root: Path, base: str, split: str, expected: int) -> bool:
    folder = root / f"{base}-{split}"
    if not folder.is_dir():
        return False
    return sum(1 for _ in folder.rglob("*.png")) == expected


def count_images(root: Path, base: str, split: str) -> int:
    folder = root / f"{base}-{split}"
    return sum(1 for _ in folder.rglob("*.png")) if folder.is_dir() else 0


def pick_columns(frame: pl.DataFrame) -> tuple[str, str]:
    """Return ``(image_column, label_column)`` for a HF image-classification file."""
    columns = frame.columns
    image_col = next(
        (c for c in ("img", "image") if c in columns),
        next((c for c in columns if "img" in c.lower() or "image" in c.lower()), None),
    )
    label_col = next(
        (c for c in ("label", "fine_label") if c in columns),
        next((c for c in columns if "label" in c.lower()), None),
    )
    if image_col is None or label_col is None:
        raise SystemExit(f"cannot find image/label columns in {columns}")
    return image_col, label_col


def convert_split(
    parquet_path: Path,
    root: Path,
    base: str,
    split: str,
    class_names: list[str],
) -> int:
    """Write every row of ``parquet_path`` as a PNG under ``<base>-<split>``."""
    frame = pl.read_parquet(parquet_path)
    image_col, label_col = pick_columns(frame)
    blobs = frame.select(pl.col(image_col).struct.field("bytes")).to_series().to_list()
    labels = frame.select(pl.col(label_col)).to_series().to_list()

    written = 0
    started = time.time()
    for index, (blob, label) in enumerate(zip(blobs, labels)):
        if blob is None:
            continue
        name = class_names[int(label) % len(class_names)]
        target = root / f"{base}-{split}" / name / f"{split}_{index:05d}.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        from io import BytesIO

        image = Image.open(BytesIO(blob))
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(target, format="PNG", optimize=False)
        written += 1
        if written % 10000 == 0:
            log(f"{base}-{split}: {written} images ({time.time() - started:.0f}s)")
    log(f"{base}-{split}: {written} images ({time.time() - started:.0f}s)")
    return written



def main() -> int:
    parser = argparse.ArgumentParser(description="HF parquet -> per-class image folders")
    parser.add_argument("--root", default="/work", help="dataset root (host mount)")
    parser.add_argument(
        "--only",
        default="cifar10,cifar100,mnist,fashionmnist",
        help="comma separated subset of cifar10,cifar100,mnist,fashionmnist",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    raw = root / "_raw" / "hf"
    wanted = [item.strip().lower() for item in args.only.split(",") if item.strip()]

    summaries: list[dict] = []
    missing: list[str] = []
    for key in wanted:
        if key not in DATASETS:
            log(f"unknown dataset key: {key}")
            continue
        folder, base, class_names, expected = DATASETS[key]
        files = {split: raw / folder / f"{split}.parquet" for split in expected}
        if not all(path.is_file() for path in files.values()):
            missing.append(f"{key} ({folder}/train|test.parquet)")
            continue
        written = 0
        skipped: list[str] = []
        for split, path in files.items():
            if split_done(root, base, split, expected[split]):
                skipped.append(split)
                written += expected[split]
                log(f"{base}-{split}: already complete, skipping")
                continue
            written += convert_split(path, root, base, split, class_names)
        summaries.append(
            {
                "dataset": base,
                "images": written,
                "classes": len(class_names),
                "splits": {
                    split: {
                        "expected": expected[split],
                        "on_disk": count_images(root, base, split),
                    }
                    for split in expected
                },
                "skipped_splits": skipped,
            }
        )

    if not summaries:
        log("nothing to do; missing parquet files: " + ("; ".join(missing) or "none"))
        return 1

    meta_dir = root / "_meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    for summary in summaries:
        (meta_dir / f"{summary['dataset']}.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    if missing:
        log("skipped (parquet not downloaded yet): " + "; ".join(missing))
    log(json.dumps(summaries, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
