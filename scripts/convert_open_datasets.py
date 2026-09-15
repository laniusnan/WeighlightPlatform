#!/usr/bin/env python3
"""Convert downloaded open datasets into the layout this platform expects.

The compression pipeline (data-driven pruning, QAT, distillation, ONNX
calibration, classification evaluation) consumes *flat per-class image
folders*::

    <dataset-root>/<class-name>/*.png

This script turns the canonical open-dataset archives below into exactly that,
one directory per split, so what lands on the server can be registered by
reference through ``POST /api/v1/datasets/import-local`` and used by any task
without further conversion::

    <root>/_raw/cifar-10-batches-py -> <root>/CIFAR10-train/<class>/*.png
                                       <root>/CIFAR10-test/<class>/*.png
    <root>/_raw/cifar-100-python    -> <root>/CIFAR100-train|test/<fine-class>/*.png
    <root>/_raw/mnist              -> <root>/MNIST-train|test/<digit>/*.png
    <root>/_raw/fashion            -> <root>/FashionMNIST-train|test/<name>/*.png

Only Pillow + numpy are needed, so it runs inside the platform worker image::

    docker run --rm -v /home/zhangxuefei/Dataset:/work -w /work \\
        backend:cpu python /work/_tools/convert_open_datasets.py --root /work

Conversion is idempotent: a split directory that already holds the expected
number of PNGs is skipped, so an interrupted run can simply be repeated.
"""
from __future__ import annotations

import argparse
import gzip
import json
import pickle
import struct
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
FASHION_CLASSES = [
    "t-shirt", "trouser", "pullover", "dress", "coat",
    "sandal", "shirt", "sneaker", "bag", "ankle-boot",
]


def log(message: str) -> None:
    print(f"[convert] {message}", flush=True)


def class_dir(root: Path, base: str, split: str, class_name: str) -> Path:
    """``<root>/<base>-<split>/<class>`` - the layout the jobs expect."""
    return root / f"{base}-{split}" / class_name


def save_png(array: "np.ndarray", path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path, format="PNG", optimize=False)


def split_done(root: Path, base: str, split: str, expected: int) -> bool:
    """True when ``<base>-<split>`` already holds ``expected`` PNGs."""
    folder = root / f"{base}-{split}"
    if not folder.is_dir():
        return False
    return sum(1 for _ in folder.rglob("*.png")) == expected


def convert_cifar(
    raw_dir: Path,
    root: Path,
    base: str,
    class_names: list[str],
    expected: dict[str, int],
) -> dict:
    """CIFAR python-pickle batches -> one flat per-class folder per split."""
    batches = {
        "train": [raw_dir / f"data_batch_{i}" for i in range(1, 6)],
        "test": [raw_dir / "test_batch"],
    }
    written_total = 0
    started = time.time()
    skipped: list[str] = []
    for split, paths in batches.items():
        if split_done(root, base, split, expected[split]):
            skipped.append(split)
            written_total += expected[split]
            log(f"{base}-{split}: already complete, skipping")
            continue
        index = 0
        for path in paths:
            if not path.is_file():
                raise SystemExit(f"missing CIFAR batch: {path}")
            with path.open("rb") as fh:
                batch = pickle.load(fh, encoding="bytes")
            data = batch[b"data"]
            labels = batch[b"labels"]
            for pixels, label in zip(data, labels):
                # CIFAR stores HWC-ordered bytes as (3, 32, 32) planes.
                arr = np.frombuffer(pixels, dtype=np.uint8).reshape(3, 32, 32)
                name = class_names[int(label) % len(class_names)]
                save_png(
                    arr.transpose(1, 2, 0),
                    class_dir(root, base, split, name) / f"{split}_{index:05d}.png",
                )
                index += 1
            log(f"{base}-{split}: {index} images written ({time.time() - started:.0f}s)")
        written_total += index

    return {
        "dataset": base,
        "images": written_total,
        "classes": len(class_names),
        "splits": {
            split: {
                "expected": expected[split],
                "on_disk": count_images(root, base, split),
            }
            for split in batches
        },
        "skipped_splits": skipped,
        "seconds": round(time.time() - started, 1),
    }


def read_idx_images(path: Path) -> "np.ndarray":
    with gzip.open(path, "rb") as fh:
        magic, count, rows, cols = struct.unpack(">IIII", fh.read(16))
        if magic != 2051:
            raise SystemExit(f"bad idx magic in {path}: {magic}")
        buf = fh.read(count * rows * cols)
    return np.frombuffer(buf, dtype=np.uint8).reshape(count, rows, cols)


def read_idx_labels(path: Path) -> "np.ndarray":
    with gzip.open(path, "rb") as fh:
        magic, count = struct.unpack(">II", fh.read(8))
        if magic != 2049:
            raise SystemExit(f"bad idx magic in {path}: {magic}")
        buf = fh.read(count)
    return np.frombuffer(buf, dtype=np.uint8)


def convert_idx(
    raw_dir: Path,
    root: Path,
    base: str,
    specs: dict[str, tuple[str, str]],
    expected: dict[str, int],
    names: dict[int, str],
) -> dict:
    """MNIST-style gzipped idx archives -> flat per-class (digit) folders."""
    written_total = 0
    started = time.time()
    skipped: list[str] = []
    for split, (images_name, labels_name) in specs.items():
        if split_done(root, base, split, expected[split]):
            skipped.append(split)
            written_total += expected[split]
            log(f"{base}-{split}: already complete, skipping")
            continue
        images = read_idx_images(raw_dir / images_name)
        labels = read_idx_labels(raw_dir / labels_name)
        for index, (image, label) in enumerate(zip(images, labels)):
            name = names.get(int(label), str(int(label)))
            save_png(image, class_dir(root, base, split, name) / f"{split}_{index:05d}.png")
        log(f"{base}-{split}: {len(labels)} images written ({time.time() - started:.0f}s)")
        written_total += len(labels)

    return {
        "dataset": base,
        "images": written_total,
        "classes": len(names),
        "splits": {
            split: {
                "expected": expected[split],
                "on_disk": count_images(root, base, split),
            }
            for split in specs
        },
        "skipped_splits": skipped,
        "seconds": round(time.time() - started, 1),
    }



def count_images(root: Path, base: str, split: str) -> int:
    folder = root / f"{base}-{split}"
    return sum(1 for _ in folder.rglob("*.png")) if folder.is_dir() else 0



def main() -> int:
    parser = argparse.ArgumentParser(description="convert open datasets to per-class folders")
    parser.add_argument("--root", default="/work", help="dataset root (host mount)")
    parser.add_argument(
        "--only",
        default="cifar10,cifar100,mnist,fashionmnist",
        help="comma separated subset of cifar10,cifar100,mnist,fashionmnist",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    raw = root / "_raw"
    wanted = {item.strip().lower() for item in args.only.split(",") if item.strip()}
    summaries: list[dict] = []

    if "cifar10" in wanted and (raw / "cifar-10-batches-py").is_dir():
        summaries.append(
            convert_cifar(
                raw / "cifar-10-batches-py",
                root,
                "CIFAR10",
                CIFAR10_CLASSES,
                {"train": 50000, "test": 10000},
            )
        )

    if "cifar100" in wanted and (raw / "cifar-100-python").is_dir():
        fine_names = CIFAR10_CLASSES
        meta_path = raw / "cifar-100-python" / "meta"
        if meta_path.is_file():
            with meta_path.open("rb") as fh:
                meta = pickle.load(fh, encoding="bytes")
            fine_names = [
                name.decode() if isinstance(name, bytes) else str(name)
                for name in meta[b"fine_label_names"]
            ]
        summaries.append(
            convert_cifar(
                raw / "cifar-100-python",
                root,
                "CIFAR100",
                fine_names,
                {"train": 50000, "test": 10000},
            )
        )

    idx_specs = {
        "train": ("train-images-idx3-ubyte", "train-labels-idx1-ubyte"),
        "test": ("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"),
    }
    idx_expected = {"train": 60000, "test": 10000}

    if "mnist" in wanted and (raw / "mnist").is_dir():
        summaries.append(
            convert_idx(
                raw / "mnist",
                root,
                "MNIST",
                idx_specs,
                idx_expected,
                {i: str(i) for i in range(10)},
            )
        )

    if "fashionmnist" in wanted and (raw / "fashion").is_dir():
        summaries.append(
            convert_idx(
                raw / "fashion",
                root,
                "FashionMNIST",
                idx_specs,
                idx_expected,
                {i: name for i, name in enumerate(FASHION_CLASSES)},
            )
        )

    if not summaries:
        log("nothing to do: raw archives missing under _raw/ or excluded by --only")
        return 1

    meta_dir = root / "_meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    for summary in summaries:
        (meta_dir / f"{summary['dataset']}.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    log(json.dumps(summaries, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
