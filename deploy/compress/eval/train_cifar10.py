#!/usr/bin/env python3
"""Train a real CIFAR-10 baseline for the compression evaluation.

Why this exists: the server's `models-local/TestCNN/model.pt` is a *randomly
initialised* smoke-test module (its own `arch.py` says so, and it scores exactly
0.10 on CIFAR10-test). Compressing a random model would make every accuracy
number meaningless, so this script trains the same `CifarCNN` architecture on the
server-local `CIFAR10-train` split and saves a real checkpoint.

    python train_cifar10.py --data /datasets --out /work/models/CifarCNN10 \
        --epochs 12 --batch-size 256 --width 32 --seed 20260922

Outputs (in --out):  arch.py (copied), model.pt (torch.save(model)), train_report.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CifarCNN on CIFAR10-train")
    parser.add_argument("--data", default="/datasets", help="数据集根目录（含 CIFAR10-train/CIFAR10-test）")
    parser.add_argument("--arch", default=None, help="arch.py 路径（默认 <out 的父目录>/TestCNN/arch.py）")
    parser.add_argument("--out", required=True, help="输出目录（写 model.pt / arch.py / train_report.json）")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_root = Path(args.data)
    arch_path = Path(args.arch) if args.arch else Path("/models/TestCNN/arch.py")
    sys.path.insert(0, str(arch_path.parent))
    import arch  # noqa: PLC0415 - paths are resolved at runtime

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(arch_path, out / "arch.py")

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    train_set = datasets.ImageFolder(str(data_root / "CIFAR10-train"), transform=train_tf)
    test_set = datasets.ImageFolder(str(data_root / "CIFAR10-test"), transform=test_tf)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=(device == "cuda"))
    test_loader = DataLoader(test_set, batch_size=512, shuffle=False, num_workers=args.workers)

    model = arch.CifarCNN(num_classes=len(train_set.classes), width=args.width).to(device)
    params = sum(p.numel() for p in model.parameters())
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    def evaluate() -> float:
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                correct += int((model(images).argmax(1) == labels).sum())
                total += labels.numel()
        return round(correct / total, 4)

    history: list[dict[str, float]] = []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.time()
        running = 0.0
        seen = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running += float(loss) * labels.numel()
            seen += labels.numel()
        scheduler.step()
        accuracy = evaluate()
        entry = {
            "epoch": epoch,
            "train_loss": round(running / max(seen, 1), 4),
            "test_accuracy": accuracy,
            "seconds": round(time.time() - epoch_start, 1),
        }
        history.append(entry)
        print(json.dumps(entry), flush=True)

    checkpoint = out / "model.pt"
    torch.save(model, checkpoint)
    report = {
        "architecture": "CifarCNN",
        "width": args.width,
        "classes": train_set.classes,
        "params": params,
        "device": device,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "train_samples": len(train_set),
        "test_samples": len(test_set),
        "test_accuracy": history[-1]["test_accuracy"],
        "checkpoint_bytes": checkpoint.stat().st_size,
        "duration_s": round(time.time() - started, 1),
        "history": history,
        "dataset_root": str(data_root),
        "arch_file": str(out / "arch.py"),
    }
    (out / "train_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "history"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
