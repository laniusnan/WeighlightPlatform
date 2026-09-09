#!/usr/bin/env python
"""FitNets one-click smoke test (no upload / DB / worker).

Builds a tiny CNN teacher + auto micro dataset, then runs the real
``app.jobs.compression.distill`` with ``method="features"`` end to end.

Usage:
    cd backend
    PYTHONPATH=. python scripts/smoke_fitnets.py
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
for _k, _v in {
    "PROJECT_NAME": "smoke",
    "SECRET_KEY": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    "FIRST_SUPERUSER": "a@b.c",
    "FIRST_SUPERUSER_PASSWORD": "pw123",
    "DATABASE_URL": "postgresql://postgres:pw@localhost:5432/app",
    "REDIS_URL": "redis://localhost:6379/0",
}.items():
    os.environ.setdefault(_k, _v)

ARCH_SRC = '''import torch.nn as nn


class Net(nn.Module):
    def __init__(self, num_classes=2, width_mult=1.0):
        super().__init__()
        w = int(8 * width_mult)
        self.features = nn.Sequential(
            nn.Conv2d(3, w, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(w, w * 2, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(w * 2, num_classes)

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))
'''


# Identical class lives in __main__ so torch.save(model) pickling resolves.
class Net(nn.Module):
    def __init__(self, num_classes=2, width_mult=1.0):
        super().__init__()
        w = int(8 * width_mult)
        self.features = nn.Sequential(
            nn.Conv2d(3, w, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(w, w * 2, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(w * 2, num_classes)

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


def main() -> int:
    from app.jobs.compression import distill

    root = Path(tempfile.mkdtemp(prefix="fitnets_smoke_"))
    try:
        (root / "arch.py").write_text(ARCH_SRC, encoding="utf-8")
        teacher = Net(num_classes=2, width_mult=1.0).eval()
        torch.save(teacher, root / "teacher.pt")

        data = root / "data"
        rng = np.random.default_rng(7)
        for label in ("cat", "dog"):
            (data / label).mkdir(parents=True)
            for i in range(6):
                arr = rng.integers(0, 256, size=(24, 24, 3), dtype=np.uint8)
                Image.fromarray(arr).save(data / label / f"{i}.png")

        out = root / "out"
        model = SimpleNamespace(file_path=str(root / "teacher.pt"))
        params = {
            "method": "features",
            "width_mult": 0.5,
            "alpha": 0.7,
            "epochs": 1,
            "lr": 1e-3,
            "batch_size": 4,
            "image_size": 24,
            "input_shape": "1,3,24,24",
            "data_dir": str(data),
        }
        _, result = distill(model, params, out, lambda p, m=None: print(f"  [{int(p)}%] {m or ''}"))
        print("SMOKE_OK", {k: result.get(k) for k in ("method", "width_mult", "epochs_completed", "last_distill_loss", "onnx_exported")})
        assert result.get("method") == "features"
        assert (out / "model_student.pt").is_file()
        return 0
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
