"""Step 2: Train the 3D segmentation model.

Usage:
  python scripts/train_model.py --config configs/default.yaml
  python scripts/train_model.py --config configs/default.yaml --set train.epochs 50
"""
from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

from backend.settings import Config
from backend.train import train
from backend.utils import get_logger


def main():
    log = get_logger("train_cli")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", action="append", default=[],
                    help="Override a dotted config key, e.g. --set train.epochs 50")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    for s in args.set:
        if " " in s:
            k, v = s.split(" ", 1)
            cfg.set(k, v)
            log.info(f"Override: {k} = {v}")

    best = train(cfg)
    log.info(f"Best checkpoint: {best}")


if __name__ == "__main__":
    main()