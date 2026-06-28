"""Step 3: Run sliding-window inference and write raw segmentations.

Usage:
  python scripts/segment.py --config configs/default.yaml --weights /kaggle/working/train_results/net_weights_BEST.pt
"""
from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

from backend.settings import Config
from backend.inference import run_inference
from backend.utils import get_logger


def main():
    log = get_logger("segment_cli")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--weights", required=True, help="path to a .pt checkpoint")
    ap.add_argument("--tomo-ids", default=None, help="comma-separated list (default: all runs)")
    ap.add_argument("--write-scoremap", action="store_true",
                    help="Also write the full per-class probability scoremap (large)")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    if args.write_scoremap:
        cfg.inference.write_scoremap = True
    tomo_ids = args.tomo_ids.split(",") if args.tomo_ids else None

    run_inference(cfg, args.weights, tomo_ids=tomo_ids)
    log.info("Segmentation complete.")


if __name__ == "__main__":
    main()