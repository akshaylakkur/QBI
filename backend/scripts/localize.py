"""Step 4 (optional): Localize particles from segmentations and write picks.

Usage:
  python scripts/localize.py --config configs/default.yaml
"""
from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

from backend.settings import Config
from backend.localize import localize_segmentations
from backend.utils import get_logger


def main():
    log = get_logger("localize_cli")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--tomo-ids", default=None)
    ap.add_argument("--segmentation-name", default=None)
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    tomo_ids = args.tomo_ids.split(",") if args.tomo_ids else None

    localize_segmentations(cfg, tomo_ids=tomo_ids, segmentation_name=args.segmentation_name)
    log.info("Localization complete.")


if __name__ == "__main__":
    main()