"""Step 1: Build sphere segmentation targets from copick picks.

Usage:
  python scripts/build_targets.py --config configs/default.yaml
"""
from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

from backend.settings import Config
from backend.data import target_build
from backend.data import copick_io
from backend.utils import get_logger


def main():
    log = get_logger("build_targets")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--voxel-size", type=float, default=None)
    ap.add_argument("--tomo-ids", default=None, help="comma-separated list (default: all)")
    ap.add_argument("--out-name", default=None)
    ap.add_argument("--out-user", default=None)
    ap.add_argument("--out-session", default=None)
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    vs = args.voxel_size or cfg.data.voxel_size
    out_name = args.out_name or cfg.data.target_name
    out_user = args.out_user or cfg.data.target_user_id
    out_session = args.out_session or cfg.data.target_session_id
    tomo_ids = args.tomo_ids.split(",") if args.tomo_ids else None

    log.info(f"Building targets: name={out_name} user={out_user} session={out_session} vs={vs}")
    target_build.build_targets(
        cfg.data.copick_config, tomo_ids, vs, out_name, out_user, out_session,
    )
    log.info("Done.")


if __name__ == "__main__":
    main()