"""One-shot pipeline: setup -> build targets -> train -> segment -> (localize).

Usage on Kaggle:
  python scripts/run_all.py --config configs/default.yaml \
      --input-root /kaggle/input/competitions/czii-cryo-et-object-identification \
      --output-root /kaggle/working

Stages can be toggled with flags. By default it runs setup, build_targets,
train, and segment (raw segmentations are the deliverable). Localization is
opt-in via --localize.
"""
from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

from backend.settings import Config
from backend.pipeline import run_pipeline
from backend.data import copick_io, target_build
from backend.utils import get_logger
from scripts.setup_copick import find_static_root, copy_overlay
from pathlib import Path
import os


def main():
    log = get_logger("run_all")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--input-root", default="/kaggle/input/competitions/czii-cryo-et-object-identification")
    ap.add_argument("--output-root", default="/kaggle/working")
    ap.add_argument("--no-setup", action="store_true")
    ap.add_argument("--no-targets", action="store_true")
    ap.add_argument("--no-train", action="store_true")
    ap.add_argument("--no-segment", action="store_true")
    ap.add_argument("--localize", action="store_true")
    ap.add_argument("--weights", default=None, help="Skip training and use this .pt for segment")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)

    # --- Stage 0: setup copick config + overlay ---
    if not args.no_setup:
        log.info("=== STAGE 0: SETUP COPICK ===")
        static_root = find_static_root(args.input_root)
        overlay_out = os.path.join(args.output_root, "overlay")
        train_overlay = Path(args.input_root) / "train" / "overlay"
        if not train_overlay.exists():
            train_overlay = Path(args.input_root) / "overlay"
        if train_overlay.exists():
            copy_overlay(str(train_overlay), overlay_out)
        copick_io.write_copick_config(cfg.data.copick_config, static_root, overlay_out)
        log.info(f"  static={static_root} overlay={overlay_out} config={cfg.data.copick_config}")

    # --- Stage 1: build targets ---
    if not args.no_targets:
        log.info("=== STAGE 1: BUILD TARGETS ===")
        target_build.build_targets(
            cfg.data.copick_config, None, cfg.data.voxel_size,
            cfg.data.target_name, cfg.data.target_user_id, cfg.data.target_session_id,
        )

    # --- Stages 2/3/4: pipeline ---
    run_pipeline(
        cfg,
        weights_path=args.weights,
        do_train=not args.no_train,
        do_segment=not args.no_segment,
        do_localize=args.localize,
    )


if __name__ == "__main__":
    main()