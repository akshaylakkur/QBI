"""Set up the copick config + overlay for the czii challenge on Kaggle.

Usage:
  python scripts/setup_copick.py \
      --input-root /kaggle/input/competitions/czii-cryo-et-object-identification \
      --output-root /kaggle/working \
      --config-out /kaggle/working/copick.config
"""
from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import os
import shutil
from pathlib import Path

from backend.data import copick_io
from backend.utils import get_logger


def find_static_root(input_root: str) -> str:
    """Locate the 'static' dir within the competition input (train subset)."""
    root = Path(input_root)
    # common layout: input_root/train/static
    candidates = [
        root / "train" / "static",
        root / "static",
        root,
    ]
    for c in candidates:
        if (c / "ExperimentRuns").exists():
            return str(c)
    # fallback: search recursively for ExperimentRuns
    for p in root.rglob("ExperimentRuns"):
        return str(p.parent)
    raise FileNotFoundError(f"Could not locate 'static/ExperimentRuns' under {input_root}")


def copy_overlay(source_overlay: str, dest_overlay: str) -> None:
    """Copy the train overlay (ground-truth picks) into the writable overlay,
    renaming files to be discoverable by copick (prefix curation_0_)."""
    src = Path(source_overlay)
    dst = Path(dest_overlay)
    if not src.exists():
        return
    for root, _dirs, files in os.walk(src):
        rel = os.path.relpath(root, src)
        target_dir = dst / rel
        target_dir.mkdir(parents=True, exist_ok=True)
        for f in files:
            new_name = f if f.startswith("curation_0_") else f"curation_0_{f}"
            shutil.copy2(Path(root) / f, target_dir / new_name)


def main():
    log = get_logger("setup")
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", default="/kaggle/input/competitions/czii-cryo-et-object-identification")
    ap.add_argument("--output-root", default="/kaggle/working")
    ap.add_argument("--config-out", default=None)
    ap.add_argument("--overlay-out", default=None)
    args = ap.parse_args()

    config_out = args.config_out or os.path.join(args.output_root, "copick.config")
    overlay_out = args.overlay_out or os.path.join(args.output_root, "overlay")

    static_root = find_static_root(args.input_root)
    log.info(f"Static root: {static_root}")
    log.info(f"Overlay out: {overlay_out}")

    # Copy train overlay (picks) into the writable overlay
    train_overlay = Path(args.input_root) / "train" / "overlay"
    if not train_overlay.exists():
        train_overlay = Path(args.input_root) / "overlay"
    if train_overlay.exists():
        copy_overlay(str(train_overlay), overlay_out)
        log.info(f"Copied overlay picks from {train_overlay} -> {overlay_out}")

    copick_io.write_copick_config(config_out, static_root, overlay_out)
    log.info(f"Wrote copick config -> {config_out}")


if __name__ == "__main__":
    main()