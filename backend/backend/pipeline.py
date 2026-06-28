"""End-to-end pipeline: train -> segment -> (optional) localize."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from .settings import Config
from .utils import get_logger
from . import train as train_mod
from . import inference as infer_mod
from . import localize as loc_mod
from .data import copick_io


def run_pipeline(
    cfg: Config,
    weights_path: Optional[str] = None,
    do_train: bool = True,
    do_segment: bool = True,
    do_localize: bool = False,
    tomo_ids: Optional[List[str]] = None,
) -> str:
    """Run selected stages. Returns the path to the trained/best weights."""
    log = get_logger("pipeline")

    if do_train:
        log.info("=== STAGE 1: TRAIN ===")
        weights_path = train_mod.train(cfg)

    if do_segment:
        log.info("=== STAGE 2: SEGMENT ===")
        if weights_path is None:
            # default to BEST in out_dir
            cand = Path(cfg.train.out_dir) / "net_weights_BEST.pt"
            weights_path = str(cand) if cand.exists() else None
        if weights_path is None:
            raise FileNotFoundError("No weights path provided and no BEST checkpoint found.")
        infer_mod.run_inference(cfg, weights_path, tomo_ids=tomo_ids)

    if do_localize:
        log.info("=== STAGE 3: LOCALIZE (optional) ===")
        loc_mod.localize_segmentations(cfg, tomo_ids=tomo_ids)

    return weights_path or ""