"""Assess tomogram grid viability (Hopkins-based QA gate).

Usage:
  python -m downstream.scripts.assess_grid_viability \\
    --csv runs/exposure_labels_2000rays.csv \\
    --out runs/grid_viability.json

  python -m downstream.scripts.assess_grid_viability \\
    --source overlay --input-root /path/to/czii-dataset \\
    --calibrate --out runs/grid_viability.json
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

from downstream.io.coordinates import load_from_csv, load_from_overlay
from downstream.qa.grid_viability import (
    GridViabilityConfig,
    assess_clouds,
    exposure_means_from_csv,
    fit_reference_stats,
    write_report,
)


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Grid Viability Index (Hopkins QA)")
    ap.add_argument(
        "--source",
        choices=["csv", "overlay"],
        default="csv",
        help="Coordinate input format",
    )
    ap.add_argument("--csv", default=None, help="Picks CSV (when --source csv)")
    ap.add_argument(
        "--input-root",
        default=None,
        help="CZII dataset root (when --source overlay)",
    )
    ap.add_argument("--out", default="runs/grid_viability.json")
    ap.add_argument("--tomo-id", action="append", default=None)
    ap.add_argument(
        "--calibrate",
        action="store_true",
        help="Use loaded tomograms as clean reference for thresholds",
    )
    ap.add_argument("--min-particles", type=int, default=30)
    args = ap.parse_args(argv)

    if args.source == "csv":
        if not args.csv:
            ap.error("--csv is required when --source csv")
        clouds = load_from_csv(args.csv, tomo_ids=args.tomo_id)
        exposure = exposure_means_from_csv(args.csv)
    else:
        if not args.input_root:
            ap.error("--input-root is required when --source overlay")
        clouds = load_from_overlay(args.input_root, tomo_ids=args.tomo_id)
        exposure = {}

    config = GridViabilityConfig(min_particles=args.min_particles)
    reference = fit_reference_stats(clouds) if args.calibrate else None
    results = assess_clouds(clouds, config=config, reference=reference, exposure_by_tomo=exposure)

    write_report(results, args.out, reference=reference)

    for r in results:
        print(
            f"{r.tomo_id}: {r.status:14s}  H={r.hopkins_h:.3f}  "
            f"n={r.n_particles:4d}  {r.message}"
        )
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
