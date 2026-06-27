"""Compute steric exposure labels for particle coordinate clouds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from downstream.geometry.exposure import steric_exposure_batch
from downstream.io.coordinates import load_from_csv, load_from_overlay
from downstream.types import ExposureConfig, ParticleCloud


def _filter_cloud(
    cloud: ParticleCloud,
    target_type: Optional[str],
) -> tuple[ParticleCloud, np.ndarray]:
    """Return cloud subset and original indices if filtering by type."""
    if target_type is None:
        return cloud, np.arange(cloud.n_particles, dtype=np.int64)

    mask = np.array([t == target_type for t in cloud.types], dtype=bool)
    if not mask.any():
        return cloud, np.array([], dtype=np.int64)

    idxs = np.where(mask)[0]
    conf = cloud.confidence[idxs] if cloud.confidence is not None else None
    filtered = ParticleCloud(
        tomo_id=cloud.tomo_id,
        types=[cloud.types[i] for i in idxs],
        coords=cloud.coords[idxs],
        radii=cloud.radii[idxs],
        confidence=conf,
    )
    return filtered, idxs


def clouds_to_dataframe(
    clouds: List[ParticleCloud],
    config: ExposureConfig,
    target_type: Optional[str] = None,
) -> pd.DataFrame:
    rows = []
    for cloud in clouds:
        _, original_idxs = _filter_cloud(cloud, target_type)
        if original_idxs.size == 0:
            continue

        results = steric_exposure_batch(cloud, config=config, target_indices=original_idxs)
        for res in results:
            i = res.index
            row = {
                "tomo_id": cloud.tomo_id,
                "particle_type": cloud.types[i],
                "x": cloud.coords[i, 0],
                "y": cloud.coords[i, 1],
                "z": cloud.coords[i, 2],
                "radius_angstrom": cloud.radii[i],
                "steric_exposure": res.steric_exposure,
                "n_rays": res.n_rays,
                "n_neighbors_considered": res.n_neighbors_considered,
            }
            if cloud.confidence is not None:
                row["confidence"] = cloud.confidence[i]
            rows.append(row)

    return pd.DataFrame(rows)


def build_summary(df: pd.DataFrame) -> dict:
    summary: dict = {"tomograms": {}}
    if df.empty:
        return summary

    for tomo_id, tomo_df in df.groupby("tomo_id"):
        class_stats = {}
        for ptype, class_df in tomo_df.groupby("particle_type"):
            exp = class_df["steric_exposure"]
            class_stats[ptype] = {
                "count": int(len(class_df)),
                "mean_exposure": float(exp.mean()),
                "median_exposure": float(exp.median()),
                "min_exposure": float(exp.min()),
                "max_exposure": float(exp.max()),
            }
        summary["tomograms"][tomo_id] = {
            "particle_count": int(len(tomo_df)),
            "mean_exposure": float(tomo_df["steric_exposure"].mean()),
            "by_class": class_stats,
        }

    summary["global"] = {
        "particle_count": int(len(df)),
        "mean_exposure": float(df["steric_exposure"].mean()),
        "by_class": {
            ptype: {
                "count": int(len(g)),
                "mean_exposure": float(g["steric_exposure"].mean()),
            }
            for ptype, g in df.groupby("particle_type")
        },
    }
    return summary


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Compute steric exposure labels")
    ap.add_argument(
        "--input-root",
        default="/Users/kjaladi/Downloads/czii-cryo-et-object-identification",
        help="CZII dataset root (for --source overlay)",
    )
    ap.add_argument(
        "--source",
        choices=["overlay", "csv"],
        default="overlay",
        help="Coordinate input format",
    )
    ap.add_argument("--csv-path", default=None, help="CSV path when --source csv")
    ap.add_argument(
        "--out",
        default="/Users/kjaladi/Desktop/QBI/runs/exposure_labels.csv",
        help="Output CSV path",
    )
    ap.add_argument(
        "--summary-out",
        default=None,
        help="Output summary JSON (default: <out>.summary.json)",
    )
    ap.add_argument("--tomo-id", action="append", default=None, help="Filter tomogram(s)")
    ap.add_argument("--target-type", default=None, help="Filter particle type for exposure")
    ap.add_argument("--n-rays", type=int, default=1000)
    ap.add_argument("--fast", action="store_true", help="Use 100 rays (smoke test)")
    ap.add_argument("--probe-radius", type=float, default=5.0)
    ap.add_argument("--surface-epsilon", type=float, default=2.0)
    args = ap.parse_args(argv)

    n_rays = 100 if args.fast else args.n_rays
    config = ExposureConfig(
        n_rays=n_rays,
        probe_radius=args.probe_radius,
        surface_epsilon=args.surface_epsilon,
    )

    if args.source == "overlay":
        clouds = load_from_overlay(args.input_root, tomo_ids=args.tomo_id)
    else:
        if not args.csv_path:
            ap.error("--csv-path is required when --source csv")
        clouds = load_from_csv(args.csv_path, tomo_ids=args.tomo_id)

    df = clouds_to_dataframe(clouds, config=config, target_type=args.target_type)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    summary_path = Path(args.summary_out or out_path.with_suffix(".summary.json"))
    summary = build_summary(df)
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"Wrote {len(df)} rows -> {out_path}")
    print(f"Summary -> {summary_path}")
    if not df.empty:
        print("\nPer-class mean steric exposure:")
        for ptype, mean_val in df.groupby("particle_type")["steric_exposure"].mean().items():
            print(f"  {ptype}: {mean_val:.3f}")


if __name__ == "__main__":
    main()
