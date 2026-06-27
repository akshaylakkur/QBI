"""Load particle coordinate clouds from CZII overlay JSON or CSV contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from ..constants import radius_for_type
from ..types import ParticleCloud


def load_from_overlay(
    input_root: str | Path,
    tomo_ids: Optional[List[str]] = None,
) -> List[ParticleCloud]:
    """Load GT picks from ``{input_root}/train/overlay/ExperimentRuns/``."""
    root = Path(input_root)
    overlay_root = root / "train" / "overlay" / "ExperimentRuns"
    if not overlay_root.exists():
        raise FileNotFoundError(f"Overlay not found: {overlay_root}")

    if tomo_ids is None:
        tomo_dirs = sorted(p for p in overlay_root.iterdir() if p.is_dir())
    else:
        tomo_dirs = [overlay_root / tid for tid in tomo_ids]

    clouds: List[ParticleCloud] = []
    for tomo_dir in tomo_dirs:
        if not tomo_dir.is_dir():
            raise FileNotFoundError(f"Tomogram overlay not found: {tomo_dir}")
        picks_dir = tomo_dir / "Picks"
        if not picks_dir.exists():
            continue

        types: list[str] = []
        coords: list[tuple[float, float, float]] = []

        for json_path in sorted(picks_dir.glob("*.json")):
            particle_type = json_path.stem
            radius_for_type(particle_type)  # validate early
            data = json.loads(json_path.read_text())
            for point in data.get("points", []):
                loc = point["location"]
                types.append(particle_type)
                coords.append((float(loc["x"]), float(loc["y"]), float(loc["z"])))

        if not coords:
            continue

        clouds.append(
            ParticleCloud.from_arrays(
                tomo_id=tomo_dir.name,
                types=types,
                coords=np.asarray(coords, dtype=np.float64),
            )
        )

    if not clouds:
        raise ValueError(f"No particle picks found under {overlay_root}")

    return clouds


def load_from_csv(
    csv_path: str | Path,
    tomo_ids: Optional[List[str]] = None,
) -> List[ParticleCloud]:
    """Load coordinates from the team CSV contract.

    Expected columns: ``tomo_id``, ``particle_type``, ``x``, ``y``, ``z``
    Optional: ``confidence``
    """
    path = Path(csv_path)
    df = pd.read_csv(path)

    required = {"tomo_id", "particle_type", "x", "y", "z"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")

    if tomo_ids is not None:
        df = df[df["tomo_id"].isin(tomo_ids)]

    has_conf = "confidence" in df.columns
    clouds: List[ParticleCloud] = []

    for tomo_id, group in df.groupby("tomo_id", sort=True):
        types = group["particle_type"].astype(str).tolist()
        coords = group[["x", "y", "z"]].to_numpy(dtype=np.float64)
        confidence = (
            group["confidence"].to_numpy(dtype=np.float64) if has_conf else None
        )
        clouds.append(
            ParticleCloud.from_arrays(
                tomo_id=str(tomo_id),
                types=types,
                coords=coords,
                confidence=confidence,
            )
        )

    if not clouds:
        raise ValueError(f"No rows loaded from {path}")

    return clouds
