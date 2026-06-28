"""Grid Viability Index (GVI) — tomogram-level QA from spatial point patterns."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Literal, Optional, Sequence

import numpy as np
from scipy.spatial import cKDTree

from ..geometry.hopkins import hopkins_statistic, hopkins_tiled
from ..types import ParticleCloud

ViabilityStatus = Literal["pass", "warn", "fail_static", "fail_blob", "insufficient"]


@dataclass
class ReferenceStats:
    """Calibration distribution from clean reference tomograms."""

    hopkins_mean: float
    hopkins_std: float
    hopkins_p75: float
    hopkins_p95: float
    n_particles_mean: float
    n_particles_p95: float
    mean_knn_mean: float
    mean_knn_p05: float

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ReferenceStats":
        return cls(**d)


@dataclass
class GridViabilityConfig:
    """Thresholds for classifying tomogram coordinate quality."""

    min_particles: int = 30
    static_h_center: float = 0.5
    static_h_tolerance: float = 0.03
    static_n_multiplier: float = 2.0
    blob_h_threshold: float = 0.75
    tile_h_warn: float = 0.85
    use_tiles: bool = True
    tile_grid: tuple[int, int, int] = (2, 2, 2)


@dataclass
class GridViabilityResult:
    """QA report for one tomogram."""

    tomo_id: str
    status: ViabilityStatus
    hopkins_h: float
    n_particles: int
    mean_knn_dist: float
    min_knn_dist: float
    tile_h_max: Optional[float] = None
    mean_exposure: Optional[float] = None
    flags: List[str] = field(default_factory=list)
    message: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _knn_distance_stats(coords: np.ndarray) -> tuple[float, float]:
    """Return (mean, min) nearest-neighbor distances among particles."""
    n = len(coords)
    if n < 2:
        return float("nan"), float("nan")
    tree = cKDTree(coords)
    dists, _ = tree.query(coords, k=2)
    if dists.ndim == 1:
        nn = np.array([float(dists)])
    else:
        nn = dists[:, 1]
    return float(np.mean(nn)), float(np.min(nn))


def fit_reference_stats(clouds: Sequence[ParticleCloud]) -> ReferenceStats:
    """Fit reference percentiles from clean training tomograms."""
    hs, ns, mean_knns = [], [], []
    for cloud in clouds:
        if cloud.n_particles < 2:
            continue
        hs.append(hopkins_statistic(cloud.coords))
        ns.append(float(cloud.n_particles))
        mk, _ = _knn_distance_stats(cloud.coords)
        mean_knns.append(mk)

    h = np.array(hs, dtype=np.float64)
    n = np.array(ns, dtype=np.float64)
    mk = np.array(mean_knns, dtype=np.float64)
    return ReferenceStats(
        hopkins_mean=float(h.mean()),
        hopkins_std=float(h.std()),
        hopkins_p75=float(np.percentile(h, 75)),
        hopkins_p95=float(np.percentile(h, 95)),
        n_particles_mean=float(n.mean()),
        n_particles_p95=float(np.percentile(n, 95)),
        mean_knn_mean=float(mk.mean()),
        mean_knn_p05=float(np.percentile(mk, 5)),
    )


def assess_cloud(
    cloud: ParticleCloud,
    config: Optional[GridViabilityConfig] = None,
    reference: Optional[ReferenceStats] = None,
    mean_exposure: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> GridViabilityResult:
    """Classify one tomogram's coordinate cloud."""
    config = config or GridViabilityConfig()
    flags: List[str] = []

    n = cloud.n_particles
    if n < config.min_particles:
        return GridViabilityResult(
            tomo_id=cloud.tomo_id,
            status="insufficient",
            hopkins_h=float("nan"),
            n_particles=n,
            mean_knn_dist=float("nan"),
            min_knn_dist=float("nan"),
            mean_exposure=mean_exposure,
            flags=["low_particle_count"],
            message=f"Fewer than {config.min_particles} particles for QA.",
        )

    h = hopkins_statistic(cloud.coords, rng=rng)
    mean_knn, min_knn = _knn_distance_stats(cloud.coords)

    tile_h_max = None
    if config.use_tiles:
        _, tile_h_max, _ = hopkins_tiled(
            cloud.coords, grid_shape=config.tile_grid, rng=rng
        )

    blob_threshold = config.blob_h_threshold
    n_static_threshold = None
    if reference is not None:
        blob_threshold = max(config.blob_h_threshold, reference.hopkins_p95)
        n_static_threshold = reference.n_particles_p95

    status: ViabilityStatus = "pass"
    message = "Spatial pattern within expected range."

    if abs(h - config.static_h_center) <= config.static_h_tolerance:
        n_cutoff = (
            n_static_threshold * config.static_n_multiplier
            if n_static_threshold is not None
            else 500
        )
        if n >= n_cutoff:
            status = "fail_static"
            flags.append("random_scatter")
            message = "Hopkins ≈ 0.5 with abnormally high particle count — possible localization noise."

    if h >= blob_threshold or (
        tile_h_max is not None
        and np.isfinite(tile_h_max)
        and tile_h_max >= config.tile_h_warn
    ):
        if status != "fail_static":
            status = "fail_blob"
            flags.append("clustering")
            message = "Elevated Hopkins — possible aggregation or dense clumping."
        else:
            flags.append("clustering")

    if reference is not None and status == "pass":
        if h > reference.hopkins_p75:
            status = "warn"
            flags.append("elevated_hopkins")
            message = "Hopkins above reference P75 — review recommended."
        if mean_exposure is not None and mean_exposure < 0.65:
            status = "warn"
            flags.append("low_mean_exposure")
            message = "Low mean steric exposure — possible crowding."

    return GridViabilityResult(
        tomo_id=cloud.tomo_id,
        status=status,
        hopkins_h=h,
        n_particles=n,
        mean_knn_dist=mean_knn,
        min_knn_dist=min_knn,
        tile_h_max=tile_h_max,
        mean_exposure=mean_exposure,
        flags=flags,
        message=message,
    )


def assess_clouds(
    clouds: Sequence[ParticleCloud],
    config: Optional[GridViabilityConfig] = None,
    reference: Optional[ReferenceStats] = None,
    exposure_by_tomo: Optional[dict[str, float]] = None,
) -> List[GridViabilityResult]:
    """Assess multiple tomograms."""
    exposure_by_tomo = exposure_by_tomo or {}
    return [
        assess_cloud(
            cloud,
            config=config,
            reference=reference,
            mean_exposure=exposure_by_tomo.get(cloud.tomo_id),
        )
        for cloud in clouds
    ]


def exposure_means_from_csv(csv_path: str | Path) -> dict[str, float]:
    """Load per-tomo mean steric exposure from a labels CSV if present."""
    import pandas as pd

    df = pd.read_csv(csv_path)
    if "steric_exposure" not in df.columns:
        return {}
    return {
        str(tid): float(grp["steric_exposure"].mean())
        for tid, grp in df.groupby("tomo_id")
    }


def write_report(
    results: Sequence[GridViabilityResult],
    out_path: str | Path,
    reference: Optional[ReferenceStats] = None,
) -> None:
    """Write JSON summary of GVI results."""
    out_path = Path(out_path)
    payload = {
        "results": [r.to_dict() for r in results],
        "summary": {
            "n_tomograms": len(results),
            "n_pass": sum(r.status == "pass" for r in results),
            "n_warn": sum(r.status == "warn" for r in results),
            "n_fail_static": sum(r.status == "fail_static" for r in results),
            "n_fail_blob": sum(r.status == "fail_blob" for r in results),
            "n_insufficient": sum(r.status == "insufficient" for r in results),
        },
    }
    if reference is not None:
        payload["reference"] = reference.to_dict()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
