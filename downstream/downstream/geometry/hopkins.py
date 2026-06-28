"""Hopkins statistic for 3D spatial point patterns (clustering vs randomness)."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree


def bounding_box(
    coords: np.ndarray,
    padding: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (lo, hi) axis-aligned bounds for ``coords`` (N, 3)."""
    coords = np.asarray(coords, dtype=np.float64)
    lo = coords.min(axis=0)
    hi = coords.max(axis=0)
    if padding > 0:
        lo = lo - padding
        hi = hi + padding
    return lo, hi


def hopkins_statistic(
    coords: np.ndarray,
    bbox: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    n_ghosts: Optional[int] = None,
    n_sample: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Compute Hopkins statistic H in [0, 1] for a 3D point set.

    - H ≈ 0.5: spatially random (Poisson-like)
    - H → 1.0: clustered
    - H → 0.0: regular / over-dispersed

    Parameters
    ----------
    coords:
        ``(N, 3)`` particle coordinates.
    bbox:
        Optional ``(lo, hi)`` study region for ghost points. Defaults to
        min/max of ``coords``.
    n_ghosts:
        Number of uniform ghost points in ``bbox``. Defaults to ``N``.
    n_sample:
        Number of data points sampled for the W term. Defaults to ``min(N, 500)``.
    """
    coords = np.asarray(coords, dtype=np.float64)
    n = coords.shape[0]
    if n < 2:
        return float("nan")

    rng = rng or np.random.default_rng()
    if bbox is None:
        lo, hi = bounding_box(coords)
    else:
        lo, hi = np.asarray(bbox[0], dtype=np.float64), np.asarray(bbox[1], dtype=np.float64)

    span = np.maximum(hi - lo, 1e-6)
    n_ghosts = n if n_ghosts is None else int(n_ghosts)
    n_sample = min(n, 500) if n_sample is None else min(int(n_sample), n)

    tree = cKDTree(coords)

    # U: ghost → nearest real particle
    ghosts = lo + rng.random((n_ghosts, 3)) * span
    u_dists, _ = tree.query(ghosts, k=1)
    u_sum = float(np.sum(u_dists))

    # W: real particle → nearest other real particle
    sample_idx = rng.choice(n, size=n_sample, replace=False)
    w_dists = []
    for i in sample_idx:
        dists, idxs = tree.query(coords[i], k=2)
        if np.isscalar(dists):
            w_dists.append(float(dists))
        else:
            w_dists.append(float(dists[1]))
    w_sum = float(np.sum(w_dists))

    denom = u_sum + w_sum
    if denom <= 0:
        return float("nan")
    return u_sum / denom


def hopkins_tiled(
    coords: np.ndarray,
    grid_shape: Tuple[int, int, int] = (2, 2, 2),
    **kwargs,
) -> Tuple[float, float, np.ndarray]:
    """Compute Hopkins per sub-volume tile.

    Returns ``(global_h, max_tile_h, tile_h_grid)`` where ``tile_h_grid`` has
    shape ``grid_shape``.
    """
    coords = np.asarray(coords, dtype=np.float64)
    lo, hi = bounding_box(coords)
    span = np.maximum(hi - lo, 1e-6)
    gx, gy, gz = grid_shape

    tile_h = np.full(grid_shape, np.nan, dtype=np.float64)
    for ix in range(gx):
        for iy in range(gy):
            for iz in range(gz):
                tile_lo = lo + span * np.array([ix / gx, iy / gy, iz / gz])
                tile_hi = lo + span * np.array(
                    [(ix + 1) / gx, (iy + 1) / gy, (iz + 1) / gz]
                )
                mask = np.all((coords >= tile_lo) & (coords < tile_hi), axis=1)
                sub = coords[mask]
                if sub.shape[0] >= 2:
                    tile_h[ix, iy, iz] = hopkins_statistic(
                        sub, bbox=(tile_lo, tile_hi), **kwargs
                    )

    global_h = hopkins_statistic(coords, **kwargs)
    finite = tile_h[np.isfinite(tile_h)]
    max_tile = float(finite.max()) if finite.size else float("nan")
    return global_h, max_tile, tile_h
