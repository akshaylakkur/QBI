"""Monte Carlo steric exposure via surface raycasting against hard spheres."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

from ..types import ExposureConfig, ExposureResult, ParticleCloud
from .directions import fibonacci_directions


def _neighbor_cutoff_radius(cloud: ParticleCloud, target_index: int, config: ExposureConfig) -> float:
    """Max center distance for a neighbor that can occlude any outward ray."""
    max_r = float(np.max(cloud.radii))
    coords = cloud.coords
    if cloud.n_particles <= 1:
        return cloud.radii[target_index] + max_r + config.neighbor_margin

    # Upper bound on how far apart centers can be while still allowing occlusion
    # along some forward ray from the target surface.
    dists = np.linalg.norm(coords - coords[target_index], axis=1)
    l_max = float(np.max(dists))
    return cloud.radii[target_index] + max_r + l_max + config.neighbor_margin


def _candidate_neighbors(
    tree: cKDTree,
    cloud: ParticleCloud,
    target_index: int,
    config: ExposureConfig,
) -> np.ndarray:
    cutoff = _neighbor_cutoff_radius(cloud, target_index, config)
    idxs = tree.query_ball_point(cloud.coords[target_index], r=cutoff)
    return np.array([j for j in idxs if j != target_index], dtype=np.int64)


def rays_blocked_by_neighbors(
    origin: np.ndarray,
    directions: np.ndarray,
    neighbor_centers: np.ndarray,
    neighbor_radii_eff: np.ndarray,
) -> np.ndarray:
    """Return boolean mask ``(M,)`` — True if ray is blocked by any neighbor.

    Parameters
    ----------
    origin:
        ``(M, 3)`` ray origins (one per direction, on target surface).
    directions:
        ``(M, 3)`` unit direction vectors.
    neighbor_centers:
        ``(K, 3)`` neighbor sphere centers.
    neighbor_radii_eff:
        ``(K,)`` effective neighbor radii (includes probe inflation).
    """
    m = directions.shape[0]
    if neighbor_centers.size == 0:
        return np.zeros(m, dtype=bool)

    # v[k, m, :] = neighbor_center[k] - origin[m]
    v = neighbor_centers[:, None, :] - origin[None, :, :]
    t0 = np.einsum("kmd,md->km", v, directions)
    perp = v - t0[:, :, None] * directions[None, :, :]
    d_perp = np.linalg.norm(perp, axis=2)

    forward = t0 > 0.0
    hit = forward & (d_perp < neighbor_radii_eff[:, None])
    return hit.any(axis=0)


def steric_exposure_one(
    cloud: ParticleCloud,
    target_index: int,
    config: Optional[ExposureConfig] = None,
    directions: Optional[np.ndarray] = None,
    tree: Optional[cKDTree] = None,
) -> ExposureResult:
    """Compute steric exposure for a single particle in ``cloud``."""
    config = config or ExposureConfig()
    if directions is None:
        directions = fibonacci_directions(config.n_rays)
    elif directions.shape[0] != config.n_rays:
        config.n_rays = directions.shape[0]

    if tree is None:
        tree = cKDTree(cloud.coords)

    center = cloud.coords[target_index]
    radius = cloud.radii[target_index]
    origins = center + (radius + config.surface_epsilon) * directions

    neighbor_idxs = _candidate_neighbors(tree, cloud, target_index, config)
    if neighbor_idxs.size == 0:
        return ExposureResult(
            index=target_index,
            steric_exposure=1.0,
            n_rays=config.n_rays,
            n_neighbors_considered=0,
            n_rays_blocked=0,
        )

    neighbor_centers = cloud.coords[neighbor_idxs]
    neighbor_radii_eff = cloud.radii[neighbor_idxs] + config.probe_radius

    blocked = rays_blocked_by_neighbors(
        origins, directions, neighbor_centers, neighbor_radii_eff
    )
    n_blocked = int(blocked.sum())
    exposure = 1.0 - (n_blocked / config.n_rays)

    return ExposureResult(
        index=target_index,
        steric_exposure=float(exposure),
        n_rays=config.n_rays,
        n_neighbors_considered=int(neighbor_idxs.size),
        n_rays_blocked=n_blocked,
    )


def steric_exposure_batch(
    cloud: ParticleCloud,
    config: Optional[ExposureConfig] = None,
    target_indices: Optional[np.ndarray] = None,
) -> List[ExposureResult]:
    """Compute steric exposure for all (or selected) particles in ``cloud``."""
    config = config or ExposureConfig()
    directions = fibonacci_directions(config.n_rays)
    tree = cKDTree(cloud.coords)

    if target_indices is None:
        target_indices = np.arange(cloud.n_particles, dtype=np.int64)
    else:
        target_indices = np.asarray(target_indices, dtype=np.int64)

    results: List[ExposureResult] = []
    for idx in target_indices:
        results.append(
            steric_exposure_one(
                cloud,
                int(idx),
                config=config,
                directions=directions,
                tree=tree,
            )
        )
    return results
