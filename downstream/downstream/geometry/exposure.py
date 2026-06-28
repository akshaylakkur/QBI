"""Monte Carlo steric exposure via surface raycasting against hard spheres."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

from ..types import ExposureConfig, ExposureResult, ParticleCloud
from .directions import fibonacci_directions, fibonacci_ray_adjacency
from .hemisphere import analyze_hemisphere


def _neighbor_cutoff_radius(cloud: ParticleCloud, target_index: int, config: ExposureConfig) -> float:
    """Max center distance for a neighbor that can occlude any outward ray."""
    max_r = float(np.max(cloud.radii))
    coords = cloud.coords
    if cloud.n_particles <= 1:
        return cloud.radii[target_index] + max_r + config.neighbor_margin

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
    """Return boolean mask ``(M,)`` — True if ray is blocked by any neighbor."""
    m = directions.shape[0]
    if neighbor_centers.size == 0:
        return np.zeros(m, dtype=bool)

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
    edge_pairs: Optional[np.ndarray] = None,
) -> ExposureResult:
    """Compute steric exposure for a single particle in ``cloud``."""
    config = config or ExposureConfig()
    if directions is None:
        directions = fibonacci_directions(config.n_rays)
    elif directions.shape[0] != config.n_rays:
        config.n_rays = directions.shape[0]

    if edge_pairs is None:
        edge_pairs, _ = fibonacci_ray_adjacency(
            config.n_rays, epsilon=config.connectivity_epsilon
        )

    if tree is None:
        tree = cKDTree(cloud.coords)

    center = cloud.coords[target_index]
    radius = cloud.radii[target_index]
    origins = center + (radius + config.surface_epsilon) * directions

    neighbor_idxs = _candidate_neighbors(tree, cloud, target_index, config)
    if neighbor_idxs.size == 0:
        metrics = analyze_hemisphere(
            directions,
            np.zeros(config.n_rays, dtype=bool),
            edge_pairs,
            config.n_rays,
        )
        return ExposureResult(
            index=target_index,
            steric_exposure=metrics.steric_exposure,
            n_rays=config.n_rays,
            n_neighbors_considered=0,
            n_rays_blocked=0,
            open_direction=metrics.open_direction,
            anisotropy_index=metrics.anisotropy_index,
            clean_extraction_score=metrics.clean_extraction_score,
            n_open_components=metrics.n_open_components,
            clean_cone_half_angle_deg=metrics.clean_cone_half_angle_deg,
            blocked=np.zeros(config.n_rays, dtype=bool),
            directions=directions,
        )

    neighbor_centers = cloud.coords[neighbor_idxs]
    neighbor_radii_eff = cloud.radii[neighbor_idxs] + config.probe_radius

    blocked = rays_blocked_by_neighbors(
        origins, directions, neighbor_centers, neighbor_radii_eff
    )
    n_blocked = int(blocked.sum())
    metrics = analyze_hemisphere(directions, blocked, edge_pairs, config.n_rays)

    return ExposureResult(
        index=target_index,
        steric_exposure=metrics.steric_exposure,
        n_rays=config.n_rays,
        n_neighbors_considered=int(neighbor_idxs.size),
        n_rays_blocked=n_blocked,
        open_direction=metrics.open_direction,
        anisotropy_index=metrics.anisotropy_index,
        clean_extraction_score=metrics.clean_extraction_score,
        n_open_components=metrics.n_open_components,
        clean_cone_half_angle_deg=metrics.clean_cone_half_angle_deg,
        blocked=blocked.copy(),
        directions=directions,
    )


def steric_exposure_batch(
    cloud: ParticleCloud,
    config: Optional[ExposureConfig] = None,
    target_indices: Optional[np.ndarray] = None,
) -> List[ExposureResult]:
    """Compute steric exposure for all (or selected) particles in ``cloud``."""
    config = config or ExposureConfig()
    directions = fibonacci_directions(config.n_rays)
    edge_pairs, _ = fibonacci_ray_adjacency(
        config.n_rays, epsilon=config.connectivity_epsilon
    )
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
                edge_pairs=edge_pairs,
            )
        )
    return results
