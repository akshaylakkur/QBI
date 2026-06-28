"""Graph augmentations (SE(3) invariant exposure targets)."""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data

from ..types import ExposureConfig, ParticleCloud
from ..geometry.exposure import steric_exposure_batch
from .features import FeatureStats
from .graph import build_graph_from_arrays


def _random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    a, b, c = rng.uniform(0, 2 * np.pi, size=3)
    rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    rx = np.array([[1, 0, 0], [0, np.cos(c), -np.sin(c)], [0, np.sin(c), np.cos(c)]])
    return rz @ ry @ rx


def _rebuild_from_coords(
    data: Data,
    coords: np.ndarray,
    feature_stats: FeatureStats | None,
    edge_cutoff: float,
    knn_k: int,
    n_rays: int,
) -> Data:
    types = list(getattr(data, "types", []))
    radii = data.radii.numpy() if hasattr(data, "radii") else np.full(len(coords), 90.0)
    cloud = ParticleCloud(tomo_id=data.tomo_id, types=types, coords=coords, radii=radii)
    exposure = np.array(
        [r.steric_exposure for r in steric_exposure_batch(cloud, config=ExposureConfig(n_rays=n_rays))],
        dtype=np.float32,
    )
    return build_graph_from_arrays(
        data.tomo_id, types, coords, radii, exposure,
        edge_cutoff=edge_cutoff, knn_k=knn_k, feature_stats=feature_stats,
    )


def _particle_coords(data: Data) -> np.ndarray:
    mask = ~data.virtual_mask.numpy() if hasattr(data, "virtual_mask") else np.ones(data.pos.shape[0], bool)
    return data.pos.numpy()[mask]


def augment_rigid(
    data: Data,
    rng: np.random.Generator,
    feature_stats: FeatureStats | None = None,
    edge_cutoff: float = 500.0,
    knn_k: int = 12,
    n_rays: int = 300,
) -> Data:
    coords = _particle_coords(data)
    center = coords.mean(axis=0)
    R = _random_rotation_matrix(rng)
    t = rng.uniform(-200, 200, size=3)
    coords = (coords - center) @ R.T + center + t
    return _rebuild_from_coords(data, coords, feature_stats, edge_cutoff, knn_k, n_rays)


def augment_jitter(
    data: Data,
    rng: np.random.Generator,
    sigma: float = 15.0,
    feature_stats: FeatureStats | None = None,
    edge_cutoff: float = 500.0,
    knn_k: int = 12,
    n_rays: int = 250,
) -> Data:
    coords = _particle_coords(data) + rng.normal(0, sigma, size=_particle_coords(data).shape)
    return _rebuild_from_coords(data, coords, feature_stats, edge_cutoff, knn_k, n_rays)


def augment_node_dropout(
    data: Data,
    rng: np.random.Generator,
    drop_frac: float = 0.1,
    feature_stats: FeatureStats | None = None,
    edge_cutoff: float = 500.0,
    knn_k: int = 12,
    n_rays: int = 250,
) -> Data:
    coords = _particle_coords(data)
    n = len(coords)
    if n <= 5:
        return data
    keep = rng.random(n) > drop_frac
    if keep.sum() < 5:
        return data
    coords = coords[keep]
    types = [data.types[i] for i, k in enumerate(keep) if k]
    radii = data.radii.numpy()[keep]
    cloud = ParticleCloud(tomo_id=data.tomo_id, types=types, coords=coords, radii=radii)
    exposure = np.array(
        [r.steric_exposure for r in steric_exposure_batch(cloud, config=ExposureConfig(n_rays=n_rays))],
        dtype=np.float32,
    )
    return build_graph_from_arrays(
        data.tomo_id, types, coords, radii, exposure,
        edge_cutoff=edge_cutoff, knn_k=knn_k, feature_stats=feature_stats,
    )


def augment_graph(
    data: Data,
    rng: np.random.Generator,
    feature_stats: FeatureStats | None = None,
    edge_cutoff: float = 500.0,
    knn_k: int = 12,
) -> Data:
    choice = rng.integers(0, 4)
    if choice == 0:
        return augment_rigid(data, rng, feature_stats, edge_cutoff, knn_k)
    if choice == 1:
        return augment_jitter(data, rng, feature_stats=feature_stats, edge_cutoff=edge_cutoff, knn_k=knn_k)
    if choice == 2:
        return augment_node_dropout(data, rng, feature_stats=feature_stats, edge_cutoff=edge_cutoff, knn_k=knn_k)
    return data
