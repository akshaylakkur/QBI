"""Synthetic hard-sphere packings for large-scale GNN pretraining."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from tqdm import tqdm

from ..constants import PARTICLE_TYPES, RADIUS_ANGSTROM, radius_for_type
from ..geometry.exposure import steric_exposure_batch
from ..types import ExposureConfig, ParticleCloud
from .graph import build_graph_from_arrays


def _random_types(n: int, rng: np.random.Generator) -> List[str]:
    return [PARTICLE_TYPES[int(rng.integers(0, len(PARTICLE_TYPES)))] for _ in range(n)]


def generate_packing(
    rng: np.random.Generator,
    n_particles: int,
    box_size: np.ndarray | None = None,
) -> ParticleCloud:
    if box_size is None:
        box_size = rng.uniform(1200, 4500, size=3)

    types = _random_types(n_particles, rng)
    radii = np.array([radius_for_type(t) for t in types], dtype=np.float64)
    order = np.argsort(-radii)
    coords = np.zeros((n_particles, 3), dtype=np.float64)
    placed: list[int] = []

    for idx in order:
        r = radii[idx]
        for _ in range(300):
            c = rng.uniform(r, box_size - r)
            if not placed or all(
                np.linalg.norm(c - coords[j]) >= 0.8 * (r + radii[j]) for j in placed
            ):
                coords[idx] = c
                placed.append(idx)
                break
        else:
            coords[idx] = rng.uniform(r, box_size - r)
            placed.append(idx)

    return ParticleCloud(
        tomo_id=f"syn_{rng.integers(1_000_000)}",
        types=types,
        coords=coords,
        radii=radii,
    )


def generate_synthetic_corpus(
    target_nodes: int = 50_000,
    n_rays: int = 2000,
    seed: int = 0,
    min_per_graph: int = 40,
    max_per_graph: int = 250,
) -> Tuple[List[ParticleCloud], List[np.ndarray]]:
    """Generate synthetic packings until ``target_nodes`` particles are labeled."""
    rng = np.random.default_rng(seed)
    clouds: List[ParticleCloud] = []
    exposures: List[np.ndarray] = []
    total = 0
    cfg = ExposureConfig(n_rays=n_rays)

    pbar = tqdm(total=target_nodes, desc="Synthetic labels")
    while total < target_nodes:
        n = int(rng.integers(min_per_graph, max_per_graph + 1))
        cloud = generate_packing(rng, n)
        exp = np.array(
            [r.steric_exposure for r in steric_exposure_batch(cloud, config=cfg)],
            dtype=np.float32,
        )
        clouds.append(cloud)
        exposures.append(exp)
        total += n
        pbar.update(n)
    pbar.close()
    return clouds, exposures


def clouds_to_graphs(
    clouds: List[ParticleCloud],
    exposures: List[np.ndarray],
    feature_stats=None,
    edge_cutoff: float = 500.0,
    knn_k: int = 12,
) -> list:
    graphs = []
    for cloud, exp in zip(clouds, exposures):
        graphs.append(
            build_graph_from_arrays(
                cloud.tomo_id,
                cloud.types,
                cloud.coords,
                cloud.radii,
                exp,
                edge_cutoff=edge_cutoff,
                knn_k=knn_k,
                feature_stats=feature_stats,
            )
        )
    return graphs
