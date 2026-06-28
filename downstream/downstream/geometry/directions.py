"""Uniformly distributed unit directions on the sphere (Fibonacci / golden spiral)."""

from __future__ import annotations

from functools import lru_cache
from typing import Tuple

import numpy as np

_GOLDEN_ANGLE = np.pi * (3.0 - np.sqrt(5.0))


def angular_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Geodesic angle (radians) between unit vectors."""
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return float(np.arccos(dot))


def fibonacci_directions(n: int) -> np.ndarray:
    """Return ``(n, 3)`` unit vectors uniformly distributed on S^2.

    Uses a golden-angle spiral (Fibonacci sphere), which avoids polar clustering
    that naive latitude/longitude sampling introduces.
    """
    if n <= 0:
        raise ValueError("n must be positive")

    i = np.arange(n, dtype=np.float64)
    y = 1.0 - (2.0 * i + 1.0) / n
    r = np.sqrt(np.maximum(0.0, 1.0 - y * y))
    theta = _GOLDEN_ANGLE * i
    x = np.cos(theta) * r
    z = np.sin(theta) * r
    dirs = np.stack([x, y, z], axis=1)
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    return dirs / norms


@lru_cache(maxsize=16)
def fibonacci_ray_adjacency(n_rays: int, epsilon: float = 1.08) -> Tuple[np.ndarray, float]:
    """Return (edge_pairs, connect_angle_deg) for the Fibonacci ray mesh.

    Connect rays *i* and *j* when their angular separation is at most
    ``epsilon * max_k nn_dist(k)``, where ``nn_dist(k)`` is the nearest-neighbor
    angle on the Fibonacci sphere.  This scales with ``n_rays`` and links each
    ray to its local neighborhood (not just a single NN edge).
    """
    if n_rays <= 1:
        return np.zeros((0, 2), dtype=np.int64), 0.0

    from scipy.spatial import cKDTree

    dirs = fibonacci_directions(n_rays)
    dots = np.clip(dirs @ dirs.T, -1.0, 1.0)
    np.fill_diagonal(dots, -1.0)
    nn_dist = np.arccos(np.max(dots, axis=1))
    threshold = float(nn_dist.max() * epsilon)
    connect_angle_deg = float(np.degrees(threshold))

    # Chord length on unit sphere for angular threshold
    chord = 2.0 * np.sin(threshold / 2.0)
    tree = cKDTree(dirs)
    pairs: set[tuple[int, int]] = set()
    for i in range(n_rays):
        for j in tree.query_ball_point(dirs[i], r=chord + 1e-9):
            j = int(j)
            if j != i:
                pairs.add((min(i, j), max(i, j)))

    edge_pairs = (
        np.array(sorted(pairs), dtype=np.int64)
        if pairs
        else np.zeros((0, 2), dtype=np.int64)
    )
    return edge_pairs, connect_angle_deg
