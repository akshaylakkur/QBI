"""Directional steric exposure metrics from per-ray blockage on the sphere."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .directions import fibonacci_ray_adjacency


@dataclass
class HemisphereMetrics:
    """Derived metrics from a sampled exposure hemisphere."""

    steric_exposure: float
    open_direction: np.ndarray
    anisotropy_index: float
    clean_extraction_score: float
    n_open_components: int
    largest_open_size: int
    clean_cone_half_angle_deg: float


def _build_adjacency_lists(n_rays: int, edge_pairs: np.ndarray) -> List[List[int]]:
    adj: List[List[int]] = [[] for _ in range(n_rays)]
    for i, j in edge_pairs:
        adj[int(i)].append(int(j))
        adj[int(j)].append(int(i))
    return adj


def connected_components_unblocked(
    blocked: np.ndarray,
    edge_pairs: np.ndarray,
) -> Tuple[List[List[int]], int]:
    """Return connected components among unblocked rays and component count."""
    n = len(blocked)
    open_mask = ~blocked
    n_open = int(open_mask.sum())
    if n_open == 0:
        return [], 0

    adj = _build_adjacency_lists(n, edge_pairs)
    visited = np.zeros(n, dtype=bool)
    components: List[List[int]] = []

    for start in range(n):
        if blocked[start] or visited[start]:
            continue
        stack = [start]
        component: List[int] = []
        visited[start] = True
        while stack:
            node = stack.pop()
            component.append(node)
            for nbr in adj[node]:
                if not blocked[nbr] and not visited[nbr]:
                    visited[nbr] = True
                    stack.append(nbr)
        components.append(component)

    return components, len(components)


def open_direction_from_indices(directions: np.ndarray, indices: List[int]) -> np.ndarray:
    """Unit vector = normalized mean of direction vectors at ``indices``."""
    if not indices:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    mean = directions[indices].mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if norm < 1e-8:
        return directions[indices[0]].astype(np.float64)
    return (mean / norm).astype(np.float64)


def clean_cone_half_angle_deg(directions: np.ndarray, component: List[int]) -> float:
    """Half-angle (degrees) of the smallest cone covering ``component`` directions."""
    if not component:
        return 0.0
    if len(component) == 1:
        return 0.0
    axis = open_direction_from_indices(directions, component)
    dots = np.clip(directions[component] @ axis, -1.0, 1.0)
    max_angle = float(np.degrees(np.arccos(dots.min())))
    return max_angle


def anisotropy_index(n_open: int, largest_open_size: int) -> float:
    """0 = all open rays in one component; →1 = scattered openings."""
    if n_open <= 0:
        return 0.0
    if n_open == 1:
        return 0.0
    return float(1.0 - largest_open_size / n_open)


def analyze_hemisphere(
    directions: np.ndarray,
    blocked: np.ndarray,
    edge_pairs: np.ndarray,
    n_rays: int,
) -> HemisphereMetrics:
    """Compute directional exposure metrics from ray blockage."""
    blocked = np.asarray(blocked, dtype=bool).reshape(-1)
    n = len(blocked)
    steric_exposure = float(1.0 - blocked.mean()) if n else 1.0

    components, n_components = connected_components_unblocked(blocked, edge_pairs)
    n_open = n - int(blocked.sum())

    if not components:
        return HemisphereMetrics(
            steric_exposure=steric_exposure,
            open_direction=np.array([0.0, 0.0, 1.0], dtype=np.float64),
            anisotropy_index=0.0,
            clean_extraction_score=0.0,
            n_open_components=0,
            largest_open_size=0,
            clean_cone_half_angle_deg=0.0,
        )

    largest = max(components, key=len)
    largest_size = len(largest)
    open_dir = open_direction_from_indices(directions, largest)
    aniso = anisotropy_index(n_open, largest_size)
    clean_score = largest_size / n_rays
    half_angle = clean_cone_half_angle_deg(directions, largest)

    return HemisphereMetrics(
        steric_exposure=steric_exposure,
        open_direction=open_dir,
        anisotropy_index=aniso,
        clean_extraction_score=float(clean_score),
        n_open_components=n_components,
        largest_open_size=largest_size,
        clean_cone_half_angle_deg=half_angle,
    )


def analyze_hemisphere_from_config(
    directions: np.ndarray,
    blocked: np.ndarray,
    n_rays: int,
    connectivity_epsilon: float = 1.08,
) -> HemisphereMetrics:
    edge_pairs, _ = fibonacci_ray_adjacency(n_rays, epsilon=connectivity_epsilon)
    return analyze_hemisphere(directions, blocked, edge_pairs, n_rays)
