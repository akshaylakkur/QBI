"""Physics-aligned node/edge features for steric exposure graphs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree

from ..constants import PARTICLE_TYPES, RADIUS_ANGSTROM

_TYPE_TO_IDX = {name: i for i, name in enumerate(PARTICLE_TYPES)}
_MAX_RADIUS = max(RADIUS_ANGSTROM.values())
SHELL_EDGES = (50.0, 100.0, 200.0, 400.0, 800.0)


@dataclass
class FeatureStats:
    """Per-feature mean/std for standardization."""

    mean: np.ndarray
    std: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / np.maximum(self.std, 1e-6)).astype(np.float32)

    @classmethod
    def fit(cls, arrays: Sequence[np.ndarray]) -> "FeatureStats":
        stacked = np.concatenate(arrays, axis=0)
        return cls(mean=stacked.mean(0), std=stacked.std(0))

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureStats":
        return cls(mean=np.array(d["mean"]), std=np.array(d["std"]))


def one_hot_types(types: Sequence[str]) -> np.ndarray:
    oh = np.zeros((len(types), len(PARTICLE_TYPES)), dtype=np.float32)
    for i, t in enumerate(types):
        if t in _TYPE_TO_IDX:
            oh[i, _TYPE_TO_IDX[t]] = 1.0
    return oh


def normalize_positions(coords: np.ndarray) -> np.ndarray:
    lo = coords.min(axis=0)
    hi = coords.max(axis=0)
    span = np.maximum(hi - lo, 1.0)
    return ((coords - lo) / span).astype(np.float32)


def shell_histogram_features_fast(
    coords: np.ndarray,
    radii: np.ndarray,
    shells: Tuple[float, ...] = SHELL_EDGES,
) -> np.ndarray:
    n = len(coords)
    if n <= 1:
        return np.zeros((n, len(shells)), dtype=np.float32)
    tree = cKDTree(coords)
    feats = np.zeros((n, len(shells)), dtype=np.float32)
    for i in range(n):
        dists, idxs = tree.query(coords[i], k=n)
        if np.isscalar(dists):
            dists, idxs = np.array([dists]), np.array([idxs])
        for j, d in zip(idxs, dists):
            if j == i:
                continue
            for si, edge in enumerate(shells):
                if d <= edge:
                    feats[i, si] += radii[j]
                    break
    if feats.max() > 0:
        feats /= feats.max()
    return feats.astype(np.float32)


def knn_distance_features(coords: np.ndarray, k: int = 5) -> np.ndarray:
    tree = cKDTree(coords)
    kk = min(k + 1, len(coords))
    dists, _ = tree.query(coords, k=kk)
    if dists.ndim == 1:
        dists = dists.reshape(-1, 1)
    if dists.shape[1] == 1:
        return np.zeros((len(coords), 2), dtype=np.float32)
    min_dist = dists[:, 1].astype(np.float32)
    mean_knn = dists[:, 1:].mean(axis=1).astype(np.float32)
    return np.stack([min_dist / 1000.0, mean_knn / 1000.0], axis=1)


def log_neighbor_count(coords: np.ndarray, radius: float = 500.0) -> np.ndarray:
    tree = cKDTree(coords)
    counts = np.array(
        [max(0, len(tree.query_ball_point(coords[i], r=radius)) - 1) for i in range(len(coords))],
        dtype=np.float32,
    )
    return np.log1p(counts).reshape(-1, 1) / np.log1p(50.0)


def build_node_features(
    types: Sequence[str],
    coords: np.ndarray,
    radii: np.ndarray,
    shells: Tuple[float, ...] = SHELL_EDGES,
    neighbor_radius: float = 500.0,
) -> np.ndarray:
    oh = one_hot_types(types)
    pos = normalize_positions(coords)
    r_feat = (radii / _MAX_RADIUS).reshape(-1, 1).astype(np.float32)
    shells_f = shell_histogram_features_fast(coords, radii, shells)
    knn = knn_distance_features(coords)
    nbr = log_neighbor_count(coords, neighbor_radius)
    return np.concatenate([oh, pos, r_feat, shells_f, knn, nbr], axis=1).astype(np.float32)


def edge_attr_ij(
    coord_i: np.ndarray,
    coord_j: np.ndarray,
    radius_i: float,
    radius_j: float,
) -> np.ndarray:
    delta = coord_j - coord_i
    dist = float(np.linalg.norm(delta))
    unit = delta / dist if dist > 1e-8 else np.array([1.0, 0.0, 0.0])
    radius_ratio = radius_j / max(radius_i, 1e-6)
    overlap = max(0.0, (radius_i + radius_j) - dist) / max(radius_i, 1e-6)
    solid_angle = min(1.0, (radius_j / max(dist, 1e-6)) ** 2)
    return np.array(
        [dist / 1000.0, unit[0], unit[1], unit[2], radius_ratio, overlap, solid_angle],
        dtype=np.float32,
    )


EDGE_DIM = 7


def build_edges(
    coords: np.ndarray,
    radii: np.ndarray,
    edge_cutoff: float = 500.0,
    knn_k: int = 12,
) -> Tuple[np.ndarray, np.ndarray]:
    """Steric edges (within r_i+r_j+cutoff) union kNN edges."""
    n = len(coords)
    if n <= 1:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0, EDGE_DIM), dtype=np.float32)

    tree = cKDTree(coords)
    pairs: set[tuple[int, int]] = set()

    max_r = float(np.max(radii))
    pair_radius = max_r + max_r + edge_cutoff
    for i in range(n):
        for j in tree.query_ball_point(coords[i], r=pair_radius):
            j = int(j)
            if j != i:
                d = float(np.linalg.norm(coords[j] - coords[i]))
                if d <= radii[i] + radii[j] + edge_cutoff:
                    pairs.add((min(i, j), max(i, j)))

    kk = min(knn_k + 1, n)
    dists, idxs = tree.query(coords, k=kk)
    if idxs.ndim == 1:
        idxs = idxs.reshape(-1, 1)
    for i in range(n):
        for j in idxs[i, 1:]:
            j = int(j)
            if j != i:
                pairs.add((min(i, j), max(i, j)))

    src, dst, attrs = [], [], []
    for i, j in pairs:
        attr_ij = edge_attr_ij(coords[i], coords[j], radii[i], radii[j])
        attr_ji = edge_attr_ij(coords[j], coords[i], radii[j], radii[i])
        src.extend([i, j])
        dst.extend([j, i])
        attrs.extend([attr_ij, attr_ji])

    return np.array([src, dst], dtype=np.int64), np.stack(attrs, axis=0).astype(np.float32)


def auxiliary_targets(coords: np.ndarray, neighbor_radius: float = 500.0) -> np.ndarray:
    """(N, 2): log_min_dist/scale, normalized neighbor count."""
    knn = knn_distance_features(coords)
    nbr = log_neighbor_count(coords, neighbor_radius)
    return np.concatenate([knn[:, :1] * 10.0, nbr], axis=1).astype(np.float32)
