"""Build PyTorch Geometric graphs from exposure label tables."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from torch_geometric.data import Data

from ..constants import PARTICLE_TYPES, RADIUS_ANGSTROM

_TYPE_TO_IDX: Dict[str, int] = {name: i for i, name in enumerate(PARTICLE_TYPES)}
_MAX_RADIUS = max(RADIUS_ANGSTROM.values())


def _one_hot_types(types: Sequence[str]) -> np.ndarray:
    oh = np.zeros((len(types), len(PARTICLE_TYPES)), dtype=np.float32)
    for i, t in enumerate(types):
        oh[i, _TYPE_TO_IDX[t]] = 1.0
    return oh


def _normalize_positions(coords: np.ndarray) -> np.ndarray:
    lo = coords.min(axis=0)
    hi = coords.max(axis=0)
    span = np.maximum(hi - lo, 1.0)
    return ((coords - lo) / span).astype(np.float32)


def _local_density(coords: np.ndarray, radius: float = 300.0) -> np.ndarray:
    tree = cKDTree(coords)
    counts = np.array(
        [len(tree.query_ball_point(coords[i], r=radius)) - 1 for i in range(len(coords))],
        dtype=np.float32,
    )
    if counts.max() > 0:
        counts /= counts.max()
    return counts.reshape(-1, 1)


def _build_edges(
    coords: np.ndarray,
    radii: np.ndarray,
    edge_cutoff: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return edge_index (2, E) and edge_attr (E, 4): dist, dx, dy, dz unit."""
    n = len(coords)
    if n <= 1:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0, 4), dtype=np.float32)

    tree = cKDTree(coords)
    max_r = float(np.max(radii))
    # Union of pairwise reachability balls around each node center.
    pair_radius = max_r + max_r + edge_cutoff
    pairs: set[tuple[int, int]] = set()
    for i in range(n):
        for j in tree.query_ball_point(coords[i], r=pair_radius):
            j = int(j)
            if j != i:
                d = float(np.linalg.norm(coords[j] - coords[i]))
                if d <= radii[i] + radii[j] + edge_cutoff:
                    pairs.add((min(i, j), max(i, j)))

    if not pairs:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0, 4), dtype=np.float32)

    src, dst, attrs = [], [], []
    for i, j in pairs:
        delta = coords[j] - coords[i]
        dist = float(np.linalg.norm(delta))
        unit = delta / dist if dist > 0 else np.array([1.0, 0.0, 0.0])
        attr_ij = np.array([dist / 1000.0, unit[0], unit[1], unit[2]], dtype=np.float32)
        attr_ji = np.array([dist / 1000.0, -unit[0], -unit[1], -unit[2]], dtype=np.float32)
        src.extend([i, j])
        dst.extend([j, i])
        attrs.extend([attr_ij, attr_ji])

    edge_index = np.array([src, dst], dtype=np.int64)
    edge_attr = np.stack(attrs, axis=0).astype(np.float32)
    return edge_index, edge_attr


def _knn_distance_features(coords: np.ndarray, k: int = 5) -> np.ndarray:
    tree = cKDTree(coords)
    kk = min(k + 1, len(coords))
    dists, _ = tree.query(coords, k=kk)
    if dists.ndim == 1:
        dists = dists.reshape(-1, 1)
    if dists.shape[1] == 1:
        min_dist = np.zeros(len(coords), dtype=np.float32)
        mean_knn = np.zeros(len(coords), dtype=np.float32)
    else:
        min_dist = dists[:, 1].astype(np.float32)
        mean_knn = dists[:, 1:].mean(axis=1).astype(np.float32)
    return np.stack([min_dist / 1000.0, mean_knn / 1000.0], axis=1)


def build_node_features(
    types: Sequence[str],
    coords: np.ndarray,
    radii: np.ndarray,
    density_radius: float = 300.0,
) -> np.ndarray:
    """Node feature matrix (N, F)."""
    oh = _one_hot_types(types)
    pos = _normalize_positions(coords)
    r_feat = (radii / _MAX_RADIUS).reshape(-1, 1).astype(np.float32)
    density = _local_density(coords, radius=density_radius)
    knn = _knn_distance_features(coords)
    return np.concatenate([oh, pos, r_feat, density, knn], axis=1).astype(np.float32)


def build_graph_from_dataframe(
    df: pd.DataFrame,
    tomo_id: str,
    edge_cutoff: float = 500.0,
    density_radius: float = 300.0,
) -> Data:
    """Build a single PyG ``Data`` graph for one tomogram."""
    sub = df[df["tomo_id"] == tomo_id].reset_index(drop=True)
    if sub.empty:
        raise ValueError(f"No rows for tomogram {tomo_id}")

    types = sub["particle_type"].astype(str).tolist()
    coords = sub[["x", "y", "z"]].to_numpy(dtype=np.float64)
    if "radius_angstrom" in sub.columns:
        radii = sub["radius_angstrom"].to_numpy(dtype=np.float64)
    else:
        radii = np.array([RADIUS_ANGSTROM[t] for t in types], dtype=np.float64)

    x = build_node_features(types, coords, radii, density_radius=density_radius)
    edge_index, edge_attr = _build_edges(coords, radii, edge_cutoff=edge_cutoff)
    y = sub["steric_exposure"].to_numpy(dtype=np.float32).reshape(-1, 1)

    data = Data(
        x=torch.from_numpy(x),
        edge_index=torch.from_numpy(edge_index),
        edge_attr=torch.from_numpy(edge_attr),
        y=torch.from_numpy(y),
        pos=torch.from_numpy(coords.astype(np.float32)),
    )
    data.tomo_id = tomo_id
    data.num_nodes = len(types)
    return data


def graphs_from_exposure_csv(
    csv_path: str | Path,
    tomo_ids: Optional[Sequence[str]] = None,
    edge_cutoff: float = 500.0,
    density_radius: float = 300.0,
) -> List[Data]:
    """Load exposure labels CSV and return one PyG graph per tomogram."""
    df = pd.read_csv(csv_path)
    required = {"tomo_id", "particle_type", "x", "y", "z", "steric_exposure"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")

    if tomo_ids is None:
        tomo_ids = sorted(df["tomo_id"].unique())

    return [
        build_graph_from_dataframe(
            df, tid, edge_cutoff=edge_cutoff, density_radius=density_radius
        )
        for tid in tomo_ids
    ]


def split_graphs_by_tomo(
    graphs: List[Data],
    val_tomo_ids: Sequence[str],
) -> Tuple[List[Data], List[Data]]:
    val_set = set(val_tomo_ids)
    train = [g for g in graphs if g.tomo_id not in val_set]
    val = [g for g in graphs if g.tomo_id in val_set]
    if not train or not val:
        raise ValueError("Train/val split produced an empty partition")
    return train, val
