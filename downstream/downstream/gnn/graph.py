"""Build PyG graphs with physics-aligned node/edge features."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from ..constants import RADIUS_ANGSTROM
from .features import EDGE_DIM, FeatureStats, build_edges, build_node_features


def build_graph_from_arrays(
    tomo_id: str,
    types: Sequence[str],
    coords: np.ndarray,
    radii: np.ndarray,
    exposure: np.ndarray,
    edge_cutoff: float = 500.0,
    knn_k: int = 12,
    feature_stats: Optional[FeatureStats] = None,
) -> Data:
    coords = np.asarray(coords, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64)
    exposure = np.asarray(exposure, dtype=np.float32).reshape(-1)

    x = build_node_features(types, coords, radii, neighbor_radius=edge_cutoff)
    if feature_stats is not None:
        x = feature_stats.transform(x)

    edge_index, edge_attr = build_edges(coords, radii, edge_cutoff=edge_cutoff, knn_k=knn_k)

    data = Data(
        x=torch.from_numpy(x.astype(np.float32)),
        edge_index=torch.from_numpy(edge_index),
        edge_attr=torch.from_numpy(edge_attr),
        y=torch.from_numpy(exposure.reshape(-1, 1)),
        pos=torch.from_numpy(coords.astype(np.float32)),
    )
    data.tomo_id = tomo_id
    data.num_nodes = len(types)
    data.types = list(types)
    data.radii = torch.from_numpy(radii.astype(np.float32))
    return data


def build_graph_from_dataframe(
    df: pd.DataFrame,
    tomo_id: str,
    edge_cutoff: float = 500.0,
    knn_k: int = 12,
    feature_stats: Optional[FeatureStats] = None,
) -> Data:
    sub = df[df["tomo_id"] == tomo_id].reset_index(drop=True)
    if sub.empty:
        raise ValueError(f"No rows for tomogram {tomo_id}")

    types = sub["particle_type"].astype(str).tolist()
    coords = sub[["x", "y", "z"]].to_numpy(dtype=np.float64)
    radii = (
        sub["radius_angstrom"].to_numpy(dtype=np.float64)
        if "radius_angstrom" in sub.columns
        else np.array([RADIUS_ANGSTROM[t] for t in types], dtype=np.float64)
    )
    exposure = sub["steric_exposure"].to_numpy(dtype=np.float32)
    return build_graph_from_arrays(
        tomo_id, types, coords, radii, exposure,
        edge_cutoff=edge_cutoff, knn_k=knn_k, feature_stats=feature_stats,
    )


def graphs_from_exposure_csv(
    csv_path: str | Path,
    tomo_ids: Optional[Sequence[str]] = None,
    edge_cutoff: float = 500.0,
    knn_k: int = 12,
    feature_stats: Optional[FeatureStats] = None,
) -> List[Data]:
    df = pd.read_csv(csv_path)
    required = {"tomo_id", "particle_type", "x", "y", "z", "steric_exposure"}
    if missing := required - set(df.columns):
        raise ValueError(f"CSV missing columns: {sorted(missing)}")
    if tomo_ids is None:
        tomo_ids = sorted(df["tomo_id"].unique())
    return [
        build_graph_from_dataframe(df, tid, edge_cutoff, knn_k, feature_stats)
        for tid in tomo_ids
    ]


def fit_feature_stats_on_graphs(graphs: List[Data]) -> FeatureStats:
    return FeatureStats.fit([g.x.numpy() for g in graphs])
