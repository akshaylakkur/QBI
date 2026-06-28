"""Detailed evaluation metrics for exposure GNN."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, r2_score

from .baseline import ridge_baseline_metrics
from .model import ExposureGNN, exposure_from_residual


def _node_mask(g) -> np.ndarray:
    if hasattr(g, "virtual_mask"):
        return ~g.virtual_mask.numpy()
    return np.ones(g.num_nodes, dtype=bool)


@torch.no_grad()
def predict_graph(model: ExposureGNN, g, device: torch.device) -> np.ndarray:
    model.eval()
    g = g.to(device)
    pred_res, _ = model(g.x, g.edge_index, g.edge_attr, virtual_mask=g.virtual_mask)
    pred = exposure_from_residual(pred_res.squeeze(-1), g.y_mean, g.virtual_mask)
    return pred.cpu().numpy()


def evaluate_graphs(
    model: ExposureGNN,
    graphs: List,
    device: torch.device,
    types_key: str = "types",
) -> Dict:
    ys, preds, types_all = [], [], []
    per_tomo = {}

    for g in graphs:
        pred = predict_graph(model, g, device)
        mask = _node_mask(g)
        y = g.y_abs.numpy().ravel()[mask]
        ys.append(y)
        preds.append(pred)
        tomo_types = list(getattr(g, types_key, ["?"] * mask.sum()))
        types_all.extend(tomo_types)
        per_tomo[g.tomo_id] = {
            "mae": float(mean_absolute_error(y, pred)),
            "r2": float(r2_score(y, pred)) if len(y) > 1 else 0.0,
            "n": int(len(y)),
            "mean_density": float(g.x[mask, -1].mean()) if g.x.shape[1] > 0 else 0.0,
        }

    y_all = np.concatenate(ys)
    p_all = np.concatenate(preds)
    types_all = np.array(types_all)

    by_class = {}
    for t in np.unique(types_all):
        m = types_all == t
        by_class[t] = {
            "mae": float(mean_absolute_error(y_all[m], p_all[m])),
            "r2": float(r2_score(y_all[m], p_all[m])) if m.sum() > 1 else 0.0,
            "n": int(m.sum()),
        }

    return {
        "mae": float(mean_absolute_error(y_all, p_all)),
        "r2": float(r2_score(y_all, p_all)),
        "n_nodes": int(len(y_all)),
        "per_class": by_class,
        "per_tomo": per_tomo,
    }


def ridge_vs_gnn_per_tomo(
    model: ExposureGNN,
    graphs: List,
    device: torch.device,
) -> Dict[str, Dict]:
    """Compare GNN and Ridge per tomogram."""
    out = {}
    for g in graphs:
        mask = _node_mask(g)
        x = g.x.numpy()[mask]
        y = g.y_abs.numpy().ravel()[mask]
        # ridge on this tomo alone is trivial; compare gnn only here
        pred = predict_graph(model, g, device)
        out[g.tomo_id] = {
            "gnn_mae": float(mean_absolute_error(y, pred)),
            "gnn_r2": float(r2_score(y, pred)) if len(y) > 1 else 0.0,
            "mean_y": float(y.mean()),
        }
    return out
