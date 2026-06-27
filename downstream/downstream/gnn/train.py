"""Training loop for steric exposure GNN surrogate."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, r2_score
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from .baseline import ridge_baseline_metrics
from .graph import graphs_from_exposure_csv, split_graphs_by_tomo
from .model import ExposureGNN


@dataclass
class TrainConfig:
    hidden_dim: int = 128
    out_channels: int = 64
    dropout: float = 0.15
    heads: int = 2
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 300
    batch_size: int = 4
    patience: int = 40
    edge_cutoff: float = 500.0
    density_radius: float = 300.0
    val_tomo_ids: Tuple[str, ...] = ("TS_73_6", "TS_99_9")
    seed: int = 42


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


@torch.no_grad()
def evaluate(model: ExposureGNN, graphs: List[Data], device: torch.device) -> Dict[str, float]:
    model.eval()
    ys, preds = [], []
    for g in graphs:
        g = g.to(device)
        pred = model(g.x, g.edge_index, g.edge_attr).cpu().numpy().ravel()
        y = g.y.cpu().numpy().ravel()
        ys.append(y)
        preds.append(pred)
    y_all = np.concatenate(ys)
    p_all = np.concatenate(preds)
    return {
        "mae": float(mean_absolute_error(y_all, p_all)),
        "r2": float(r2_score(y_all, p_all)),
        "n_nodes": int(len(y_all)),
    }


def train_exposure_gnn(
    csv_path: str | Path,
    out_path: str | Path,
    config: Optional[TrainConfig] = None,
    val_tomo_ids: Optional[Sequence[str]] = None,
) -> Dict:
    """Train GNN and save checkpoint. Returns metrics dict."""
    config = config or TrainConfig()
    if val_tomo_ids is not None:
        config.val_tomo_ids = tuple(val_tomo_ids)

    _set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    graphs = graphs_from_exposure_csv(
        csv_path,
        edge_cutoff=config.edge_cutoff,
        density_radius=config.density_radius,
    )
    train_graphs, val_graphs = split_graphs_by_tomo(graphs, config.val_tomo_ids)

    in_channels = train_graphs[0].x.shape[1]
    model = ExposureGNN(
        in_channels=in_channels,
        hidden_dim=config.hidden_dim,
        out_channels=config.out_channels,
        dropout=config.dropout,
        heads=config.heads,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    train_loader = DataLoader(train_graphs, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=1, shuffle=False)

    # Feature-only baseline on concatenated nodes
    x_tr = torch.cat([g.x for g in train_graphs], dim=0).numpy()
    y_tr = torch.cat([g.y for g in train_graphs], dim=0).numpy()
    x_va = torch.cat([g.x for g in val_graphs], dim=0).numpy()
    y_va = torch.cat([g.y for g in val_graphs], dim=0).numpy()
    baseline_mae, baseline_r2 = ridge_baseline_metrics(x_tr, y_tr, x_va, y_va)

    best_val_mae = float("inf")
    best_state = None
    stale = 0
    history = []

    for epoch in range(config.epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = F.mse_loss(pred, batch.y)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        train_metrics = evaluate(model, train_graphs, device)
        val_metrics = evaluate(model, val_graphs, device)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "train_mae": train_metrics["mae"],
            "train_r2": train_metrics["r2"],
            "val_mae": val_metrics["mae"],
            "val_r2": val_metrics["r2"],
        }
        history.append(row)

        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= config.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    final_train = evaluate(model, train_graphs, device)
    final_val = evaluate(model, val_graphs, device)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "in_channels": in_channels,
        "config": asdict(config),
        "train_tomo_ids": [g.tomo_id for g in train_graphs],
        "val_tomo_ids": list(config.val_tomo_ids),
    }
    torch.save(checkpoint, out_path)

    metrics = {
        "device": str(device),
        "train_tomograms": [g.tomo_id for g in train_graphs],
        "val_tomograms": list(config.val_tomo_ids),
        "train_nodes": final_train["n_nodes"],
        "val_nodes": final_val["n_nodes"],
        "baseline_val_mae": baseline_mae,
        "baseline_val_r2": baseline_r2,
        "gnn_train_mae": final_train["mae"],
        "gnn_train_r2": final_train["r2"],
        "gnn_val_mae": final_val["mae"],
        "gnn_val_r2": final_val["r2"],
        "best_val_mae": best_val_mae,
        "epochs_ran": len(history),
    }
    metrics_path = out_path.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2))
    history_path = out_path.with_suffix(".history.json")
    history_path.write_text(json.dumps(history, indent=2))
    return metrics
