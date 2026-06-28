"""Training: synthetic pretrain, LOTO CV on real tomograms."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from .baseline import ridge_baseline_metrics
from .features import EDGE_DIM, FeatureStats
from .graph import fit_feature_stats_on_graphs, graphs_from_exposure_csv
from .model import ExposureGNN
from .synthetic import clouds_to_graphs, generate_synthetic_corpus


@dataclass
class TrainConfig:
    hidden_dim: int = 64
    out_channels: int = 32
    dropout: float = 0.25
    heads: int = 2
    lr: float = 1e-3
    finetune_lr: float = 2e-4
    weight_decay: float = 1e-4
    pretrain_epochs: int = 60
    finetune_epochs: int = 120
    batch_size: int = 16
    patience: int = 20
    edge_cutoff: float = 500.0
    knn_k: int = 12
    seed: int = 42
    synthetic_nodes: int = 50_000
    synthetic_n_rays: int = 2000


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


@torch.no_grad()
def _eval_mae_r2(model, graphs, device) -> Tuple[float, float]:
    from sklearn.metrics import mean_absolute_error, r2_score

    model.eval()
    ys, ps = [], []
    for g in graphs:
        g = g.to(device)
        pred = model(g.x, g.edge_index, g.edge_attr).cpu().numpy().ravel()
        ys.append(g.y.numpy().ravel())
        ps.append(pred)
    y, p = np.concatenate(ys), np.concatenate(ps)
    return float(mean_absolute_error(y, p)), float(r2_score(y, p))


def _train_epochs(
    model: ExposureGNN,
    train_graphs: List[Data],
    val_graphs: List[Data],
    device: torch.device,
    epochs: int,
    lr: float,
    batch_size: int,
    patience: int,
    weight_decay: float,
) -> Tuple[float, float, float, float]:
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_mae, best_state, stale = float("inf"), None, 0

    for _ in range(epochs):
        model.train()
        for batch in DataLoader(train_graphs, batch_size=batch_size, shuffle=True):
            batch = batch.to(device)
            opt.zero_grad()
            F.mse_loss(model(batch.x, batch.edge_index, batch.edge_attr), batch.y).backward()
            opt.step()

        val_mae, _ = _eval_mae_r2(model, val_graphs, device)
        if val_mae < best_mae:
            best_mae = val_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    tr_mae, tr_r2 = _eval_mae_r2(model, train_graphs, device)
    va_mae, va_r2 = _eval_mae_r2(model, val_graphs, device)
    return tr_mae, tr_r2, va_mae, va_r2


def pretrain_on_graphs(
    model: ExposureGNN,
    graphs: List[Data],
    device: torch.device,
    config: TrainConfig,
) -> Dict:
    n_val = max(1, len(graphs) // 10)
    train_g, val_g = graphs[n_val:], graphs[:n_val]
    tr_mae, _, va_mae, va_r2 = _train_epochs(
        model, train_g, val_g, device,
        config.pretrain_epochs, config.lr, config.batch_size,
        config.patience, config.weight_decay,
    )
    return {"graphs": len(graphs), "pretrain_val_mae": va_mae, "pretrain_val_r2": va_r2}


def leave_one_tomo_out(
    csv_path: str | Path,
    model: ExposureGNN,
    device: torch.device,
    config: TrainConfig,
    feature_stats: FeatureStats,
) -> Dict:
    all_ids = sorted({g.tomo_id for g in graphs_from_exposure_csv(csv_path)})
    base_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    folds = []

    for holdout in all_ids:
        model.load_state_dict(base_state)
        train_g = graphs_from_exposure_csv(
            csv_path, [t for t in all_ids if t != holdout],
            config.edge_cutoff, config.knn_k, feature_stats,
        )
        val_g = graphs_from_exposure_csv(
            csv_path, [holdout], config.edge_cutoff, config.knn_k, feature_stats,
        )
        _, _, va_mae, va_r2 = _train_epochs(
            model, train_g, val_g, device,
            config.finetune_epochs, config.finetune_lr, config.batch_size,
            config.patience, config.weight_decay,
        )
        x_tr = np.concatenate([g.x.numpy() for g in train_g])
        y_tr = np.concatenate([g.y.numpy().ravel() for g in train_g])
        x_va = np.concatenate([g.x.numpy() for g in val_g])
        y_va = np.concatenate([g.y.numpy().ravel() for g in val_g])
        ridge_mae, ridge_r2 = ridge_baseline_metrics(x_tr, y_tr, x_va, y_va)
        folds.append({
            "holdout": holdout,
            "gnn_mae": va_mae,
            "gnn_r2": va_r2,
            "ridge_mae": ridge_mae,
            "ridge_r2": ridge_r2,
            "gnn_beats_ridge": va_mae < ridge_mae,
        })

    gnn_maes = [f["gnn_mae"] for f in folds]
    ridge_maes = [f["ridge_mae"] for f in folds]
    return {
        "folds": folds,
        "loto_gnn_mae_mean": float(np.mean(gnn_maes)),
        "loto_gnn_mae_std": float(np.std(gnn_maes)),
        "loto_ridge_mae_mean": float(np.mean(ridge_maes)),
        "loto_ridge_mae_std": float(np.std(ridge_maes)),
        "gnn_wins": int(sum(f["gnn_beats_ridge"] for f in folds)),
    }


def run_pipeline(csv_path: str | Path, out_path: str | Path, config: Optional[TrainConfig] = None) -> Dict:
    config = config or TrainConfig()
    _set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"=== [1/3] Synthetic data: {config.synthetic_nodes:,} nodes ===")
    t0 = time.time()
    clouds, exposures = generate_synthetic_corpus(
        target_nodes=config.synthetic_nodes,
        n_rays=config.synthetic_n_rays,
        seed=config.seed,
    )
    synth_graphs_raw = clouds_to_graphs(clouds, exposures, None, config.edge_cutoff, config.knn_k)
    feature_stats = fit_feature_stats_on_graphs(synth_graphs_raw)
    synth_graphs = clouds_to_graphs(
        clouds, exposures, feature_stats, config.edge_cutoff, config.knn_k,
    )
    synth_seconds = round(time.time() - t0, 1)
    print(f"  Generated {len(clouds)} graphs, {sum(len(c.coords) for c in clouds):,} nodes in {synth_seconds}s")

    in_channels = synth_graphs[0].x.shape[1]
    model = ExposureGNN(
        in_channels=in_channels,
        hidden_dim=config.hidden_dim,
        out_channels=config.out_channels,
        dropout=config.dropout,
        heads=config.heads,
        edge_dim=EDGE_DIM,
    ).to(device)

    print("=== [2/3] Pretrain GNN on synthetic graphs ===")
    pretrain_info = pretrain_on_graphs(model, synth_graphs, device, config)
    pretrain_info["synthetic_seconds"] = synth_seconds
    pretrain_info["synthetic_nodes"] = sum(len(c.coords) for c in clouds)

    print("=== [3/3] LOTO CV on 7 real tomograms ===")
    loto = leave_one_tomo_out(csv_path, model, device, config, feature_stats)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "in_channels": in_channels,
        "edge_dim": EDGE_DIM,
        "config": asdict(config),
        "feature_stats": feature_stats.to_dict(),
    }, out_path)

    metrics = {"device": str(device), "pretrain": pretrain_info, "loto": loto}
    out_path.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    return metrics
