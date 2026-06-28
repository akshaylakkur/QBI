"""Run trained GNN to predict steric exposure."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from downstream.gnn.features import EDGE_DIM, FeatureStats
from downstream.gnn.graph import build_graph_from_dataframe
from downstream.gnn.model import ExposureGNN


@torch.no_grad()
def predict_csv(checkpoint_path, csv_path, out_path) -> pd.DataFrame:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    stats = FeatureStats.from_dict(ckpt["feature_stats"])

    model = ExposureGNN(
        in_channels=ckpt["in_channels"],
        hidden_dim=cfg.get("hidden_dim", 64),
        out_channels=cfg.get("out_channels", 32),
        dropout=cfg.get("dropout", 0.25),
        heads=cfg.get("heads", 2),
        edge_dim=ckpt.get("edge_dim", EDGE_DIM),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    df = pd.read_csv(csv_path)
    if "steric_exposure" not in df.columns:
        df = df.copy()
        df["steric_exposure"] = 0.0

    chunks = []
    for tomo_id in sorted(df["tomo_id"].unique()):
        sub = df[df["tomo_id"] == tomo_id].copy().reset_index(drop=True)
        g = build_graph_from_dataframe(
            sub, tomo_id,
            edge_cutoff=cfg.get("edge_cutoff", 500.0),
            knn_k=cfg.get("knn_k", 12),
            feature_stats=stats,
        )
        pred = model(g.x, g.edge_index, g.edge_attr).numpy().ravel()
        sub["gnn_steric_exposure"] = pred
        chunks.append(sub)

    out = pd.concat(chunks, ignore_index=True)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="/Users/kjaladi/Desktop/QBI/runs/exposure_gnn.pt")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    predict_csv(args.checkpoint, args.csv, args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
