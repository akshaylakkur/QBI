"""Run trained GNN to predict steric exposure on a coordinate CSV."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
import torch

from downstream.gnn.graph import build_graph_from_dataframe
from downstream.gnn.model import ExposureGNN


@torch.no_grad()
def predict_csv(
    checkpoint_path: str | Path,
    labels_or_coords_csv: str | Path,
    out_path: str | Path,
) -> pd.DataFrame:
    """Predict steric exposure per row; writes CSV with gnn_steric_exposure column."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    in_channels = ckpt["in_channels"]
    cfg = ckpt.get("config", {})
    edge_cutoff = cfg.get("edge_cutoff", 500.0)
    density_radius = cfg.get("density_radius", 300.0)

    model = ExposureGNN(
        in_channels=in_channels,
        hidden_dim=cfg.get("hidden_dim", 128),
        out_channels=cfg.get("out_channels", 64),
        dropout=cfg.get("dropout", 0.15),
        heads=cfg.get("heads", 2),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    df = pd.read_csv(labels_or_coords_csv)
    chunks = []
    for tomo_id in sorted(df["tomo_id"].unique()):
        sub = df[df["tomo_id"] == tomo_id].copy().reset_index(drop=True)
        if "steric_exposure" not in sub.columns:
            sub["steric_exposure"] = 0.0
        graph = build_graph_from_dataframe(
            sub,
            tomo_id,
            edge_cutoff=edge_cutoff,
            density_radius=density_radius,
        )
        pred = model(graph.x, graph.edge_index, graph.edge_attr).numpy().ravel()
        sub["gnn_steric_exposure"] = pred
        chunks.append(sub)

    out = pd.concat(chunks, ignore_index=True)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return out


def main(argv: Optional[list] = None) -> None:
    ap = argparse.ArgumentParser(description="Predict steric exposure with trained GNN")
    ap.add_argument("--checkpoint", default="/Users/kjaladi/Desktop/QBI/runs/exposure_gnn.pt")
    ap.add_argument("--csv", required=True, help="Input CSV (exposure labels or coordinates)")
    ap.add_argument("--out", required=True, help="Output CSV with gnn_steric_exposure")
    args = ap.parse_args(argv)

    predict_csv(args.checkpoint, args.csv, args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
