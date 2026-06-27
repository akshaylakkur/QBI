"""Train GATv2 surrogate for steric exposure.

Usage:
  python -m downstream.scripts.train_exposure_gnn \
    --labels /Users/kjaladi/Desktop/QBI/runs/exposure_labels_2000rays.csv \
    --out /Users/kjaladi/Desktop/QBI/runs/exposure_gnn.pt
"""

from __future__ import annotations

import argparse
import json
from typing import Optional

from downstream.gnn.train import TrainConfig, train_exposure_gnn


def main(argv: Optional[list] = None) -> None:
    ap = argparse.ArgumentParser(description="Train steric exposure GNN surrogate")
    ap.add_argument(
        "--labels",
        default="/Users/kjaladi/Desktop/QBI/runs/exposure_labels_2000rays.csv",
        help="Exposure labels CSV from compute_exposure",
    )
    ap.add_argument(
        "--out",
        default="/Users/kjaladi/Desktop/QBI/runs/exposure_gnn.pt",
        help="Checkpoint output path",
    )
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--edge-cutoff", type=float, default=500.0)
    ap.add_argument(
        "--val-tomo",
        action="append",
        default=None,
        help="Validation tomogram id (default: TS_73_6 TS_99_9)",
    )
    args = ap.parse_args(argv)

    val_tomos = tuple(args.val_tomo) if args.val_tomo else ("TS_73_6", "TS_99_9")
    config = TrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        edge_cutoff=args.edge_cutoff,
        val_tomo_ids=val_tomos,
    )

    metrics = train_exposure_gnn(args.labels, args.out, config=config)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
