"""Train steric exposure GNN: synthetic pretrain + LOTO CV."""

from __future__ import annotations

import argparse

from downstream.gnn.train import TrainConfig, run_pipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="/Users/kjaladi/Desktop/QBI/runs/exposure_labels_2000rays.csv")
    ap.add_argument("--out", default="/Users/kjaladi/Desktop/QBI/runs/exposure_gnn.pt")
    ap.add_argument("--synthetic-nodes", type=int, default=50_000)
    ap.add_argument("--synthetic-rays", type=int, default=2000)
    ap.add_argument("--pretrain-epochs", type=int, default=60)
    ap.add_argument("--finetune-epochs", type=int, default=120)
    args = ap.parse_args()

    config = TrainConfig(
        synthetic_nodes=args.synthetic_nodes,
        synthetic_n_rays=args.synthetic_rays,
        pretrain_epochs=args.pretrain_epochs,
        finetune_epochs=args.finetune_epochs,
    )
    run_pipeline(args.labels, args.out, config)


if __name__ == "__main__":
    main()
