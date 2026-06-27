"""Segmentation metrics (per-class precision/recall/F1, macro-F1, mIoU)."""

from __future__ import annotations

import torch


@torch.no_grad()
def segmentation_metrics(
    logits: torch.Tensor, target: torch.Tensor, n_class: int, ignore_index: int = -1
) -> dict:
    """Compute per-class precision/recall/F1 and macro averages from a batch.

    logits: (B,C,D,H,W)  target: (B,D,H,W) integer class indices
    """
    pred = logits.argmax(1)
    valid = target != ignore_index
    pred = pred[valid]
    tgt = target[valid]

    tp = torch.zeros(n_class, dtype=torch.long, device=logits.device)
    fp = torch.zeros(n_class, dtype=torch.long, device=logits.device)
    fn = torch.zeros(n_class, dtype=torch.long, device=logits.device)

    for c in range(n_class):
        tp[c] = ((pred == c) & (tgt == c)).sum()
        fp[c] = ((pred == c) & (tgt != c)).sum()
        fn[c] = ((pred != c) & (tgt == c)).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)

    return {
        "precision": precision.cpu().numpy(),
        "recall": recall.cpu().numpy(),
        "f1": f1.cpu().numpy(),
        "iou": iou.cpu().numpy(),
        "macro_f1": float(f1.mean().item()),
        "macro_iou": float(iou.mean().item()),
    }


class MetricAccumulator:
    """Running mean of segmentation metrics across batches."""

    def __init__(self, n_class: int):
        self.n_class = n_class
        self.reset()

    def reset(self):
        import numpy as np

        self._f1 = []
        self._iou = []
        self._prec = []
        self._rec = []
        self._loss = []

    def update(self, metrics: dict, loss: float):
        import numpy as np

        self._f1.append(metrics["f1"])
        self._iou.append(metrics["iou"])
        self._prec.append(metrics["precision"])
        self._rec.append(metrics["recall"])
        self._loss.append(loss)

    def compute(self) -> dict:
        import numpy as np

        if not self._f1:
            return {"macro_f1": 0.0, "macro_iou": 0.0, "loss": 0.0}
        f1 = np.stack(self._f1).mean(0)
        iou = np.stack(self._iou).mean(0)
        prec = np.stack(self._prec).mean(0)
        rec = np.stack(self._rec).mean(0)
        return {
            "f1": f1,
            "iou": iou,
            "precision": prec,
            "recall": rec,
            "macro_f1": float(f1.mean()),
            "macro_iou": float(iou.mean()),
            "loss": float(np.mean(self._loss)),
        }