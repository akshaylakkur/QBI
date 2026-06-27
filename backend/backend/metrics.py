"""Segmentation metrics (per-class precision/recall/F1, macro-F1, mIoU).

Macro averages are computed over *active* classes only (background + scored
particles), excluding dead heads (e.g. membrane when not trained), so the
checkpoint-selection metric reflects real task performance rather than being
dragged down by phantom channels.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch


@torch.no_grad()
def segmentation_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
    n_class: int,
    ignore_index: int = -1,
    active_classes: Optional[List[int]] = None,
) -> dict:
    """Compute per-class precision/recall/F1 and macro averages from a batch.

    logits: (B,C,D,H,W)  target: (B,D,H,W) integer class indices
    active_classes: classes to include in the *macro* average. If None, all
        classes 0..n_class-1 are used. Per-class arrays always have length
        n_class; only the macro is restricted.
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

    classes = active_classes if active_classes is not None else list(range(n_class))
    cls_idx = torch.as_tensor(classes, device=f1.device, dtype=torch.long)
    macro_f1 = float(f1[cls_idx].mean().item())
    macro_iou = float(iou[cls_idx].mean().item())

    return {
        "precision": precision.cpu().numpy(),
        "recall": recall.cpu().numpy(),
        "f1": f1.cpu().numpy(),
        "iou": iou.cpu().numpy(),
        "macro_f1": macro_f1,
        "macro_iou": macro_iou,
    }


class MetricAccumulator:
    """Running mean of segmentation metrics across batches."""

    def __init__(self, n_class: int, active_classes: Optional[List[int]] = None):
        self.n_class = n_class
        self.active_classes = active_classes
        self.reset()

    def reset(self):
        self._f1 = []
        self._iou = []
        self._prec = []
        self._rec = []
        self._loss = []

    def update(self, metrics: dict, loss: float):
        self._f1.append(metrics["f1"])
        self._iou.append(metrics["iou"])
        self._prec.append(metrics["precision"])
        self._rec.append(metrics["recall"])
        self._loss.append(loss)

    def compute(self) -> dict:
        if not self._f1:
            return {"macro_f1": 0.0, "macro_iou": 0.0, "loss": 0.0,
                    "f1": np.zeros(self.n_class), "iou": np.zeros(self.n_class),
                    "precision": np.zeros(self.n_class), "recall": np.zeros(self.n_class)}
        f1 = np.stack(self._f1).mean(0)
        iou = np.stack(self._iou).mean(0)
        prec = np.stack(self._prec).mean(0)
        rec = np.stack(self._rec).mean(0)
        classes = self.active_classes if self.active_classes is not None else list(range(self.n_class))
        cls_idx = np.asarray(classes, dtype=np.int64)
        return {
            "f1": f1,
            "iou": iou,
            "precision": prec,
            "recall": rec,
            "macro_f1": float(f1[cls_idx].mean()),
            "macro_iou": float(iou[cls_idx].mean()),
            "loss": float(np.mean(self._loss)),
        }