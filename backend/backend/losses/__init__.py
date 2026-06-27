"""Loss functions for 3D multi-class segmentation (PyTorch, logits + integer targets).

Targets are expected to carry *contiguous model-class indices* in
``0..n_class-1`` (see ``labels.py``). Inactive classes (e.g. membrane when not
stamped) receive zero weight in the CE term and are excluded from the Tversky
sum so they don't inflate the loss or waste gradient.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _softmax_probs(logits: torch.Tensor) -> torch.Tensor:
    return F.softmax(logits, dim=1)


def _resolve_class_weights(
    class_weights, n_class: int, active_classes: Optional[List[int]]
) -> Optional[torch.Tensor]:
    """Combine explicit per-class weights with an active-class mask.

    Any class not in ``active_classes`` is forced to zero weight so inactive
    heads (e.g. membrane when not trained) don't pull the model.
    """
    if class_weights is None:
        if active_classes is None:
            return None
        w = torch.zeros(n_class, dtype=torch.float32)
        for c in active_classes:
            w[c] = 1.0
        return w
    w = torch.as_tensor(class_weights, dtype=torch.float32)
    if w.numel() != n_class:
        # broadcast or pad
        w = w.reshape(-1)
        if w.numel() < n_class:
            w = torch.cat([w, torch.ones(n_class - w.numel())])
    if active_classes is not None:
        mask = torch.zeros(n_class, dtype=torch.float32)
        for c in active_classes:
            mask[c] = 1.0
        w = w * mask
    return w


# ---------------------------------------------------------------------------
# Tversky
# ---------------------------------------------------------------------------
class TverskyLoss(nn.Module):
    """Multi-class Tversky loss summed over ACTIVE foreground classes only.

    Operates on logits (B,C,D,H,W) and integer targets (B,D,H,W).
    """

    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-3, ignore_index=-1,
                 active_classes: Optional[List[int]] = None):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.active_classes = active_classes  # classes to sum over (excl. bg)

    def forward(self, logits, target):
        n_class = logits.shape[1]
        probs = _softmax_probs(logits)
        valid = (target != self.ignore_index)
        t = target.clone()
        t[~valid] = 0
        oh = F.one_hot(t, num_classes=n_class).permute(0, 4, 1, 2, 3).float()
        mask = valid.unsqueeze(1).float()
        probs = probs * mask
        oh = oh * mask

        dims = (0, 2, 3, 4)
        tp = (probs * oh).sum(dims)
        fp = (probs * (1 - oh)).sum(dims)
        fn = ((1 - probs) * oh).sum(dims)
        ti = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)

        # Only sum over active foreground classes; ignore dead heads.
        if self.active_classes is not None:
            idx = torch.as_tensor(self.active_classes, device=ti.device, dtype=torch.long)
            # exclude background (class 0) from the loss sum
            fg = idx[idx != 0]
            n_active = int(fg.numel())
            return n_active - ti[fg].sum()
        # default: sum over all classes except background (index 0)
        fg_idx = list(range(1, n_class))
        return float(len(fg_idx)) - ti[fg_idx].sum()


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, gamma=2.0, smooth=1e-3, ignore_index=-1,
                 active_classes: Optional[List[int]] = None):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.active_classes = active_classes

    def forward(self, logits, target):
        n_class = logits.shape[1]
        probs = _softmax_probs(logits)
        valid = (target != self.ignore_index)
        t = target.clone()
        t[~valid] = 0
        oh = F.one_hot(t, num_classes=n_class).permute(0, 4, 1, 2, 3).float()
        mask = valid.unsqueeze(1).float()
        probs = probs * mask
        oh = oh * mask

        dims = (0, 2, 3, 4)
        tp = (probs * oh).sum(dims)
        fp = (probs * (1 - oh)).sum(dims)
        fn = ((1 - probs) * oh).sum(dims)
        ti = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        focal = torch.pow((1.0 - ti), self.gamma)

        if self.active_classes is not None:
            idx = torch.as_tensor(self.active_classes, device=focal.device, dtype=torch.long)
            fg = idx[idx != 0]
            return focal[fg].mean()
        fg_idx = list(range(1, n_class))
        return focal[fg_idx].mean()


# ---------------------------------------------------------------------------
# Dice
# ---------------------------------------------------------------------------
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-3, ignore_index=-1, active_classes: Optional[List[int]] = None):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.active_classes = active_classes

    def forward(self, logits, target):
        n_class = logits.shape[1]
        probs = _softmax_probs(logits)
        valid = (target != self.ignore_index)
        t = target.clone()
        t[~valid] = 0
        oh = F.one_hot(t, num_classes=n_class).permute(0, 4, 1, 2, 3).float()
        mask = valid.unsqueeze(1).float()
        probs = probs * mask
        oh = oh * mask
        dims = (0, 2, 3, 4)
        inter = (probs * oh).sum(dims)
        denom = (probs + oh).sum(dims)
        dice = (2 * inter + self.smooth) / (denom + self.smooth)
        if self.active_classes is not None:
            idx = torch.as_tensor(self.active_classes, device=dice.device, dtype=torch.long)
            fg = idx[idx != 0]
            return 1.0 - dice[fg].mean()
        fg_idx = list(range(1, n_class))
        return 1.0 - dice[fg_idx].mean()


# ---------------------------------------------------------------------------
# Combined CE + Tversky
# ---------------------------------------------------------------------------
class CETverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, ce_weight=1.0, tversky_weight=1.0,
                 smooth=1e-3, ignore_index=-1, class_weights=None,
                 active_classes: Optional[List[int]] = None,
                 n_class: int = 8):
        super().__init__()
        self.ce_weight = ce_weight
        self.tversky_weight = tversky_weight
        self.tversky = TverskyLoss(alpha, beta, smooth, ignore_index, active_classes)
        w = _resolve_class_weights(class_weights, n_class, active_classes)
        self.ce = nn.CrossEntropyLoss(
            weight=w,
            ignore_index=ignore_index,
        )
        self._weight_tensor = w

    def forward(self, logits, target):
        l_ce = self.ce(logits, target)
        l_tv = self.tversky(logits, target)
        return self.ce_weight * l_ce + self.tversky_weight * l_tv


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def build_loss(cfg, n_class: int = 8) -> nn.Module:
    """Build a loss from a LossCfg dataclass.

    ``n_class`` must match the model's output channel count so the CE class
    weight vector is sized correctly.
    """
    cw = cfg.class_weights
    weights = None
    if isinstance(cw, list):
        weights = [float(x) for x in cw]
    active = getattr(cfg, "active_classes", None)

    if cfg.type == "ce_tversky":
        return CETverskyLoss(
            alpha=cfg.alpha, beta=cfg.beta, ce_weight=cfg.ce_weight,
            tversky_weight=cfg.tversky_weight, smooth=cfg.smooth,
            ignore_index=cfg.ignore_index, class_weights=weights,
            active_classes=active, n_class=n_class,
        )
    if cfg.type == "tversky":
        return TverskyLoss(cfg.alpha, cfg.beta, cfg.smooth, cfg.ignore_index, active)
    if cfg.type == "focal_tversky":
        return FocalTverskyLoss(cfg.alpha, cfg.beta, cfg.gamma, cfg.smooth, cfg.ignore_index, active)
    if cfg.type == "dice":
        return DiceLoss(cfg.smooth, cfg.ignore_index, active)
    if cfg.type == "ce":
        w = _resolve_class_weights(weights, n_class, active)
        return nn.CrossEntropyLoss(weight=w, ignore_index=cfg.ignore_index)
    raise ValueError(f"Unknown loss type '{cfg.type}'")