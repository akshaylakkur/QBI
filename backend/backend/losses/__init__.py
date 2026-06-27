"""Loss functions for 3D multi-class segmentation (PyTorch, logits + integer targets)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _softmax_probs(logits: torch.Tensor) -> torch.Tensor:
    return F.softmax(logits, dim=1)


# ---------------------------------------------------------------------------
# Tversky
# ---------------------------------------------------------------------------
class TverskyLoss(nn.Module):
    """Multi-class Tversky loss (summed over foreground classes).

    Operates on logits (B,C,D,H,W) and integer targets (B,D,H,W).
    """

    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-3, ignore_index=-1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        n_class = logits.shape[1]
        probs = _softmax_probs(logits)
        # one-hot the target (ignore invalid)
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
        # sum over classes -> loss = n_class - sum(ti)
        return torch.tensor(float(n_class), device=logits.device, dtype=logits.dtype) - ti.sum()


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, gamma=2.0, smooth=1e-3, ignore_index=-1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        self.ignore_index = ignore_index

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
        return focal.mean()


# ---------------------------------------------------------------------------
# Dice
# ---------------------------------------------------------------------------
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-3, ignore_index=-1):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

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
        return 1.0 - dice.mean()


# ---------------------------------------------------------------------------
# Combined CE + Tversky
# ---------------------------------------------------------------------------
class CETverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, ce_weight=1.0, tversky_weight=1.0,
                 smooth=1e-3, ignore_index=-1, class_weights=None):
        super().__init__()
        self.ce_weight = ce_weight
        self.tversky_weight = tversky_weight
        self.tversky = TverskyLoss(alpha, beta, smooth, ignore_index)
        self.ce = nn.CrossEntropyLoss(
            weight=None if class_weights is None else torch.as_tensor(class_weights, dtype=torch.float32),
            ignore_index=ignore_index,
        )

    def forward(self, logits, target):
        l_ce = self.ce(logits, target)
        l_tv = self.tversky(logits, target)
        return self.ce_weight * l_ce + self.tversky_weight * l_tv


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def build_loss(cfg) -> nn.Module:
    """Build a loss from a LossCfg dataclass."""
    cw = cfg.class_weights
    weights = None
    if isinstance(cw, list):
        weights = cw

    if cfg.type == "ce_tversky":
        return CETverskyLoss(
            alpha=cfg.alpha, beta=cfg.beta, ce_weight=cfg.ce_weight,
            tversky_weight=cfg.tversky_weight, smooth=cfg.smooth,
            ignore_index=cfg.ignore_index, class_weights=weights,
        )
    if cfg.type == "tversky":
        return TverskyLoss(cfg.alpha, cfg.beta, cfg.smooth, cfg.ignore_index)
    if cfg.type == "focal_tversky":
        return FocalTverskyLoss(cfg.alpha, cfg.beta, cfg.gamma, cfg.smooth, cfg.ignore_index)
    if cfg.type == "dice":
        return DiceLoss(cfg.smooth, cfg.ignore_index)
    if cfg.type == "ce":
        return nn.CrossEntropyLoss(
            weight=None if weights is None else torch.as_tensor(weights, dtype=torch.float32),
            ignore_index=cfg.ignore_index,
        )
    raise ValueError(f"Unknown loss type '{cfg.type}'")