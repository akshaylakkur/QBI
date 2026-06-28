"""Feature-only baseline (no message passing)."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler


def ridge_baseline_metrics(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> Tuple[float, float]:
    """Ridge regression on node features only. Returns (val_mae, val_r2)."""
    scaler = StandardScaler()
    x_tr = scaler.fit_transform(x_train)
    x_va = scaler.transform(x_val)
    model = Ridge(alpha=1.0)
    model.fit(x_tr, y_train.ravel())
    pred = model.predict(x_va)
    pred = np.clip(pred, 0.0, 1.0)
    return float(mean_absolute_error(y_val, pred)), float(r2_score(y_val, pred))
