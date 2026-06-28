"""Dataset splitting helpers."""

from __future__ import annotations

import random
from typing import List, Tuple


def split_runs(
    tomo_ids: List[str],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    n = len(tomo_ids)
    rng = random.Random(seed)
    ids = list(tomo_ids)
    rng.shuffle(ids)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    train = ids[:n_train]
    val = ids[n_train : n_train + n_val]
    test = ids[n_train + n_val :]
    return train, val, test