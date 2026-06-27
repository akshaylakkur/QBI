"""Reproducibility helpers."""

import os
import random
import numpy as np


def seed_everything(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # torch not installed (e.g. for utils-only use)
        pass


def worker_init_fn(worker_id: int) -> None:
    """Per-worker numpy/random seed for DataLoader workers."""
    import numpy as np
    import random

    seed = (np.random.get_state()[1][0] + worker_id) % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)