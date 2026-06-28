"""Label <-> model-class mapping for the CZII CryoET challenge.

copick pickable objects use *non-contiguous* integer labels:

    0  background (implicit)
    1  apo-ferritin
    2  beta-amylase
    3  beta-galactosidase
    4  ribosome
    5  thyroglobulin
    6  virus-like-particle
    8  membrane
    9  background (named object)

The model emits a contiguous softmax over `n_class` channels `0..n_class-1`.
To avoid a dead channel and an out-of-range one-hot crash, we remap copick
labels to contiguous model-class indices via ``LABEL_TO_CLASS`` and back via
``CLASS_TO_LABEL``.

With ``n_class = 8`` (the default), the mapping is:

    copick label 0 (background)      -> class 0
    copick label 1 (apo-ferritin)    -> class 1
    copick label 2 (beta-amylase)    -> class 2
    copick label 3 (beta-galactos.)  -> class 3
    copick label 4 (ribosome)        -> class 4
    copick label 5 (thyroglobulin)   -> class 5
    copick label 6 (virus-like-part) -> class 6
    copick label 8 (membrane)        -> class 7

If membrane is not used, class 7 has no target signal; ``ACTIVE_CLASSES``
lists the classes that actually appear in targets and should participate in
loss weighting / metric aggregation / model selection.
"""

from __future__ import annotations

from typing import Dict, List

# copick label -> contiguous model class index
LABEL_TO_CLASS: Dict[int, int] = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    8: 7,
}

# contiguous model class index -> copick label
CLASS_TO_LABEL: Dict[int, int] = {v: k for k, v in LABEL_TO_CLASS.items()}

# Particle (foreground) copick labels, in scored order.
PARTICLE_LABELS: List[int] = [1, 2, 3, 4, 5, 6]

# Particle (foreground) model class indices.
PARTICLE_CLASSES: List[int] = [LABEL_TO_CLASS[l] for l in PARTICLE_LABELS]

# Classes that actually carry target signal when membrane is NOT stamped.
# (background + the six particles). Class 7 (membrane) is excluded until
# membrane targets are explicitly built.
ACTIVE_CLASSES: List[int] = [0] + PARTICLE_CLASSES

# Name table for logging / localization.
CLASS_TO_NAME: Dict[int, str] = {
    0: "background",
    1: "apo-ferritin",
    2: "beta-amylase",
    3: "beta-galactosidase",
    4: "ribosome",
    5: "thyroglobulin",
    6: "virus-like-particle",
    7: "membrane",
}


def remap_volume(vol, mapping: Dict[int, int] = None) -> "vol":
    """Remap an integer label volume in-place-safe using a lookup table.

    Values not present in ``mapping`` are mapped to 0 (treated as background).
    Works on numpy uint8 arrays and torch integer tensors.
    """
    import numpy as np

    m = mapping or LABEL_TO_CLASS
    src_dtype = vol.dtype if hasattr(vol, "dtype") else None
    # build a LUT large enough to cover any value present in the input
    max_key = max(max(m.keys()), int(vol.max()) if hasattr(vol, "max") else 0)
    table = np.zeros(max_key + 1, dtype=np.int64)
    for k, v in m.items():
        if k <= max_key:
            table[k] = v
    if isinstance(vol, np.ndarray):
        return table[vol.astype(np.int64)].astype(src_dtype)
    # torch tensor
    import torch

    t = torch.from_numpy(table).to(vol.device).to(vol.dtype)
    mask = (vol >= 0) & (vol <= max_key)
    out = torch.zeros_like(vol)
    out[mask] = t[vol[mask].long()]
    return out


def inverse_remap_volume(vol) -> "vol":
    """Map model class indices back to copick labels."""
    import numpy as np

    inv = {v: k for k, v in LABEL_TO_CLASS.items()}
    return remap_volume(vol, inv)