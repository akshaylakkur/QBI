"""
Data loading utilities for CryoET inference.
Loads tomogram volumes from the CZII CryoET dataset (Zarr format).
"""

import os
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import zarr


# Constants matching the Kaggle competition
ANGSTROMS_IN_PIXEL = 10.012

# 6-class target definitions (label, name, radius in Angstroms)
TARGET_CLASSES = (
    {"name": "apo-ferritin", "label": 0, "radius": 60},
    {"name": "beta-galactosidase", "label": 1, "radius": 90},
    {"name": "ribosome", "label": 2, "radius": 150},
    {"name": "thyroglobulin", "label": 3, "radius": 130},
    {"name": "virus-like-particle", "label": 4, "radius": 135},
    {"name": "beta-amylase", "label": 5, "radius": 65},
)

CLASS_LABEL_TO_NAME = {c["label"]: c["name"] for c in TARGET_CLASSES}
CLASS_NAME_TO_LABEL = {c["name"]: c["label"] for c in TARGET_CLASSES}
TARGET_SIGMAS = [c["radius"] / ANGSTROMS_IN_PIXEL for c in TARGET_CLASSES]


def normalize_volume_to_unit_range(volume: np.ndarray) -> np.ndarray:
    """Normalize volume to [0, 1] range."""
    vol_min = volume.min()
    vol_max = volume.max()
    if vol_max - vol_min > 0:
        volume = (volume - vol_min) / (vol_max - vol_min)
    else:
        volume = np.zeros_like(volume)
    return volume.astype(np.float32)


def load_volume(
    data_root: Union[str, Path],
    study_name: str,
    split: str = "test",
    voxel_spacing: str = "VoxelSpacing10.000",
    tomo_type: str = "denoised",
) -> np.ndarray:
    """
    Load a tomogram volume from the CZII CryoET dataset.

    Args:
        data_root: Root directory of the dataset.
        study_name: e.g. "TS_5_4".
        split: "train" or "test".
        voxel_spacing: e.g. "VoxelSpacing10.000".
        tomo_type: e.g. "denoised", "isonetcorrected", "wbp".

    Returns:
        3D numpy array of shape (Z, Y, X).
    """
    zarr_path = os.path.join(
        str(data_root),
        split,
        "static",
        "ExperimentRuns",
        study_name,
        voxel_spacing,
        f"{tomo_type}.zarr",
    )

    store = zarr.DirectoryStore(zarr_path)
    zgroup = zarr.open(store, mode="r")

    # The data is at the highest resolution (path '0')
    volume = zgroup["0"][:]
    return np.asarray(volume, dtype=np.float32)


def compute_better_tiles_1d(length: int, window_size: int, num_tiles: int):
    """
    Compute slices for a sliding window over one dimension.
    Distributes tiles evenly so first tile starts at 0 and last tile ends at length.
    """
    last_tile_start = length - window_size
    starts = np.linspace(0, last_tile_start, num_tiles, dtype=int)
    ends = starts + window_size
    for start, end in zip(starts, ends):
        yield slice(start, end)


def compute_tiles(
    volume_shape: Tuple[int, int, int],
    window_size: Tuple[int, int, int],
    tiles_per_dim: Tuple[int, int, int],
):
    """
    Compute tile slices for a volume.

    Args:
        volume_shape: (Z, Y, X)
        window_size: (D, H, W) tile size
        tiles_per_dim: (nz, ny, nx) number of tiles per dimension

    Yields:
        (z_slice, y_slice, x_slice) tuples
    """
    wz, wy, wx = window_size
    nz, ny, nx = tiles_per_dim
    z, y, x = volume_shape

    for z_slice in compute_better_tiles_1d(z, wz, nz):
        for y_slice in compute_better_tiles_1d(y, wy, ny):
            for x_slice in compute_better_tiles_1d(x, wx, nx):
                yield (z_slice, y_slice, x_slice)
