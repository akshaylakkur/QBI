"""PyTorch-native CryoET 3D segmentation toolkit (czii-style challenge).

A clean reimplementation of the DeepFindET pipeline in PyTorch:
  1. build sphere-based segmentation targets from copick picks
  2. train a 3D residual U-Net with class-balanced bootstrap patch sampling
  3. sliding-window inference producing a per-class scoremap / labelmap
  4. (optional) connected-component localization to particle coordinates
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("backend")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.1.0"