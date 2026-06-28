"""Downstream geometry metrics for Cryo-ET particle coordinate clouds."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("downstream")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.1.0"
