"""Tests for Fibonacci sphere directions."""

import numpy as np
import pytest

from downstream.geometry.directions import fibonacci_directions


def test_fibonacci_directions_unit_length():
    dirs = fibonacci_directions(500)
    norms = np.linalg.norm(dirs, axis=1)
    np.testing.assert_allclose(norms, 1.0, rtol=1e-10, atol=1e-10)


def test_fibonacci_directions_reproducible():
    a = fibonacci_directions(200)
    b = fibonacci_directions(200)
    np.testing.assert_array_equal(a, b)


def test_fibonacci_directions_invalid_n():
    with pytest.raises(ValueError):
        fibonacci_directions(0)
