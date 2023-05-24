"""Tests for synthetic_kw_helpers script."""
import numpy as np
import pytest

from adcraft import rust
from adcraft.synthetic_kw_helpers import generic_cost


RNG = np.random.default_rng(40)


@pytest.mark.unit
@pytest.mark.parametrize(
    "x",
    [
        (RNG.uniform(0.01, 0.9, 100000)),
    ],
)
def test_cost_mut(x: np.array) -> None:
    rng = np.random.default_rng(40)
    generic_result = generic_cost(x, rng)
    generic_mean = generic_result.mean()
    generic_std = generic_result.std()
    rust.cost_mut(x)
    rust_mean = x.mean()
    rust_std = x.std()

    assert round(rust_mean, 2) == round(generic_mean, 2)
    assert round(rust_std, 2) == round(generic_std, 2)


@pytest.mark.unit
@pytest.mark.parametrize(
    "x",
    [
        (RNG.uniform(0.01, 0.9, 100000)),
    ],
)
def test_cost_trans(x: np.array) -> None:
    rng = np.random.default_rng(40)
    generic_result = generic_cost(x, rng)
    generic_mean = generic_result.mean()
    generic_std = generic_result.std()
    xr = rust.cost_trans(x)
    rust_mean = xr.mean()
    rust_std = xr.std()

    assert round(rust_mean, 2) == round(generic_mean, 2)
    assert round(rust_std, 2) == round(generic_std, 2)
