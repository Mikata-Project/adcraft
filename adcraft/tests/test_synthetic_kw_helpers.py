"""Tests for synthetic_kw_helpers script."""
from typing import Union

import numpy as np
import pytest

from adcraft import rust
from adcraft import synthetic_kw_classes as skc


@pytest.mark.unit
@pytest.mark.parametrize(
    "x,expected_result",
    [
        (0, 0),
        (1, 1),
        (2, 1),
        (-1, 0),
        (0.5, 0.5),
        (np.array([0, 1, 2, -1, 0.5]), np.array([0, 1, 1, 0, 0.5])),
    ],
)
def test_probify(x: Union[int, float, np.array], expected_result: float) -> None:
    actual = skc.probify(x)
    if isinstance(x, np.ndarray):
        assert np.array_equal(actual, expected_result)
    else:
        assert actual == expected_result


@pytest.mark.unit
@pytest.mark.parametrize(
    "x,expected_result",
    [
        (0, 0),
        (1, 1),
        (2, 2),
        (-1, 0),
        (0.5, 0.5),
        (np.array([0, 1, 2, -1, 0.5]), np.array([0, 1, 2, 0, 0.5])),
    ],
)
def test_nonnegify(x: Union[int, float, np.array], expected_result: float) -> None:
    actual = skc.nonnegify(x)
    if isinstance(x, np.ndarray):
        assert np.array_equal(actual, expected_result)
    else:
        assert actual == expected_result


@pytest.mark.unit
@pytest.mark.parametrize(
    "x,expected_result",
    [
        (2, -0.5),
        (-1, -2),
        (0.5, 1),
    ],
)
def test_beta_param(x: float, expected_result: float) -> None:
    actual = skc.beta_param(x)
    assert actual == expected_result


def test_beta_param_zero_div() -> None:
    with pytest.raises(ZeroDivisionError):
        skc.beta_param(0)


@pytest.mark.unit
@pytest.mark.parametrize(
    "x,s,t,expected_result",
    [
        (0, 0, 0, 0.5),
        (1, 0, 0, 0.5),
        (1, 1, 0, 0.7311),
        (1, 1, 1, 0.5),
        (-1, 1, 0, 0.2689),
        (-10, 1, 0, 0.0),
        (1, -1, 0, 0.2689),
        (1, -10, 0, 0.0),
    ],
)
def test_sigmoid(
    x: Union[int, float],
    s: Union[int, float],
    t: Union[int, float],
    expected_result: float,
) -> None:
    actual = round(rust.sigmoid(x, s, t), 4)
    assert actual == expected_result
