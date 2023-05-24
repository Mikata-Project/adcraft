"""Tests for synthetic_kw_helpers script."""
from typing import Union

import numpy as np
import pytest

from adcraft import rust


@pytest.mark.unit
@pytest.mark.parametrize(
    "x,expected_result",
    [
        (np.array([True, True, True, True]), 4),
        (np.array([False, False, False, False]), 0),
        (np.array([True, False, True, False]), 2),
    ],
)
def test_sum_array_bool(x: np.array, expected_result: int) -> None:
    actual = rust.sum_array_bool(x)
    assert actual == expected_result


@pytest.mark.unit
@pytest.mark.parametrize(
    "x",
    [
        (np.array([1, 1, 1, 1])),
        (np.array([0.0, 0.0, 0.0, 0.0])),
        ([True, True, True, True]),
    ],
)
def test_sum_array_bool_type(x: Union[list, np.array]) -> None:
    with pytest.raises(TypeError):
        rust.sum_array_bool(x)


@pytest.mark.unit
@pytest.mark.parametrize(
    "x,expected_result",
    [
        ([True, True, True, True], 4),
        ([False, False, False, False], 0),
        ([True, False, True, False], 2),
    ],
)
def test_sum_list_bool(x: list, expected_result: int) -> None:
    actual = rust.sum_list_bool(x)
    assert actual == expected_result


@pytest.mark.unit
@pytest.mark.parametrize(
    "x",
    [
        (np.array([1, 1, 1, 1])),
        (np.array([0.0, 0.0, 0.0, 0.0])),
        (np.array([True, True, True, True])),
        ([1, 1, 1, 1]),
        ([1.0, 0.0, 1.0, 0.0]),
    ],
)
def test_sum_list_bool_type(x: Union[list, np.array]) -> None:
    with pytest.raises(TypeError):
        rust.sum_list_bool(x)


@pytest.mark.unit
@pytest.mark.parametrize(
    "x,expected_result",
    [
        (np.array([1.0, 1.0, 1.0, 1.0]), 4),
        (np.array([0.0, 0.0, 0.0, 0.0]), 0),
        (np.array([1.0, 0.0, 1.0, 0.0]), 2),
    ],
)
def test_sum_array(x: np.array, expected_result: int) -> None:
    actual = rust.sum_array(x)
    assert actual == expected_result


@pytest.mark.unit
@pytest.mark.parametrize(
    "x",
    [
        (np.array([1, 1, 1, 1])),
        (np.array([0, 0, 0, 0])),
        (np.array([1, 0, 1, 0])),
        ([True, True, True, True]),
        (np.array([True, True, False, False])),
    ],
)
def test_sum_array_type(x: Union[list, np.array]) -> None:
    with pytest.raises(TypeError):
        rust.sum_array(x)


@pytest.mark.unit
@pytest.mark.parametrize(
    "x,expected_result",
    [
        ([1, 1, 1, 1], 4),
        ([1.0, 1.0, 1.0, 1.0], 4),
        ([0, 0, 0, 0], 0),
        ([0.0, 0.0, 0.0, 0.0], 0),
        ([1, 0, 1, 0], 2),
        ([1.0, 0.0, 1.0, 0.0], 2),
        ([True, True, False, False], 2),
        (np.array([True, True, False, False]), 2),
        (
            np.array([True, True, False, False]),
            np.array([True, True, False, False]).sum(),
        ),
        (
            np.array([44.7, 88.465, 38.462, 300.0]),
            np.array([44.7, 88.465, 38.462, 300.0]).sum(),
        ),
    ],
)
def test_sum_list(x: list, expected_result: int) -> None:
    actual = rust.sum_list(x)
    assert actual == expected_result


@pytest.mark.unit
@pytest.mark.parametrize(
    "x,expected_result",
    [
        ([1, 1, 1, 1], np.array([0.0, 0.0, 0.0, 0.0])),
        ([1.0, 1.0, 1.0, 1.0], np.array([0.0, 0.0, 0.0, 0.0])),
        ([0, 0, 0, 0], np.array([0.0, 0.0, 0.0, 0.0])),
    ],
)
def test_list_to_zeros(x: list, expected_result: np.array) -> None:
    actual = rust.list_to_zeros(x)
    assert np.array_equal(actual, expected_result)


@pytest.mark.unit
@pytest.mark.parametrize(
    "x,y,z",
    [
        (0.0, 0.0, 1.0),
        (-10.0, 0.0, 1.0),
        (10.0, 0.0, 1.0),
        (0.5, 0.0, 1.0),
        (0.4999, 0.0, 1.0),
    ],
)
def test_probify_float(x: float, y: float, z: float) -> None:
    assert rust.probify_float(x, y, z) == np.clip(x, y, z)
