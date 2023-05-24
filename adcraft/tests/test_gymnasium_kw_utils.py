"""Tests for gymnasium_kw_utils script."""
from gymnasium import spaces
import pytest

from adcraft import gymnasium_kw_utils as gutils


@pytest.mark.unit
def test_get_action_space() -> None:
    actual = type(gutils.get_action_space(1))
    expected = spaces.Dict
    assert actual == expected
