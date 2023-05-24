import pytest


@pytest.mark.unit
def test_sanity() -> None:
    assert True


@pytest.mark.unit
def test_another() -> None:
    assert 1 == 1
