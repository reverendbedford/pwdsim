import pytest

from pwdsim import units


def test_conversions():
    assert pytest.approx(0.3048) == units.FOOT
    assert pytest.approx(0.45359237) == 16 * units.OUNCE
    assert pytest.approx(3.141592653589793) == 180 * units.DEGREE
