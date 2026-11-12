"""Overflow-safe finite-number checks for untrusted JSON values."""

from __future__ import annotations

import json

import pytest

from metria._numbers import is_finite


@pytest.mark.parametrize("value", [0, -3, 1.5, 1e308, -(2**1000)])
def test_finite_numbers_are_accepted(value) -> None:
    assert is_finite(value)


@pytest.mark.parametrize(
    "value",
    [
        json.loads("9" * 400),
        -json.loads("9" * 400),
        float("inf"),
        float("-inf"),
        float("nan"),
        True,
        False,
        "1",
        None,
    ],
)
def test_non_finite_overflowing_and_non_numeric_values_are_rejected(value) -> None:
    assert is_finite(value) is False
