"""Overflow-safe numeric checks for untrusted JSON values."""

from __future__ import annotations

import math
from typing import Any


def is_finite(value: Any) -> bool:
    """Return whether ``value`` is a finite int or float, never raising.

    JSON decodes integer literals of any size to ``int``; ``math.isfinite``
    raises ``OverflowError`` for one beyond the float range. Such a value is
    not a usable finite number, so it is rejected like infinity or NaN.
    """

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False
