"""Recursive normalization helpers for immutable Metria evidence."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence, Set
from enum import Enum
from types import MappingProxyType
from typing import Any, TypeVar, cast

T = TypeVar("T")


def freeze_value(value: Any, *, _active: set[int] | None = None) -> Any:
    """Return an immutable, detached representation of an evidence value.

    Metria's core records intentionally retain JSON-like configuration and
    evidence rather than arbitrary live runtime objects. Mappings are copied
    into read-only proxies, sequences become tuples, sets become frozensets,
    bytearrays become bytes, and path-like values become their filesystem
    representation. Cyclic containers and unsupported object types are
    rejected so a recorded value cannot keep a mutable alias into caller state.
    """

    if value is None or isinstance(value, (str, bytes, bool, int, float, Enum)):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, os.PathLike):
        return os.fspath(value)

    active = _active if _active is not None else set()
    if isinstance(value, Mapping):
        marker = id(value)
        if marker in active:
            raise ValueError("cyclic evidence mappings are not supported")
        active.add(marker)
        try:
            frozen: dict[str, Any] = {}
            for key, item in value.items():
                if not isinstance(key, str):
                    raise TypeError("evidence mapping keys must be strings")
                frozen[key] = freeze_value(item, _active=active)
            return MappingProxyType(frozen)
        finally:
            active.remove(marker)

    if isinstance(value, Sequence):
        marker = id(value)
        if marker in active:
            raise ValueError("cyclic evidence sequences are not supported")
        active.add(marker)
        try:
            return tuple(freeze_value(item, _active=active) for item in value)
        finally:
            active.remove(marker)

    if isinstance(value, Set):
        marker = id(value)
        if marker in active:
            raise ValueError("cyclic evidence sets are not supported")
        active.add(marker)
        try:
            return frozenset(freeze_value(item, _active=active) for item in value)
        finally:
            active.remove(marker)

    raise TypeError(
        "unsupported evidence value type "
        f"{type(value).__module__}.{type(value).__qualname__}; "
        "store a serializable value or an artifact reference instead"
    )


def freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """Copy and deeply freeze a string-keyed evidence mapping."""

    return cast(Mapping[str, Any], freeze_value(value))


def freeze_typed_mapping(value: Mapping[str, T]) -> Mapping[str, T]:
    """Copy a mapping whose values are already immutable typed objects."""

    copied: dict[str, T] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise TypeError("mapping keys must be strings")
        copied[key] = item
    return MappingProxyType(copied)
