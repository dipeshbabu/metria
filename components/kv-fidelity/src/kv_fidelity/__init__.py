"""Source compatibility namespace; use metria.fidelity."""

from importlib import import_module
from typing import Any

_implementation = import_module("metria.fidelity")
__all__ = getattr(_implementation, "__all__", [])


def __getattr__(name: str) -> Any:
    return getattr(_implementation, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_implementation)))
