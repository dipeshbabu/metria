"""Narrow preflight for a confirmed unsupported native vLLM cache format."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def fp8_cache_gap(
    version: str | None, dtype: str, hardware: Mapping[str, Any]
) -> str | None:
    """Reject the pinned CUDA/Turing case observed to fail before inference.

    No positive support claim is made for other devices or runtime builds.
    The 0.30 CUDA build's default Turing attention backend is Triton, whose
    FP8 cache path requires native fp8e4nv support (SM89 or newer).
    """
    if version != "0.30.0+cu129" or not dtype.startswith("fp8"):
        return None
    if hardware.get("status") != "observed" or hardware.get("device_type") != "cuda":
        return None
    capability = hardware.get("compute_capability")
    if (
        not isinstance(capability, (list, tuple))
        or len(capability) != 2
        or any(type(value) is not int for value in capability)
    ):
        return None
    if tuple(capability) != (7, 5):
        return None
    return (
        "vLLM 0.30 CUDA FP8 KV cache is unsupported on the observed SM75 device: "
        "its default Triton backend requires SM89+ native FP8 support. Use "
        "the model's automatic cache dtype here, or qualify FP8 on a supported GPU."
    )
