"""Environment variables that can change numerics, placement or collectives.

Settings such as ``NCCL_P2P_DISABLE`` or ``CUDA_VISIBLE_DEVICES`` decide which
devices a run uses and how they communicate, yet they live outside the recipe.
Runs record an allow-listed snapshot so two runs that differ only by one of
these variables are not silently treated as identical.

Names that look like credentials are never recorded. Values that look like
filesystem paths are recorded only as SHA-256, so private paths stay out of
evidence.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Mapping

_EXACT = frozenset(
    {
        "CUDA_VISIBLE_DEVICES",
        "CUDA_DEVICE_ORDER",
        "CUDA_LAUNCH_BLOCKING",
        "CUBLAS_WORKSPACE_CONFIG",
        "PYTORCH_CUDA_ALLOC_CONF",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
    }
)
_PREFIXES = ("NCCL_", "VLLM_")
_SECRET = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|AUTH", re.IGNORECASE)
_MAX_VALUE = 256


def _looks_like_path(value: str) -> bool:
    return "/" in value or "\\" in value or value.startswith("~")


def numerics_environment(
    environ: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Return the allow-listed, non-secret variables, paths hashed."""

    source = os.environ if environ is None else environ
    result: dict[str, object] = {}
    for name in sorted(source):
        if name not in _EXACT and not name.startswith(_PREFIXES):
            continue
        if _SECRET.search(name):
            continue
        value = source[name]
        if _looks_like_path(value) or len(value) > _MAX_VALUE:
            result[name] = {"sha256": hashlib.sha256(value.encode()).hexdigest()}
        else:
            result[name] = value
    return result
