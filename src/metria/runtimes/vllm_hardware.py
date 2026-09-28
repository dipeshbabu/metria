"""Runtime-observed device identity, never inferred from environment flags."""

from __future__ import annotations

import hashlib
import importlib
from typing import Any


def native_hardware() -> dict[str, Any]:
    unknown = {"status": "unknown", "source": "vllm_native_platform"}
    try:
        platform = importlib.import_module("vllm.platforms").current_platform
        device_type = getattr(platform, "device_type", None)
        if device_type == "cpu":
            return {
                "status": "observed",
                "device_type": "cpu",
                "source": "vllm_native_platform",
            }
        if device_type != "cuda":
            return unknown
        torch = importlib.import_module("torch")
        if getattr(torch.version, "hip", None) is not None:
            return {
                "status": "observed",
                "device_type": "rocm",
                "source": "vllm_native_platform_and_torch_hip",
                "hip_runtime": torch.version.hip,
            }
        device = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device)
        uuid = getattr(properties, "uuid", None)
        return {
            "status": "observed",
            "device_type": "cuda",
            "source": "vllm_native_platform_and_torch_cuda",
            "logical_device": device,
            "name": properties.name,
            "compute_capability": [properties.major, properties.minor],
            "capacity_bytes": properties.total_memory,
            "uuid_sha256": hashlib.sha256(str(uuid).encode()).hexdigest()
            if uuid is not None
            else None,
            "cuda_runtime": torch.version.cuda,
        }
    except (ImportError, AttributeError, RuntimeError):
        return unknown
