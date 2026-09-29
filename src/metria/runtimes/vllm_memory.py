"""Named native worker observations for CUDA allocator peaks and KV allocations."""

from __future__ import annotations

from typing import Any


class MemoryWorkerExtension:
    def metria_reset_memory(self) -> dict[str, Any]:
        import torch

        device = getattr(self, "device", None)
        if getattr(device, "type", None) != "cuda":
            return {"available": False, "reason": "native worker is not CUDA"}
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        return {"available": True, "reset": "torch_cuda_allocator_peak"}

    def metria_memory_snapshot(self) -> dict[str, Any]:
        import torch

        device = getattr(self, "device", None)
        if getattr(device, "type", None) != "cuda":
            return {"available": False, "reason": "native worker is not CUDA"}
        torch.cuda.synchronize(device)
        caches = getattr(getattr(self, "model_runner", None), "kv_caches", None)
        if not isinstance(caches, list) or not caches:
            return {
                "available": False,
                "reason": "native KV allocation inventory is unavailable",
            }
        storages = {}
        for tensor in caches:
            if not isinstance(tensor, torch.Tensor) or tensor.device != device:
                return {
                    "available": False,
                    "reason": "native KV allocation layout is unsupported",
                }
            storage = tensor.untyped_storage()
            storages[storage.data_ptr()] = storage.nbytes()
        return {
            "available": True,
            "method": "metria.vllm_torch_allocator_and_kv_storage",
            "version": "1",
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "allocated_kv_bytes": sum(storages.values()),
            "scope": "native worker PyTorch allocator and unique KV tensor storage; excludes other processes and non-PyTorch allocations",
        }
