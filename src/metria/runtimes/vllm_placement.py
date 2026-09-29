"""Native worker placement observations for the qualified CPU upgrade profile."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..protocols import RuntimeSession
from .vllm import VLLMAdapter, VLLMSession


def worker_affinity(tasks: Path = Path("/proc/self/task")) -> list[int] | None:
    import os

    if not hasattr(os, "sched_getaffinity"):
        return None
    cpus: set[int] = set()
    try:
        for task in tasks.iterdir():
            try:
                cpus.update(os.sched_getaffinity(int(task.name)))
            except ProcessLookupError:
                continue
    except (OSError, ValueError):
        return None
    return sorted(cpus) or None


def observe_worker(worker: Any) -> dict[str, Any]:

    import torch

    return {
        "device_type": str(worker.device.type),
        "cpu_affinity": worker_affinity(),
        "affinity_source": "linux_worker_thread_union",
        "torch_intraop_threads": torch.get_num_threads(),
    }


class CPUUpgradeAdapter(VLLMAdapter):
    worker_extension_cls = "metria.runtimes.vllm_placement.MetriaWorkerExtension"

    def launch(
        self, resolved: Mapping[str, Any], environment: Mapping[str, Any]
    ) -> RuntimeSession:
        session = super().launch(resolved, environment)
        assert isinstance(session, VLLMSession)
        try:
            session.capture_worker_placement()
        except BaseException:
            session.close()
            raise
        return session

    def observe(self, session: RuntimeSession) -> Mapping[str, Any]:
        if not isinstance(session, VLLMSession):
            raise TypeError("CPU upgrade observation requires a vLLM session")
        session.capture_worker_placement()
        return super().observe(session)


class MetriaWorkerExtension:
    """Named control RPCs loaded from the same pinned Metria installation."""

    def metria_worker_placement(self) -> dict[str, Any]:
        return observe_worker(self)
