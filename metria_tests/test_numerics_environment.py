"""Allow-listed numerics environment recorded with vLLM runs."""

from __future__ import annotations

import hashlib

from metria_tests.test_vllm_runtime import _patch_vllm, _spec

from metria.runtimes.numerics_environment import numerics_environment
from metria.runtimes.vllm import VLLMAdapter


def test_only_allow_listed_non_secret_variables_are_recorded():
    environ = {
        "NCCL_P2P_DISABLE": "1",
        "CUDA_VISIBLE_DEVICES": "0,1",
        "VLLM_CPU_OMP_THREADS_BIND": "0-1|2-3",
        "VLLM_API_KEY": "secret-value",
        "NCCL_AUTH_TOKEN": "secret-value",
        "HF_TOKEN": "secret-value",
        "PATH": "/usr/bin",
        "HOME": "/home/someone",
    }
    recorded = numerics_environment(environ)
    assert recorded == {
        "CUDA_VISIBLE_DEVICES": "0,1",
        "NCCL_P2P_DISABLE": "1",
        "VLLM_CPU_OMP_THREADS_BIND": "0-1|2-3",
    }
    assert "secret-value" not in str(recorded)


def test_path_like_and_long_values_are_hashed():
    path = "/home/someone/.cache/vllm"
    recorded = numerics_environment(
        {"VLLM_CACHE_ROOT": path, "NCCL_DEBUG_FILE": "x" * 300}
    )
    assert recorded["VLLM_CACHE_ROOT"] == {
        "sha256": hashlib.sha256(path.encode()).hexdigest()
    }
    assert set(recorded["NCCL_DEBUG_FILE"]) == {"sha256"}
    assert path not in str(recorded)


def test_vllm_runs_record_the_snapshot(monkeypatch):
    _patch_vllm(monkeypatch)
    monkeypatch.setenv("NCCL_P2P_DISABLE", "1")
    monkeypatch.setenv("VLLM_API_KEY", "never-recorded")
    adapter = VLLMAdapter()
    session = adapter.launch(adapter.resolve(_spec(), {}), {})
    observed = adapter.observe(session)
    assert observed["numerics_environment"]["NCCL_P2P_DISABLE"] == "1"
    assert "VLLM_API_KEY" not in observed["numerics_environment"]
