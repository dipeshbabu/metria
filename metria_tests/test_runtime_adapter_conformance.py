from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from runtime_contract import RuntimeContractCase, exercise_runtime_contract

from metria import RunSpec
from metria.protocols import InferenceRequest
from metria.runtimes import LlamaCppAdapter, VLLMAdapter
import metria.runtimes.vllm as vllm_runtime


def _write_executable(path: Path) -> None:
    path.write_text("fake runtime binary\n", encoding="utf-8")
    if os.name != "nt":
        path.chmod(0o755)


def test_llamacpp_adapter_conforms_to_shared_runtime_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suffix = ".exe" if os.name == "nt" else ""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_executable(bin_dir / f"llama-cli{suffix}")
    _write_executable(bin_dir / f"llama-completion{suffix}")
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"fake gguf model")

    def fake_run(
        command: list[str],
        *,
        capture_output: bool,
        text: bool,
        timeout: float,
        check: bool,
        cwd: None,
        env: dict[str, str],
    ) -> subprocess.CompletedProcess[str]:
        del capture_output, text, timeout, check, cwd, env
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout="contract output",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    spec = RunSpec(
        model={"id": "example/model", "path": str(model_path)},
        runtime={
            "name": "llamacpp",
            "bin_dir": str(bin_dir),
            "n_gpu_layers": 0,
        },
        scenario={"name": "decode", "max_tokens": 4},
        measurements=("contract.measurement",),
    )
    case = RuntimeContractCase(
        adapter=LlamaCppAdapter(),
        spec=spec,
        environment={"hardware_class": "contract-host"},
        requests=(InferenceRequest(prompt="private llama contract prompt"),),
        privacy_terms=("private llama contract prompt",),
    )

    exercise_runtime_contract(case)


class _FakeSamplingParams:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class _FakeLLM:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def generate(
        self,
        prompts: list[str],
        *,
        sampling_params: list[_FakeSamplingParams],
        use_tqdm: bool,
    ) -> list[SimpleNamespace]:
        assert use_tqdm is False
        assert len(prompts) == len(sampling_params)
        return [
            SimpleNamespace(
                outputs=[SimpleNamespace(text="contract output", token_ids=[1, 2])]
            )
            for _ in prompts
        ]


def test_vllm_adapter_conforms_to_shared_runtime_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_module = SimpleNamespace(
        LLM=_FakeLLM,
        SamplingParams=_FakeSamplingParams,
        __version__="test-version",
    )
    monkeypatch.setitem(sys.modules, "vllm", fake_module)
    monkeypatch.setattr(vllm_runtime, "_optional_dependency_available", lambda: True)

    spec = RunSpec(
        model={"id": "example/model", "revision": "abc123"},
        runtime={"name": "vllm", "max_model_len": 4096, "enforce_eager": True},
        scenario={"name": "decode", "max_tokens": 4},
        measurements=("contract.measurement",),
    )
    case = RuntimeContractCase(
        adapter=VLLMAdapter(),
        spec=spec,
        environment={"hardware_class": "contract-host"},
        requests=(InferenceRequest(prompt="private vllm contract prompt"),),
        privacy_terms=("private vllm contract prompt",),
    )

    exercise_runtime_contract(case)
