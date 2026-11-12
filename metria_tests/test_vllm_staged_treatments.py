"""STAGED vLLM treatments: validation, engine arguments, readback and gating."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
from metria_tests.test_vllm_runtime import FakeLLM, FakeVLLMModule, _patch_vllm, _spec

from metria import TreatmentSpec, TreatmentType
from metria.inspection import inspect_run_capabilities
from metria.runtimes.vllm import VLLMAdapter
from metria.runtimes.vllm_treatments import resolve_staged_treatments

SCHEDULER = TreatmentSpec(
    name="vllm.scheduler",
    kind=TreatmentType.RUNTIME_FEATURE,
    config={"enable_chunked_prefill": True, "max_num_batched_tokens": 2048},
)
CACHE = TreatmentSpec(
    name="vllm.cache_layout",
    kind=TreatmentType.RUNTIME_FEATURE,
    config={"block_size": 32},
)


class ReadbackLLM(FakeLLM):
    """Expose the scheduler and cache settings the engine actually applied."""

    applied_overrides: dict[str, Any] = {}

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        applied = {**kwargs, **self.applied_overrides}
        config = self.llm_engine.vllm_config
        config.scheduler_config = SimpleNamespace(
            enable_chunked_prefill=applied.get("enable_chunked_prefill", False),
            max_num_batched_tokens=applied.get("max_num_batched_tokens", 8192),
        )
        config.cache_config.block_size = applied.get("block_size", 16)


class ReadbackModule(FakeVLLMModule):
    LLM = ReadbackLLM


def _staged_spec(*treatments: TreatmentSpec, override: bool = True):
    spec = _spec()
    return replace(
        spec,
        treatments=(*spec.treatments, *treatments),
        trial_policy=(
            {"capability_overrides": ["vllm.staged_treatments"]} if override else {}
        ),
    )


def test_staged_settings_reach_the_engine_and_are_read_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_vllm(monkeypatch, ReadbackModule)
    ReadbackLLM.applied_overrides = {}
    adapter = VLLMAdapter()
    resolved = adapter.resolve(_staged_spec(SCHEDULER, CACHE), {})
    assert resolved["staged_treatments"]["vllm.cache_layout"] == {"block_size": 32}

    session = adapter.launch(resolved, {})
    llm = FakeLLM.instances[-1]
    assert llm.kwargs["enable_chunked_prefill"] is True
    assert llm.kwargs["max_num_batched_tokens"] == 2048
    assert llm.kwargs["block_size"] == 32

    readback = adapter.observe(session)["staged_treatments"]
    assert readback["qualification"] == "staged"
    assert {row["status"] for row in readback["settings"].values()} == {"applied"}
    assert (
        readback["settings"]["vllm.cache_layout.block_size"]["source"]
        == "cache_config.block_size"
    )


def test_contradicting_engine_readback_fails_the_launch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_vllm(monkeypatch, ReadbackModule)
    ReadbackLLM.applied_overrides = {"block_size": 16}
    adapter = VLLMAdapter()
    resolved = adapter.resolve(_staged_spec(CACHE), {})
    try:
        with pytest.raises(RuntimeError, match="did not apply staged"):
            adapter.launch(resolved, {})
    finally:
        ReadbackLLM.applied_overrides = {}


def test_missing_engine_readback_is_recorded_as_unobserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_vllm(monkeypatch)  # the base fake exposes no scheduler_config
    adapter = VLLMAdapter()
    session = adapter.launch(adapter.resolve(_staged_spec(SCHEDULER), {}), {})
    settings = adapter.observe(session)["staged_treatments"]["settings"]
    assert {row["status"] for row in settings.values()} == {"unobserved"}


def test_runs_without_staged_treatments_keep_their_evidence_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_vllm(monkeypatch)
    adapter = VLLMAdapter()
    resolved = adapter.resolve(_spec(), {})
    assert "staged_treatments" not in resolved
    assert "staged_treatments" not in resolved["support"]
    session = adapter.launch(resolved, {})
    assert "staged_treatments" not in adapter.observe(session)
    assert "block_size" not in FakeLLM.instances[-1].kwargs


@pytest.mark.parametrize(
    ("config", "match"),
    [
        ({"block_size": 24}, "one of"),
        ({"block_size": True}, "one of"),
        ({}, "at least one"),
        ({"swap_space": 4}, "unsupported"),
    ],
)
def test_invalid_cache_layout_settings_are_rejected(config, match) -> None:
    treatment = TreatmentSpec(
        name="vllm.cache_layout", kind=TreatmentType.RUNTIME_FEATURE, config=config
    )
    with pytest.raises(ValueError, match=match):
        resolve_staged_treatments((treatment,))


def test_invalid_scheduler_settings_and_duplicates_are_rejected() -> None:
    bad = TreatmentSpec(
        name="vllm.scheduler",
        kind=TreatmentType.RUNTIME_FEATURE,
        config={"max_num_batched_tokens": 0},
    )
    with pytest.raises(ValueError, match="positive"):
        resolve_staged_treatments((bad,))
    with pytest.raises(TypeError, match="boolean"):
        resolve_staged_treatments(
            (replace(SCHEDULER, config={"enable_chunked_prefill": 1}),)
        )
    with pytest.raises(ValueError, match="only one"):
        resolve_staged_treatments((SCHEDULER, SCHEDULER))
    with pytest.raises(ValueError, match="runtime_feature"):
        resolve_staged_treatments(
            (replace(CACHE, kind=TreatmentType.MODEL_TRANSFORMATION),)
        )


def test_preflight_blocks_staged_treatments_without_explicit_override() -> None:
    blocked = inspect_run_capabilities(_staged_spec(SCHEDULER, override=False))
    assert "vllm.staged_treatments" in {item.name for item in blocked.blocking}

    allowed = inspect_run_capabilities(_staged_spec(SCHEDULER))
    assert not allowed.blocking
    check = allowed.checks["vllm.staged_treatments"]
    assert check["override"]["applied"] is True


def test_preflight_ignores_runs_without_staged_treatments() -> None:
    result = inspect_run_capabilities(_spec())
    assert result.checks["vllm.staged_treatments"]["applicable"] is False
