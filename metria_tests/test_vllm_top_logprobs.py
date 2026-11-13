"""STAGED vLLM raw top-k logprob capture feeding trajectory KL divergence."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import pytest
from metria_tests.test_vllm_runtime import FakeLLM, FakeVLLMModule, _patch_vllm, _spec

from metria.measurements import (
    TokenDistributionProtocol,
    compare_trajectory_distributions,
)
from metria.protocols import CaptureRequest, InferenceRequest
from metria.runtimes.vllm import VLLMAdapter
from metria.runtimes.vllm_logprobs import TOP_LOGPROBS_K

CAPTURES = (CaptureRequest(kind="token_ids"), CaptureRequest(kind="token_top_logprobs"))


def _step(sampled: int, *, extra_rank: int | None = None, shift: float = 0.0):
    """Top-20 entries (token ids 100..119) plus an optional out-of-top-k sample."""
    raw = [math.exp(-0.3 * rank) for rank in range(TOP_LOGPROBS_K)]
    raw[0] *= math.exp(shift)
    norm = sum(raw) / 0.97
    entries = {
        100 + rank: SimpleNamespace(
            logprob=math.log(value / norm), rank=rank + 1, decoded_token="x"
        )
        for rank, value in enumerate(raw)
    }
    if extra_rank is not None:
        entries[sampled] = SimpleNamespace(
            logprob=-12.0, rank=extra_rank, decoded_token="y"
        )
    return entries


class LogprobLLM(FakeLLM):
    mode: Any = "raw_logprobs"
    shift = 0.0
    drop_step = False

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.llm_engine.vllm_config.model_config = SimpleNamespace(
            logprobs_mode=self.mode
        )

    def generate(self, prompts, *, sampling_params, use_tqdm):
        responses = super().generate(
            prompts, sampling_params=sampling_params, use_tqdm=use_tqdm
        )
        for response, params in zip(responses, sampling_params, strict=True):
            assert params.kwargs.get("logprobs") == TOP_LOGPROBS_K
            output = response.outputs[0]
            steps = [
                _step(output.token_ids[0], shift=self.shift),
                _step(output.token_ids[1], extra_rank=57, shift=self.shift),
            ]
            output.logprobs = steps[:1] if self.drop_step else steps
        return responses


class LogprobModule(FakeVLLMModule):
    LLM = LogprobLLM


@pytest.fixture
def session(monkeypatch):
    _patch_vllm(monkeypatch, LogprobModule)
    LogprobLLM.mode, LogprobLLM.shift, LogprobLLM.drop_step = "raw_logprobs", 0.0, False
    adapter = VLLMAdapter()
    return adapter, adapter.launch(adapter.resolve(_spec(), {}), {})


def test_capture_returns_constant_width_rank_ordered_distributions(session) -> None:
    adapter, live = session
    batch = live.infer((InferenceRequest("a"), InferenceRequest("b")), capture=CAPTURES)
    distributions = batch.captures["token_top_logprobs"]
    assert len(distributions) == 2
    for prompt in distributions:
        assert [len(step) for step in prompt] == [TOP_LOGPROBS_K, TOP_LOGPROBS_K]
        for step in prompt:
            assert [token for token, _ in step] == list(range(100, 120))
            values = [logprob for _, logprob in step]
            assert values == sorted(values, reverse=True)
    invocation = adapter.observe(live)["invocations"][0]
    assert invocation["top_logprobs"] == {
        "top_k": 20,
        "mode": "raw_logprobs",
        "qualification": "staged",
    }


def test_token_only_capture_does_not_request_logprobs(session, monkeypatch) -> None:
    _, live = session
    seen = {}

    def generate(prompts, *, sampling_params, use_tqdm):
        seen["logprobs"] = sampling_params[0].kwargs.get("logprobs")
        return FakeLLM.generate(
            live._llm, prompts, sampling_params=sampling_params, use_tqdm=use_tqdm
        )

    monkeypatch.setattr(live._llm, "generate", generate)
    batch = live.infer(
        (InferenceRequest("a"),), capture=(CaptureRequest(kind="token_ids"),)
    )
    assert seen["logprobs"] is None
    assert "token_top_logprobs" not in batch.captures


def test_processed_logprob_mode_is_refused(monkeypatch) -> None:
    _patch_vllm(monkeypatch, LogprobModule)
    LogprobLLM.mode = "processed_logprobs"
    try:
        adapter = VLLMAdapter()
        live = adapter.launch(adapter.resolve(_spec(), {}), {})
        with pytest.raises(RuntimeError, match="raw_logprobs"):
            live.infer((InferenceRequest("a"),), capture=CAPTURES)
    finally:
        LogprobLLM.mode = "raw_logprobs"


def test_missing_step_coverage_fails(session) -> None:
    _, live = session
    LogprobLLM.drop_step = True
    try:
        with pytest.raises(RuntimeError, match="every generated token"):
            live.infer((InferenceRequest("a"),), capture=CAPTURES)
    finally:
        LogprobLLM.drop_step = False


def test_vllm_capture_feeds_trajectory_kld(session) -> None:
    _, live = session
    config = {"prompts": [{"id": "a", "prompt": "a"}, {"id": "b", "prompt": "b"}]}
    reference = TokenDistributionProtocol().execute(live, {}, config)
    repeat = TokenDistributionProtocol().execute(live, {}, config)
    assert (
        compare_trajectory_distributions(reference, repeat)
        .metrics["trajectory_kld_mean"]
        .value
        == 0.0
    )
    LogprobLLM.shift = 0.5
    try:
        shifted = TokenDistributionProtocol().execute(live, {}, config)
    finally:
        LogprobLLM.shift = 0.0
    kld = compare_trajectory_distributions(reference, shifted).metrics[
        "trajectory_kld_mean"
    ]
    assert kld.value > 0.0 and math.isfinite(kld.value)


def test_probe_reports_experimental_distribution_capture(monkeypatch) -> None:
    _patch_vllm(monkeypatch, LogprobModule)
    report = VLLMAdapter().probe(_spec(), {})
    assert report.evidence["token_top_logprobs_capture"]["status"] == "experimental"
