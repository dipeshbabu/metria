from __future__ import annotations

from types import SimpleNamespace

import pytest

from metria.measurements import prefix_workload as workload
from metria.measurements.trajectory import compare_trajectory_results
from metria.protocols import InferenceBatch
from metria.runtimes.vllm import VLLMSession, _response_capture


class Session:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.calls = []
        self.resets = []
        self.cache = False

    def reset(self, scope="measurement"):
        self.resets.append(scope)
        self.cache = False

    def infer(self, requests, capture=()):
        assert len(requests) == 1
        self.calls.append(requests[0])
        cached = 4 if self.cache and self.enabled else 0
        self.cache = True
        return InferenceBatch(
            outputs=("private model answer",),
            captures={"token_ids": ((1, 2),)},
            metadata={
                "invocations": (
                    {"prompt_tokens": 8, "cached_tokens": cached, "finished": True},
                )
            },
        )

    def close(self):
        pass


def config(**overrides):
    return {
        "prompts": [
            {"id": "a", "prompt": "private shared prefix a"},
            {"id": "b", "prompt": "private shared prefix b"},
        ],
        "warmup_trials": 1,
        "measured_trials": 2,
        **overrides,
    }


def test_trials_reset_caches_and_exclude_warmup_from_measurements(monkeypatch):
    ticks = iter(range(0, 12_000, 1000))
    monkeypatch.setattr(workload.time, "perf_counter_ns", lambda: next(ticks))
    session = Session()
    result = workload.PrefixWorkloadProtocol().execute(session, {}, config())
    assert session.resets == ["prefix-cache"] * 3
    assert [row.prompt for row in session.calls] == [
        "private shared prefix a",
        "private shared prefix b",
    ] * 3
    assert result.evidence["n_prompts"] == 4
    assert [row["id"] for row in result.evidence["prompts"]] == [
        "0:a",
        "0:b",
        "1:a",
        "1:b",
    ]
    metric = result.metrics["request_latency_seconds"]
    assert len(metric.samples) == 4 and metric.value == pytest.approx(0.000001)
    samples = result.evidence["workload"]["samples"]
    assert [row["cached_tokens"] for row in samples] == [0, 4, 0, 4, 0, 4]
    assert sum(row["measured"] for row in samples) == 4
    assert "private" not in str(result.evidence)


def test_profile_reuses_trajectory_method_and_reference_role():
    protocol = workload.PrefixWorkloadProtocol()
    reference = protocol.execute(Session(False), {}, config())
    candidate = protocol.execute(Session(True), {}, config())
    compared = compare_trajectory_results(reference, candidate)
    assert compared.metrics["trajectory_agreement_score"].value == 100


@pytest.mark.parametrize(
    "overrides",
    [
        {"measured_trials": 0},
        {"measured_trials": True},
        {"warmup_trials": -1},
        {"warmup_trials": 4},
        {"measured_trials": 11},
        {"unknown": 1},
    ],
)
def test_invalid_trial_policy_is_rejected_during_requirements(overrides):
    with pytest.raises(ValueError):
        workload.PrefixWorkloadProtocol().requirements(config(**overrides))


def test_backwards_clock_never_becomes_a_zero_latency(monkeypatch):
    ticks = iter([1000, 0])
    monkeypatch.setattr(workload.time, "perf_counter_ns", lambda: next(ticks))
    with pytest.raises(RuntimeError, match="clock"):
        workload.PrefixWorkloadProtocol().execute(Session(), {}, config())


def test_missing_native_cache_counts_remain_unknown():
    session = Session()
    original = session.infer

    def infer(requests, capture=()):
        result = original(requests, capture)
        return InferenceBatch(outputs=result.outputs, captures=result.captures)

    session.infer = infer
    result = workload.PrefixWorkloadProtocol().execute(session, {}, config())
    assert all(
        row["cached_tokens"] is None for row in result.evidence["workload"]["samples"]
    )


@pytest.mark.parametrize("cached", [-1, True, 1.5, 9])
def test_native_cache_counts_must_be_valid(cached):
    response = SimpleNamespace(
        outputs=[SimpleNamespace(text="answer", token_ids=[1])],
        prompt_token_ids=list(range(8)),
        num_cached_tokens=cached,
        finished=True,
    )
    with pytest.raises(RuntimeError, match="cached-token"):
        _response_capture(response, 0)


def test_native_capture_keeps_real_counts_and_completion_state():
    response = SimpleNamespace(
        outputs=[SimpleNamespace(text="answer", token_ids=[1, 2])],
        prompt_token_ids=list(range(8)),
        num_cached_tokens=4,
        finished=True,
    )
    text, tokens, facts = _response_capture(response, 0)
    assert text == "answer" and tokens == (1, 2)
    assert facts == {"prompt_tokens": 8, "cached_tokens": 4, "finished": True}


@pytest.mark.parametrize("confirmed", [True, False, None])
def test_prefix_reset_requires_public_confirmation(confirmed):
    session = object.__new__(VLLMSession)
    session._closed = False
    session._reset_count = 0
    session._reset_events = []
    session._llm = SimpleNamespace(reset_prefix_cache=lambda: confirmed)
    if confirmed is True:
        session.reset("prefix-cache")
        assert session._reset_count == 1
        assert session._reset_events == [
            {"scope": "prefix-cache", "mode": "public_prefix_cache_reset"}
        ]
    else:
        with pytest.raises(RuntimeError, match="confirm"):
            session.reset("prefix-cache")
        assert session._reset_count == 0
