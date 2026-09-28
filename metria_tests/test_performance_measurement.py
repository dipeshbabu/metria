from __future__ import annotations

from dataclasses import replace

import pytest

from metria import RunRecord, RunSpec, RunStatus
from metria.measurements import TokenTrajectoryProtocol
from metria.measurements.performance import (
    compare_performance,
    measure_invocation_performance,
)


def _run(duration=2.0):
    return RunRecord(
        study_name="performance",
        run_id="run",
        requested=RunSpec(
            model={"id": "test"},
            runtime={"name": "llamacpp"},
            scenario={"max_tokens": 2},
            measurements=(TokenTrajectoryProtocol.name,),
        ),
        resolved={},
        observed={
            "invocations": [
                {
                    "duration_seconds": duration,
                    "generation": {"max_tokens": 2},
                    "prompt_sha256": "a" * 64,
                    "returncode": 0,
                }
            ]
        },
        status=RunStatus.COMPLETED,
        evidence={
            "measurements": {
                TokenTrajectoryProtocol.name: {
                    "prompts": [
                        {"id": "one", "prompt_sha256": "a" * 64, "token_count": 2}
                    ]
                }
            }
        },
    )


@pytest.mark.parametrize(
    "candidate,direction,absolute,relative",
    [
        (1.0, "improved", -1.0, -0.5),
        (3.0, "regressed", 1.0, 0.5),
        (2.0, "unchanged", 0, 0),
    ],
)
def test_latency_delta_retains_full_methodology(
    candidate, direction, absolute, relative
):
    impact = compare_performance(
        measure_invocation_performance(_run()),
        measure_invocation_performance(_run(candidate)),
        comparable=True,
    )
    assert impact["available"]
    assert impact["absolute_delta"] == absolute
    assert impact["relative_delta"] == relative
    assert impact["direction"] == direction
    assert impact["methodology"]["warmup_trials"] == 0
    assert impact["methodology"]["trials_per_prompt"] == 1
    assert impact["methodology"]["scope"].startswith("process startup")
    assert "decode_throughput" in impact["unsupported_metrics"]


@pytest.mark.parametrize("duration", [0, 1e-15, 1e-9])
def test_zero_or_near_zero_reference_has_no_relative_delta(duration):
    impact = compare_performance(
        measure_invocation_performance(_run(duration)),
        measure_invocation_performance(_run(1)),
        comparable=True,
    )
    assert impact["available"] and impact["relative_delta"] is None
    assert impact["relative_unavailable_reason"]


@pytest.mark.parametrize("duration", [None, True, -1, float("nan"), float("inf")])
def test_missing_and_invalid_clocks_remain_unavailable(duration):
    result = measure_invocation_performance(_run(duration))
    assert not result.evidence["available"]
    assert not result.metrics


@pytest.mark.parametrize(
    "field,value",
    [("unit", "milliseconds"), ("method", "estimated.words"), ("version", "2")],
)
def test_incompatible_metric_identity_blocks_delta(field, value):
    left = measure_invocation_performance(_run())
    metric = left.metrics["request_latency_seconds"]
    wrong = replace(metric, definition=replace(metric.definition, **{field: value}))
    right = replace(left, metrics={"request_latency_seconds": wrong})
    assert not compare_performance(left, right, comparable=True)["available"]


def test_workload_or_trial_policy_change_blocks_delta():
    left = measure_invocation_performance(_run())
    for field, value in [
        ("workload_sha256", "b" * 64),
        ("trials_per_prompt", 2),
        ("warmup_trials", 1),
    ]:
        right = replace(
            left,
            evidence={
                **left.evidence,
                "methodology": {**left.evidence["methodology"], field: value},
            },
        )
        assert not compare_performance(left, right, comparable=True)["available"]
    assert not compare_performance(left, left, comparable=False)["available"]


def test_partial_runs_and_unsupported_runtimes_never_produce_a_speedup():
    run = _run()
    for changed in (
        replace(run, status=RunStatus.PARTIAL),
        replace(run, requested=replace(run.requested, runtime={"name": "vllm"})),
        replace(run, observed={}),
    ):
        result = measure_invocation_performance(changed)
        assert not result.evidence["available"]
        assert not compare_performance(result, result, comparable=True)["available"]
