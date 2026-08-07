from __future__ import annotations

import pytest

from metria import (
    ComparisonPlan,
    MetricDefinition,
    MetricDirection,
    MetricSample,
    MetricSummary,
    RunRecord,
    RunSpec,
    RunStatus,
    StudySpec,
    TreatmentSpec,
    TreatmentType,
    compare_runs,
)


def _run_spec(*, runtime: str = "llamacpp", treatment: str = "baseline") -> RunSpec:
    return RunSpec(
        model={"id": "example/model", "revision": "abc123"},
        runtime={"name": runtime, "version": "1"},
        scenario={"name": "decode", "context": 4096},
        measurements=("decode_tps", "distribution_drift"),
        treatments=(
            TreatmentSpec(
                name=treatment,
                kind=TreatmentType.RUNTIME_FEATURE,
                config={"kv_dtype": treatment},
            ),
        ),
        trial_policy={"warmup": 1, "repetitions": 3},
        environment_selector={"hardware_class": "gpu-a"},
    )


def _metric(*, method: str = "wall_clock") -> MetricSummary:
    definition = MetricDefinition(
        name="decode_tps",
        unit="tokens/s",
        direction=MetricDirection.HIGHER_IS_BETTER,
        method=method,
    )
    return MetricSummary(
        definition=definition,
        value=42.0,
        samples=(MetricSample(40.0), MetricSample(42.0), MetricSample(44.0)),
        aggregation="mean",
        uncertainty={"stdev": 2.0},
    )


def _record(
    *,
    runtime: str = "llamacpp",
    treatment: str = "baseline",
    method: str = "wall_clock",
    hardware_class: str = "gpu-a",
) -> RunRecord:
    return RunRecord(
        study_name="runtime-study",
        run_id=f"{runtime}-{treatment}",
        requested=_run_spec(runtime=runtime, treatment=treatment),
        resolved={"runtime": runtime, "kv_dtype": treatment},
        observed={"hardware_class": hardware_class, "runtime": runtime},
        status=RunStatus.COMPLETED,
        metrics={"decode_tps": _metric(method=method)},
    )


def test_comparison_plan_rejects_overlapping_dimensions() -> None:
    with pytest.raises(ValueError, match="cannot overlap"):
        ComparisonPlan(vary=frozenset({"runtime"}), control=frozenset({"runtime"}))


def test_study_requires_a_run() -> None:
    with pytest.raises(ValueError, match="at least one run"):
        StudySpec(
            name="empty",
            runs=(),
            comparison=ComparisonPlan(),
        )


def test_runtime_can_be_an_intentional_variable() -> None:
    plan = ComparisonPlan(
        vary=frozenset({"runtime", "treatments"}),
        control=frozenset({"model", "scenario", "trial_policy"}),
        block_by=frozenset({"observed.hardware_class"}),
    )

    report = compare_runs(
        _record(runtime="llamacpp", treatment="fp16"),
        _record(runtime="vllm", treatment="fp8"),
        plan,
    )

    assert report.compatible
    assert report.comparable_metrics == ("decode_tps",)


def test_controlled_dimension_mismatch_is_reported() -> None:
    left = _record()
    right = RunRecord(
        study_name=left.study_name,
        run_id="different-model",
        requested=RunSpec(
            model={"id": "other/model", "revision": "def456"},
            runtime=left.requested.runtime,
            scenario=left.requested.scenario,
            measurements=left.requested.measurements,
            treatments=left.requested.treatments,
            trial_policy=left.requested.trial_policy,
            environment_selector=left.requested.environment_selector,
        ),
        resolved=left.resolved,
        observed=left.observed,
        status=RunStatus.COMPLETED,
        metrics=left.metrics,
    )
    plan = ComparisonPlan(
        vary=frozenset({"treatments"}),
        control=frozenset({"model", "scenario"}),
    )

    report = compare_runs(left, right, plan)

    assert not report.compatible
    assert report.issues[0].dimension == "model"
    assert report.issues[0].reason == "controlled dimension differs"


def test_blocking_dimension_must_match_for_pairwise_comparison() -> None:
    plan = ComparisonPlan(
        vary=frozenset({"runtime"}),
        block_by=frozenset({"observed.hardware_class"}),
    )

    report = compare_runs(
        _record(runtime="llamacpp", hardware_class="gpu-a"),
        _record(runtime="vllm", hardware_class="gpu-b"),
        plan,
    )

    assert not report.compatible
    assert report.issues[0].reason == "runs belong to different comparison blocks"


def test_metric_method_identity_prevents_invalid_raw_comparison() -> None:
    plan = ComparisonPlan(vary=frozenset({"runtime"}))

    report = compare_runs(
        _record(runtime="llamacpp", method="full_vocabulary_kld"),
        _record(runtime="vllm", method="top_k_kld"),
        plan,
    )

    assert not report.compatible
    assert report.comparable_metrics == ()
    assert "decode_tps" in report.incompatible_metrics


def test_requested_resolved_and_observed_state_are_distinct() -> None:
    record = _record(treatment="fp8")

    assert record.requested.treatments[0].config["kv_dtype"] == "fp8"
    assert record.resolved["kv_dtype"] == "fp8"
    assert record.observed["runtime"] == "llamacpp"
