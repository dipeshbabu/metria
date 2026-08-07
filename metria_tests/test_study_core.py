from __future__ import annotations

from dataclasses import replace

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
from metria.protocols import InferenceBatch, InferenceRequest, SupportReport


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


def test_missing_nested_dimension_cannot_prove_comparability() -> None:
    plan = ComparisonPlan(
        vary=frozenset({"runtime"}),
        block_by=frozenset({"observed.hardware_class"}),
    )
    left = replace(_record(runtime="llamacpp"), observed={"runtime": "llamacpp"})
    right = replace(_record(runtime="vllm"), observed={"runtime": "vllm"})

    report = compare_runs(left, right, plan)

    assert not report.compatible
    assert report.issues[0].left == "<missing>"
    assert report.issues[0].right == "<missing>"
    assert report.issues[0].reason == "required block dimension is missing"


def test_explicit_none_is_distinct_from_a_missing_dimension() -> None:
    plan = ComparisonPlan(
        vary=frozenset({"runtime"}),
        block_by=frozenset({"observed.hardware_class"}),
    )
    left = replace(
        _record(runtime="llamacpp"),
        observed={"runtime": "llamacpp", "hardware_class": None},
    )
    right = replace(
        _record(runtime="vllm"),
        observed={"runtime": "vllm", "hardware_class": None},
    )

    report = compare_runs(left, right, plan)

    assert report.compatible


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


def test_evidence_models_detach_and_deeply_freeze_nested_values() -> None:
    config = {"kv": {"flags": ["fp8", "scaled"]}}
    treatment = TreatmentSpec(
        name="fp8",
        kind=TreatmentType.RUNTIME_FEATURE,
        config=config,
    )
    config["kv"]["flags"].append("mutated")

    assert treatment.config["kv"]["flags"] == ("fp8", "scaled")
    with pytest.raises(TypeError):
        treatment.config["new"] = "value"  # type: ignore[index]
    with pytest.raises(TypeError):
        treatment.config["kv"]["new"] = "value"  # type: ignore[index]


def test_run_record_evidence_is_detached_from_source_mappings() -> None:
    resolved = {"runtime": {"flags": ["--ctx-size", "4096"]}}
    observed = {"hardware": {"gpus": ["gpu-a"]}}
    record = RunRecord(
        study_name="immutability",
        run_id="one",
        requested=_run_spec(),
        resolved=resolved,
        observed=observed,
        status=RunStatus.COMPLETED,
        events=({"stage": "launch", "details": {"attempts": [1]}},),
        artifacts=({"path": "run.json", "hashes": ["abc"]},),
        provenance={"commits": ["deadbeef"]},
    )
    resolved["runtime"]["flags"].append("--mutated")
    observed["hardware"]["gpus"].append("gpu-b")

    assert record.resolved["runtime"]["flags"] == ("--ctx-size", "4096")
    assert record.observed["hardware"]["gpus"] == ("gpu-a",)
    assert record.events[0]["details"]["attempts"] == (1,)
    assert record.artifacts[0]["hashes"] == ("abc",)
    assert record.provenance["commits"] == ("deadbeef",)


def test_runtime_payloads_are_deeply_immutable() -> None:
    generation = {"stop": ["</s>"]}
    request = InferenceRequest(prompt="hello", generation=generation)
    batch = InferenceBatch(
        outputs=["world"],
        captures={"token_ids": [1, 2, 3]},
        metadata={"runtime": {"version": ["1"]}},
    )
    support = SupportReport(
        status="supported",
        evidence={"features": ["logprobs"]},
    )
    generation["stop"].append("mutated")

    assert request.generation["stop"] == ("</s>",)
    assert batch.outputs == ("world",)
    assert batch.captures["token_ids"] == (1, 2, 3)
    assert batch.metadata["runtime"]["version"] == ("1",)
    assert support.evidence["features"] == ("logprobs",)


def test_unsupported_live_objects_are_rejected_from_evidence() -> None:
    with pytest.raises(TypeError, match="unsupported evidence value type"):
        TreatmentSpec(
            name="bad",
            kind=TreatmentType.INSTRUMENTATION,
            config={"live_object": object()},
        )
