from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from metria import (
    ComparisonPlan,
    MeasurementResult,
    MetricDefinition,
    MetricDirection,
    MetricSummary,
    RunSpec,
    RunStatus,
    StudySpec,
)
from metria.protocols import (
    CaptureRequest,
    InferenceBatch,
    InferenceRequest,
    SupportReport,
)
from metria.study_execution import StudyExecutionResult, execute_study


class FakeSession:
    def __init__(self) -> None:
        self.closed = False

    def infer(
        self,
        requests: Sequence[InferenceRequest],
        capture: Sequence[CaptureRequest] = (),
    ) -> InferenceBatch:
        del capture
        return InferenceBatch(outputs=tuple(request.prompt for request in requests))

    def reset(self, scope: str = "measurement") -> None:
        del scope

    def close(self) -> None:
        self.closed = True


class FakeAdapter:
    def __init__(self, name: str, *, fail_launch: bool = False) -> None:
        self.name = name
        self.fail_launch = fail_launch
        self.calls: list[str] = []

    def probe(
        self,
        spec: RunSpec,
        environment: Mapping[str, Any],
    ) -> SupportReport:
        del spec
        self.calls.append("probe")
        return SupportReport(
            status="supported",
            evidence={"hardware_class": environment.get("hardware_class")},
        )

    def resolve(
        self,
        spec: RunSpec,
        environment: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        del environment
        self.calls.append("resolve")
        return {"runtime": {"name": self.name}, "model": spec.model}

    def launch(
        self,
        resolved: Mapping[str, Any],
        environment: Mapping[str, Any],
    ) -> FakeSession:
        del resolved, environment
        self.calls.append("launch")
        if self.fail_launch:
            raise RuntimeError("synthetic launch failure")
        return FakeSession()

    def observe(self, session: Any) -> Mapping[str, Any]:
        assert isinstance(session, FakeSession)
        self.calls.append("observe")
        return {"runtime": self.name, "hardware_class": "gpu-a"}


class FakeMeasurement:
    name = "study.metric"
    version = "1"

    def __init__(self) -> None:
        self.configs: list[Mapping[str, Any]] = []

    def requirements(self, config: Mapping[str, Any]) -> tuple[CaptureRequest, ...]:
        del config
        return ()

    def execute(
        self,
        session: Any,
        scenario: Mapping[str, Any],
        config: Mapping[str, Any],
    ) -> MeasurementResult:
        del session, scenario
        self.configs.append(config)
        value = float(config.get("value", 1.0))
        return MeasurementResult(
            metrics={
                "score": MetricSummary(
                    definition=MetricDefinition(
                        name="score",
                        unit="units",
                        direction=MetricDirection.HIGHER_IS_BETTER,
                        method=self.name,
                        version=self.version,
                    ),
                    value=value,
                )
            },
            evidence={"value": value},
        )


def _run(runtime: str, *, model: str = "example/model") -> RunSpec:
    return RunSpec(
        model={"id": model},
        runtime={"name": runtime},
        scenario={"name": "decode"},
        measurements=(FakeMeasurement.name,),
    )


def _study(*runs: RunSpec) -> StudySpec:
    return StudySpec(
        name="cross-runtime-study",
        runs=tuple(runs),
        comparison=ComparisonPlan(
            vary=frozenset({"runtime"}),
            control=frozenset({"model", "scenario", "measurements"}),
            block_by=frozenset({"observed.hardware_class"}),
        ),
    )


def test_execute_study_routes_runs_and_builds_pairwise_comparisons() -> None:
    llama = FakeAdapter("llamacpp")
    vllm = FakeAdapter("vllm")
    measurement = FakeMeasurement()

    result = execute_study(
        _study(_run("llamacpp"), _run("vllm")),
        adapters={"llamacpp": llama, "vllm": vllm},
        measurements={measurement.name: measurement},
        measurement_configs={measurement.name: {"value": 3.0}},
        environment={"hardware_class": "gpu-a"},
    )

    assert isinstance(result, StudyExecutionResult)
    assert tuple(record.run_id for record in result.records) == ("run-0000", "run-0001")
    assert all(record.status is RunStatus.COMPLETED for record in result.records)
    assert tuple(record.metrics["score"].value for record in result.records) == (
        3.0,
        3.0,
    )
    assert len(result.comparisons) == 1
    comparison = result.comparisons[0]
    assert comparison.left_run_id == "run-0000"
    assert comparison.right_run_id == "run-0001"
    assert comparison.report.compatible
    assert comparison.report.comparable_metrics == ("score",)


def test_registry_errors_fail_before_any_run_starts() -> None:
    llama = FakeAdapter("llamacpp")
    measurement = FakeMeasurement()
    study = _study(_run("llamacpp"), _run("missing"))

    with pytest.raises(ValueError, match="unregistered runtime 'missing'"):
        execute_study(
            study,
            adapters={"llamacpp": llama},
            measurements={measurement.name: measurement},
            measurement_configs={},
            environment={"hardware_class": "gpu-a"},
        )

    assert llama.calls == []
    assert measurement.configs == []


def test_registry_key_must_match_implementation_name() -> None:
    measurement = FakeMeasurement()

    with pytest.raises(ValueError, match="does not match adapter.name"):
        execute_study(
            _study(_run("alias")),
            adapters={"alias": FakeAdapter("actual")},
            measurements={measurement.name: measurement},
            measurement_configs={},
            environment={},
        )

    with pytest.raises(ValueError, match="does not match measurement.name"):
        execute_study(
            _study(_run("llamacpp")),
            adapters={"llamacpp": FakeAdapter("llamacpp")},
            measurements={"alias": measurement},
            measurement_configs={},
            environment={},
        )


def test_multi_measurement_run_is_rejected_before_execution() -> None:
    adapter = FakeAdapter("llamacpp")
    measurement = FakeMeasurement()
    run = RunSpec(
        model={"id": "example/model"},
        runtime={"name": "llamacpp"},
        scenario={"name": "decode"},
        measurements=(measurement.name, "second.metric"),
    )

    with pytest.raises(ValueError, match="exactly one measurement"):
        execute_study(
            _study(run),
            adapters={"llamacpp": adapter},
            measurements={measurement.name: measurement},
            measurement_configs={},
            environment={},
        )

    assert adapter.calls == []


def test_experimental_failure_is_preserved_and_does_not_abort_later_run() -> None:
    broken = FakeAdapter("broken", fail_launch=True)
    healthy = FakeAdapter("healthy")
    measurement = FakeMeasurement()

    result = execute_study(
        _study(_run("broken"), _run("healthy")),
        adapters={"broken": broken, "healthy": healthy},
        measurements={measurement.name: measurement},
        measurement_configs={},
        environment={"hardware_class": "gpu-a"},
    )

    assert result.records[0].status is RunStatus.FAILED
    assert result.records[1].status is RunStatus.COMPLETED
    assert healthy.calls == ["probe", "resolve", "launch", "observe"]
    assert len(result.comparisons) == 1
    report = result.comparisons[0].report
    assert not report.compatible
    status_issue = next(issue for issue in report.issues if issue.dimension == "status")
    assert status_issue.reason == "both runs must be completed for direct comparison"


def test_controlled_dimension_mismatch_is_visible_at_study_level() -> None:
    measurement = FakeMeasurement()
    result = execute_study(
        _study(_run("llamacpp", model="model/a"), _run("vllm", model="model/b")),
        adapters={
            "llamacpp": FakeAdapter("llamacpp"),
            "vllm": FakeAdapter("vllm"),
        },
        measurements={measurement.name: measurement},
        measurement_configs={},
        environment={"hardware_class": "gpu-a"},
    )

    report = result.comparisons[0].report
    assert not report.compatible
    assert any(
        issue.dimension == "model" and issue.reason == "controlled dimension differs"
        for issue in report.issues
    )


def test_unknown_measurement_config_is_rejected_before_execution() -> None:
    adapter = FakeAdapter("llamacpp")
    measurement = FakeMeasurement()

    with pytest.raises(ValueError, match="unregistered measurements: unknown"):
        execute_study(
            _study(_run("llamacpp")),
            adapters={"llamacpp": adapter},
            measurements={measurement.name: measurement},
            measurement_configs={"unknown": {}},
            environment={},
        )

    assert adapter.calls == []
