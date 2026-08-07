"""Study-level orchestration across registered runtimes and measurements."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations
from typing import Any

from .comparison import compare_runs
from .execution import execute_run
from .models import CompatibilityReport, RunRecord, RunSpec, StudySpec
from .protocols import MeasurementProtocol, RuntimeAdapter


@dataclass(frozen=True)
class StudyPairComparison:
    """Compatibility result for one deterministic pair of study run records."""

    left_run_id: str
    right_run_id: str
    report: CompatibilityReport


@dataclass(frozen=True)
class StudyExecutionResult:
    """All run records and pairwise compatibility results for one study."""

    study: StudySpec
    records: tuple[RunRecord, ...]
    comparisons: tuple[StudyPairComparison, ...]

    def __post_init__(self) -> None:
        """Normalize result collections into immutable tuples."""

        object.__setattr__(self, "records", tuple(self.records))
        object.__setattr__(self, "comparisons", tuple(self.comparisons))


def _runtime_name(spec: RunSpec, *, index: int) -> str:
    """Return the explicit runtime registry name for one requested run."""

    value = spec.runtime.get("name")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"study run[{index}] runtime.name must be a non-empty string")
    return value


def _measurement_name(spec: RunSpec, *, index: int) -> str:
    """Return the one measurement supported by the initial study executor."""

    if len(spec.measurements) != 1:
        raise ValueError(
            f"study run[{index}] must request exactly one measurement; "
            "multi-measurement scheduling is not implemented yet"
        )
    value = spec.measurements[0]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"study run[{index}] measurement name must be a non-empty string"
        )
    return value


def _validate_registries(
    study: StudySpec,
    adapters: Mapping[str, RuntimeAdapter],
    measurements: Mapping[str, MeasurementProtocol],
) -> tuple[tuple[str, str], ...]:
    """Validate every run route before executing any experiment work."""

    routes: list[tuple[str, str]] = []
    for key, adapter in adapters.items():
        if not isinstance(key, str) or not key:
            raise ValueError("runtime registry keys must be non-empty strings")
        if adapter.name != key:
            raise ValueError(
                f"runtime registry key {key!r} does not match adapter.name "
                f"{adapter.name!r}"
            )
    for key, measurement in measurements.items():
        if not isinstance(key, str) or not key:
            raise ValueError("measurement registry keys must be non-empty strings")
        if measurement.name != key:
            raise ValueError(
                f"measurement registry key {key!r} does not match measurement.name "
                f"{measurement.name!r}"
            )

    for index, spec in enumerate(study.runs):
        runtime_name = _runtime_name(spec, index=index)
        measurement_name = _measurement_name(spec, index=index)
        if runtime_name not in adapters:
            raise ValueError(
                f"study run[{index}] requires unregistered runtime {runtime_name!r}"
            )
        if measurement_name not in measurements:
            raise ValueError(
                f"study run[{index}] requires unregistered measurement "
                f"{measurement_name!r}"
            )
        routes.append((runtime_name, measurement_name))
    return tuple(routes)


def execute_study(
    study: StudySpec,
    *,
    adapters: Mapping[str, RuntimeAdapter],
    measurements: Mapping[str, MeasurementProtocol],
    measurement_configs: Mapping[str, Mapping[str, Any]],
    environment: Mapping[str, Any],
) -> StudyExecutionResult:
    """Execute every run in a study and retain all outcomes for comparison.

    Registry/configuration mistakes are validated before the first run starts so
    a programming error does not create a half-executed study. Once execution
    begins, experimental failures are preserved as ``RunRecord`` values and do
    not abort later runs.

    The initial study executor uses one shared environment mapping for all runs
    and supports exactly one measurement per ``RunSpec``. These constraints are
    explicit so future environment placement and multi-measurement scheduling
    can be added without silently changing current study semantics.
    """

    routes = _validate_registries(study, adapters, measurements)
    unknown_configs = sorted(set(measurement_configs) - set(measurements))
    if unknown_configs:
        raise ValueError(
            "measurement_configs contains unregistered measurements: "
            + ", ".join(unknown_configs)
        )

    records: list[RunRecord] = []
    for index, (spec, route) in enumerate(zip(study.runs, routes, strict=True)):
        runtime_name, measurement_name = route
        record = execute_run(
            study_name=study.name,
            run_id=f"run-{index:04d}",
            spec=spec,
            adapter=adapters[runtime_name],
            measurement=measurements[measurement_name],
            measurement_config=measurement_configs.get(measurement_name, {}),
            environment=environment,
        )
        records.append(record)

    pairwise: list[StudyPairComparison] = []
    for left, right in combinations(records, 2):
        pairwise.append(
            StudyPairComparison(
                left_run_id=left.run_id,
                right_run_id=right.run_id,
                report=compare_runs(left, right, study.comparison),
            )
        )

    return StudyExecutionResult(
        study=study,
        records=tuple(records),
        comparisons=tuple(pairwise),
    )
