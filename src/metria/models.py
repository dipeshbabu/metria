"""Small, explicit data model for Metria studies and run evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TreatmentType(str, Enum):
    """How a study changes the system under test."""

    MODEL_TRANSFORMATION = "model_transformation"
    RUNTIME_FEATURE = "runtime_feature"
    EXECUTION_POLICY = "execution_policy"
    INSTRUMENTATION = "instrumentation"


class MetricDirection(str, Enum):
    """Whether a larger or smaller metric is preferable."""

    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"
    TARGET_IS_BEST = "target_is_best"
    DESCRIPTIVE = "descriptive"


class RunStatus(str, Enum):
    """Lifecycle states retained as evidence in a run record."""

    PLANNED = "planned"
    PREFLIGHT_FAILED = "preflight_failed"
    RUNNING = "running"
    PARTIAL = "partial"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"


@dataclass(frozen=True)
class TreatmentSpec:
    """One deliberate treatment applied to a system under test."""

    name: str
    kind: TreatmentType
    config: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ComparisonPlan:
    """Declare which study dimensions may vary and which must be controlled."""

    vary: frozenset[str] = frozenset()
    control: frozenset[str] = frozenset()
    block_by: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        overlap = (
            (self.vary & self.control)
            | (self.vary & self.block_by)
            | (self.control & self.block_by)
        )
        if overlap:
            names = ", ".join(sorted(overlap))
            raise ValueError(f"comparison dimensions cannot overlap: {names}")


@dataclass(frozen=True)
class RunSpec:
    """Requested configuration for one execution within a study."""

    model: Mapping[str, Any]
    runtime: Mapping[str, Any]
    scenario: Mapping[str, Any]
    measurements: tuple[str, ...]
    treatments: tuple[TreatmentSpec, ...] = ()
    trial_policy: Mapping[str, Any] = field(default_factory=dict)
    environment_selector: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StudySpec:
    """A study defines planned runs and the semantics of valid comparison."""

    name: str
    runs: tuple[RunSpec, ...]
    comparison: ComparisonPlan
    constants: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("study name must not be empty")
        if not self.runs:
            raise ValueError("study must contain at least one run")


@dataclass(frozen=True)
class MetricDefinition:
    """Identity and semantics of a measured quantity."""

    name: str
    unit: str
    direction: MetricDirection
    method: str
    version: str = "1"

    @property
    def identity(self) -> tuple[str, str, str, str, str]:
        return (
            self.name,
            self.unit,
            self.direction.value,
            self.method,
            self.version,
        )


@dataclass(frozen=True)
class MetricSample:
    """One raw observation for a metric."""

    value: float
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MetricSummary:
    """Summary plus retained raw samples for a metric."""

    definition: MetricDefinition
    value: float
    samples: tuple[MetricSample, ...] = ()
    aggregation: str = "single"
    uncertainty: Mapping[str, float] = field(default_factory=dict)
    coverage: float | None = None


@dataclass(frozen=True)
class RunRecord:
    """Evidence produced by executing one run specification.

    Requested state records user intent. Resolved state records what Metria
    selected before launch. Observed state records what the runtime and host
    actually reported after launch. These are kept separate intentionally.
    """

    study_name: str
    run_id: str
    requested: RunSpec
    resolved: Mapping[str, Any]
    observed: Mapping[str, Any]
    status: RunStatus
    metrics: Mapping[str, MetricSummary] = field(default_factory=dict)
    events: tuple[Mapping[str, Any], ...] = ()
    artifacts: tuple[Mapping[str, Any], ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CompatibilityIssue:
    """One reason two run records are not directly comparable."""

    dimension: str
    left: Any
    right: Any
    reason: str


@dataclass(frozen=True)
class CompatibilityReport:
    """Study-level and metric-level comparison result."""

    compatible: bool
    issues: tuple[CompatibilityIssue, ...] = ()
    comparable_metrics: tuple[str, ...] = ()
    incompatible_metrics: Mapping[str, str] = field(default_factory=dict)


def freeze_treatments(items: Sequence[TreatmentSpec]) -> tuple[TreatmentSpec, ...]:
    """Return treatment inputs as an immutable tuple."""

    return tuple(items)
