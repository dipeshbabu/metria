"""Small, explicit data model for Metria studies and run evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ._freeze import freeze_mapping, freeze_typed_mapping


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

    def __post_init__(self) -> None:
        """Detach treatment configuration from caller-owned mutable state."""

        object.__setattr__(self, "config", freeze_mapping(self.config))


@dataclass(frozen=True)
class ComparisonPlan:
    """Declare study comparison dimensions and requested pairwise analyses."""

    vary: frozenset[str] = frozenset()
    control: frozenset[str] = frozenset()
    block_by: frozenset[str] = frozenset()
    analyses: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Normalize comparison roles and reject contradictory declarations."""

        object.__setattr__(self, "vary", frozenset(self.vary))
        object.__setattr__(self, "control", frozenset(self.control))
        object.__setattr__(self, "block_by", frozenset(self.block_by))
        object.__setattr__(self, "analyses", tuple(self.analyses))
        overlap = (
            (self.vary & self.control)
            | (self.vary & self.block_by)
            | (self.control & self.block_by)
        )
        if overlap:
            names = ", ".join(sorted(overlap))
            raise ValueError(f"comparison dimensions cannot overlap: {names}")
        if any(not isinstance(name, str) or not name.strip() for name in self.analyses):
            raise ValueError("comparison analyses must be non-empty strings")
        if len(set(self.analyses)) != len(self.analyses):
            raise ValueError("comparison analyses must not contain duplicates")


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

    def __post_init__(self) -> None:
        """Normalize a requested run into detached immutable values."""

        object.__setattr__(self, "model", freeze_mapping(self.model))
        object.__setattr__(self, "runtime", freeze_mapping(self.runtime))
        object.__setattr__(self, "scenario", freeze_mapping(self.scenario))
        object.__setattr__(self, "measurements", tuple(self.measurements))
        object.__setattr__(self, "treatments", tuple(self.treatments))
        object.__setattr__(self, "trial_policy", freeze_mapping(self.trial_policy))
        object.__setattr__(
            self,
            "environment_selector",
            freeze_mapping(self.environment_selector),
        )


@dataclass(frozen=True)
class StudySpec:
    """A study defines planned runs and the semantics of valid comparison."""

    name: str
    runs: tuple[RunSpec, ...]
    comparison: ComparisonPlan
    constants: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the study and detach retained study metadata."""

        object.__setattr__(self, "runs", tuple(self.runs))
        object.__setattr__(self, "constants", freeze_mapping(self.constants))
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))
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
        """Return the complete identity required for direct metric comparison."""

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

    def __post_init__(self) -> None:
        """Detach per-sample metadata from mutable caller state."""

        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))


@dataclass(frozen=True)
class MetricSummary:
    """Summary plus retained raw samples for a metric."""

    definition: MetricDefinition
    value: float
    samples: tuple[MetricSample, ...] = ()
    aggregation: str = "single"
    uncertainty: Mapping[str, float] = field(default_factory=dict)
    coverage: float | None = None

    def __post_init__(self) -> None:
        """Freeze retained samples and summary uncertainty metadata."""

        object.__setattr__(self, "samples", tuple(self.samples))
        object.__setattr__(self, "uncertainty", freeze_mapping(self.uncertainty))


@dataclass(frozen=True)
class RunRecord:
    """Evidence produced by executing one run specification.

    Requested state records user intent. Resolved state records what Metria
    selected before launch. Observed state records what the runtime and host
    actually reported after launch. Method-specific run evidence is kept in
    ``evidence`` rather than being mixed into runtime provenance or artifacts.
    """

    study_name: str
    run_id: str
    requested: RunSpec
    resolved: Mapping[str, Any]
    observed: Mapping[str, Any]
    status: RunStatus
    metrics: Mapping[str, MetricSummary] = field(default_factory=dict)
    evidence: Mapping[str, Any] = field(default_factory=dict)
    events: tuple[Mapping[str, Any], ...] = ()
    artifacts: tuple[Mapping[str, Any], ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Seal run evidence against mutation through retained aliases."""

        object.__setattr__(self, "resolved", freeze_mapping(self.resolved))
        object.__setattr__(self, "observed", freeze_mapping(self.observed))
        object.__setattr__(self, "metrics", freeze_typed_mapping(self.metrics))
        object.__setattr__(self, "evidence", freeze_mapping(self.evidence))
        object.__setattr__(
            self,
            "events",
            tuple(freeze_mapping(event) for event in self.events),
        )
        object.__setattr__(
            self,
            "artifacts",
            tuple(freeze_mapping(artifact) for artifact in self.artifacts),
        )
        object.__setattr__(self, "provenance", freeze_mapping(self.provenance))


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

    def __post_init__(self) -> None:
        """Normalize report collections into immutable values."""

        object.__setattr__(self, "issues", tuple(self.issues))
        object.__setattr__(self, "comparable_metrics", tuple(self.comparable_metrics))
        object.__setattr__(
            self,
            "incompatible_metrics",
            freeze_mapping(self.incompatible_metrics),
        )


def freeze_treatments(items: Sequence[TreatmentSpec]) -> tuple[TreatmentSpec, ...]:
    """Return treatment inputs as an immutable tuple."""

    return tuple(items)
