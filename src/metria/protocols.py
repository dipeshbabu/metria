"""Provisional runtime and measurement boundaries for Metria."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from ._freeze import freeze_mapping, freeze_typed_mapping
from .models import MetricSummary, RunSpec


@dataclass(frozen=True)
class SupportReport:
    """Evidence-backed support state for a requested run."""

    status: str
    reasons: tuple[str, ...] = ()
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Detach support evidence from mutable adapter-owned state."""

        object.__setattr__(self, "reasons", tuple(self.reasons))
        object.__setattr__(self, "evidence", freeze_mapping(self.evidence))


@dataclass(frozen=True)
class CaptureRequest:
    """Semantic request for inference evidence required by a measurement."""

    kind: str
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Freeze capture options so measurement requirements stay stable."""

        object.__setattr__(self, "options", freeze_mapping(self.options))


@dataclass(frozen=True)
class InferenceRequest:
    """Runtime-neutral request payload."""

    prompt: str
    generation: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Freeze generation settings before the request reaches a runtime."""

        object.__setattr__(self, "generation", freeze_mapping(self.generation))


@dataclass(frozen=True)
class InferenceBatch:
    """Runtime-neutral inference result plus captured evidence."""

    outputs: tuple[str, ...]
    captures: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Seal returned outputs, captures, and runtime metadata."""

        object.__setattr__(self, "outputs", tuple(self.outputs))
        object.__setattr__(self, "captures", freeze_mapping(self.captures))
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))


@dataclass(frozen=True)
class MeasurementResult:
    """Metrics plus immutable evidence produced by a measurement method.

    Metrics are compact numerical summaries suitable for comparison and
    reporting. Evidence retains the method-specific observations needed to
    reproduce or derive those summaries. Large binary payloads should be stored
    externally and represented here by artifact references instead of live
    objects.
    """

    metrics: Mapping[str, MetricSummary] = field(default_factory=dict)
    evidence: Mapping[str, Any] = field(default_factory=dict)
    artifacts: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        """Detach all retained measurement output from mutable caller state."""

        object.__setattr__(self, "metrics", freeze_typed_mapping(self.metrics))
        object.__setattr__(self, "evidence", freeze_mapping(self.evidence))
        object.__setattr__(
            self,
            "artifacts",
            tuple(freeze_mapping(artifact) for artifact in self.artifacts),
        )


class RuntimeSession(Protocol):
    """Prepared runtime used by measurement protocols."""

    def infer(
        self,
        requests: Sequence[InferenceRequest],
        capture: Sequence[CaptureRequest] = (),
    ) -> InferenceBatch:
        """Execute requests and return outputs plus requested captures."""
        ...

    def reset(self, scope: str = "measurement") -> None:
        """Reset runtime state at the requested isolation scope."""
        ...

    def close(self) -> None:
        """Release all resources owned by the runtime session."""
        ...


class RuntimeAdapter(Protocol):
    """Resolve, launch, and observe an inference runtime."""

    def probe(
        self,
        spec: RunSpec,
        environment: Mapping[str, Any],
    ) -> SupportReport:
        """Report whether the requested run is supported in this environment."""
        ...

    def resolve(
        self,
        spec: RunSpec,
        environment: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Resolve user intent into an exact pre-launch runtime specification."""
        ...

    def launch(
        self,
        resolved: Mapping[str, Any],
        environment: Mapping[str, Any],
    ) -> RuntimeSession:
        """Launch a runtime session from the resolved specification."""
        ...

    def observe(self, session: RuntimeSession) -> Mapping[str, Any]:
        """Collect post-launch evidence of what the runtime actually applied."""
        ...


class MeasurementProtocol(Protocol):
    """A named, versioned measurement method with explicit evidence requirements."""

    name: str
    version: str

    def requirements(self, config: Mapping[str, Any]) -> tuple[CaptureRequest, ...]:
        """Describe the inference evidence required by this measurement."""
        ...

    def execute(
        self,
        session: RuntimeSession,
        scenario: Mapping[str, Any],
        config: Mapping[str, Any],
    ) -> MeasurementResult:
        """Execute the method and return numerical summaries plus evidence."""
        ...
