"""Provisional runtime and measurement boundaries for Metria."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from .models import MetricSummary, RunSpec


@dataclass(frozen=True)
class SupportReport:
    """Evidence-backed support state for a requested run."""

    status: str
    reasons: tuple[str, ...] = ()
    evidence: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CaptureRequest:
    """Semantic request for inference evidence required by a measurement."""

    kind: str
    options: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InferenceRequest:
    """Runtime-neutral request payload."""

    prompt: str
    generation: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InferenceBatch:
    """Runtime-neutral inference result plus captured evidence."""

    outputs: tuple[str, ...]
    captures: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


class RuntimeSession(Protocol):
    """Prepared runtime used by measurement protocols."""

    def infer(
        self,
        requests: Sequence[InferenceRequest],
        capture: Sequence[CaptureRequest] = (),
    ) -> InferenceBatch: ...

    def reset(self, scope: str = "measurement") -> None: ...

    def close(self) -> None: ...


class RuntimeAdapter(Protocol):
    """Resolve, launch, and observe an inference runtime."""

    def probe(
        self,
        spec: RunSpec,
        environment: Mapping[str, Any],
    ) -> SupportReport: ...

    def resolve(self, spec: RunSpec) -> Mapping[str, Any]: ...

    def launch(
        self,
        resolved: Mapping[str, Any],
        environment: Mapping[str, Any],
    ) -> RuntimeSession: ...

    def observe(self, session: RuntimeSession) -> Mapping[str, Any]: ...


class MeasurementProtocol(Protocol):
    """A named measurement method with explicit evidence requirements."""

    def requirements(self, config: Mapping[str, Any]) -> tuple[CaptureRequest, ...]: ...

    def execute(
        self,
        session: RuntimeSession,
        scenario: Mapping[str, Any],
        config: Mapping[str, Any],
    ) -> Mapping[str, MetricSummary]: ...
