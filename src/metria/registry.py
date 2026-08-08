"""Explicit built-in registries for the first Metria execution CLI."""

from __future__ import annotations

import importlib.util
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum

from ._freeze import freeze_typed_mapping
from .measurements import TokenTrajectoryProtocol, TrajectoryAgreementAnalysis
from .models import StudySpec
from .protocols import MeasurementProtocol, PairwiseAnalysis, RuntimeAdapter
from .runtimes import LlamaCppAdapter, VLLMAdapter


class PluginKind(str, Enum):
    """Kinds of executable implementation registered by Metria."""

    RUNTIME = "runtime"
    MEASUREMENT = "measurement"
    ANALYSIS = "analysis"


class PluginAvailability(str, Enum):
    """Static availability conclusion before recipe-specific preflight."""

    AVAILABLE = "available"
    RECIPE_DEPENDENT = "recipe_dependent"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class PluginDescriptor:
    """Data-only description of one explicitly registered implementation."""

    name: str
    kind: PluginKind
    availability: PluginAvailability
    version: str | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("plugin descriptor name must not be empty")
        if self.version is not None and not self.version.strip():
            raise ValueError("plugin descriptor version must not be empty")
        if self.reason is not None and not self.reason.strip():
            raise ValueError("plugin descriptor reason must not be empty")

    def to_mapping(self) -> Mapping[str, str | None]:
        """Return a JSON-friendly descriptor without importing plugin code."""

        return {
            "name": self.name,
            "kind": self.kind.value,
            "availability": self.availability.value,
            "version": self.version,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class RegistryBundle:
    """Explicit implementation registries plus their data-only descriptors."""

    runtimes: Mapping[str, RuntimeAdapter]
    measurements: Mapping[str, MeasurementProtocol]
    analyses: Mapping[str, PairwiseAnalysis]
    descriptors: tuple[PluginDescriptor, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "runtimes", freeze_typed_mapping(self.runtimes))
        object.__setattr__(
            self,
            "measurements",
            freeze_typed_mapping(self.measurements),
        )
        object.__setattr__(self, "analyses", freeze_typed_mapping(self.analyses))
        object.__setattr__(self, "descriptors", tuple(self.descriptors))

        identities = [(item.kind, item.name) for item in self.descriptors]
        if len(identities) != len(set(identities)):
            raise ValueError("registry descriptors must have unique kind/name identities")

    def descriptor(self, kind: PluginKind, name: str) -> PluginDescriptor | None:
        """Return one descriptor by exact kind/name identity."""

        for descriptor in self.descriptors:
            if descriptor.kind is kind and descriptor.name == name:
                return descriptor
        return None


def _module_available(name: str) -> bool:
    """Probe a top-level optional module without importing it or raising on bad specs."""

    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def builtin_registry() -> RegistryBundle:
    """Return only first-party implementations shipped by the Metria package."""

    trajectory = TokenTrajectoryProtocol()
    trajectory_analysis = TrajectoryAgreementAnalysis()
    vllm_available = _module_available("vllm")
    descriptors = (
        PluginDescriptor(
            name="llamacpp",
            kind=PluginKind.RUNTIME,
            availability=PluginAvailability.RECIPE_DEPENDENT,
            reason=(
                "availability depends on recipe-local llama.cpp binaries and a local "
                "GGUF model path"
            ),
        ),
        PluginDescriptor(
            name="vllm",
            kind=PluginKind.RUNTIME,
            availability=(
                PluginAvailability.AVAILABLE
                if vllm_available
                else PluginAvailability.UNAVAILABLE
            ),
            reason=(
                None
                if vllm_available
                else "optional dependency 'vllm' is not installed in this environment"
            ),
        ),
        PluginDescriptor(
            name=trajectory.name,
            kind=PluginKind.MEASUREMENT,
            availability=PluginAvailability.AVAILABLE,
            version=trajectory.version,
        ),
        PluginDescriptor(
            name=trajectory_analysis.name,
            kind=PluginKind.ANALYSIS,
            availability=PluginAvailability.AVAILABLE,
            version=trajectory_analysis.version,
        ),
    )
    return RegistryBundle(
        runtimes={
            "llamacpp": LlamaCppAdapter(),
            "vllm": VLLMAdapter(),
        },
        measurements={trajectory.name: trajectory},
        analyses={trajectory_analysis.name: trajectory_analysis},
        descriptors=descriptors,
    )


def validate_study_availability(study: StudySpec, registry: RegistryBundle) -> None:
    """Fail before execution when a requested implementation is absent/unavailable."""

    for index, run in enumerate(study.runs):
        runtime_name = run.runtime.get("name")
        if not isinstance(runtime_name, str) or not runtime_name:
            raise ValueError(f"study run[{index}] runtime.name must be a non-empty string")
        if runtime_name not in registry.runtimes:
            raise ValueError(f"study run[{index}] uses unregistered runtime {runtime_name!r}")
        descriptor = registry.descriptor(PluginKind.RUNTIME, runtime_name)
        if descriptor is None:
            raise ValueError(f"runtime {runtime_name!r} has no registry descriptor")
        if descriptor.availability is PluginAvailability.UNAVAILABLE:
            suffix = f": {descriptor.reason}" if descriptor.reason else ""
            raise ValueError(f"runtime {runtime_name!r} is unavailable{suffix}")

        for measurement_name in run.measurements:
            if measurement_name not in registry.measurements:
                raise ValueError(
                    f"study run[{index}] uses unregistered measurement "
                    f"{measurement_name!r}"
                )
            measurement_descriptor = registry.descriptor(
                PluginKind.MEASUREMENT,
                measurement_name,
            )
            if measurement_descriptor is None:
                raise ValueError(
                    f"measurement {measurement_name!r} has no registry descriptor"
                )
            if measurement_descriptor.availability is PluginAvailability.UNAVAILABLE:
                raise ValueError(f"measurement {measurement_name!r} is unavailable")

    for analysis_name in study.comparison.analyses:
        if analysis_name not in registry.analyses:
            raise ValueError(f"comparison uses unregistered analysis {analysis_name!r}")
        descriptor = registry.descriptor(PluginKind.ANALYSIS, analysis_name)
        if descriptor is None:
            raise ValueError(f"analysis {analysis_name!r} has no registry descriptor")
        if descriptor.availability is PluginAvailability.UNAVAILABLE:
            raise ValueError(f"analysis {analysis_name!r} is unavailable")
