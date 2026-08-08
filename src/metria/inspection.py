"""Shared capability inspection and preflight policy for requested runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .capabilities import evaluate_turboquant_kv_capability, inspect_model_geometry
from .identity import Capability, CapabilitySet, SupportLevel
from .models import RunSpec, TreatmentSpec, TreatmentType

_TURBO_CAPABILITY = "turboquant.kv_cache.geometry"
_SUPPORTED_OVERRIDES = frozenset({_TURBO_CAPABILITY})
_KV_TREATMENT_NAMES = frozenset(
    {"kv_cache", "llamacpp.kv_cache", "turboquant.kv_cache"}
)


@dataclass(frozen=True)
class PreflightCapabilityResult:
    """Capability conclusions plus any rules that must block execution."""

    capabilities: CapabilitySet
    blocking: tuple[Capability, ...] = ()


def _override_names(spec: RunSpec) -> frozenset[str]:
    raw = spec.trial_policy.get("capability_overrides", ())
    if isinstance(raw, str) or not isinstance(raw, Sequence):
        raise TypeError("trial_policy.capability_overrides must be a sequence of names")
    names = tuple(raw)
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("capability override names must be non-empty strings")
    return frozenset(names)


def _kv_treatment(spec: RunSpec) -> TreatmentSpec | None:
    matches = [
        treatment
        for treatment in spec.treatments
        if treatment.name in _KV_TREATMENT_NAMES
        and treatment.kind is TreatmentType.RUNTIME_FEATURE
    ]
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(
            "only one KV-cache runtime treatment may be capability-checked"
        )
    return matches[0]


def _unknown_override_capability(overrides: frozenset[str]) -> Capability | None:
    unknown = tuple(sorted(overrides - _SUPPORTED_OVERRIDES))
    if not unknown:
        return None
    return Capability(
        name="metria.capability_overrides",
        status=SupportLevel.UNKNOWN,
        reasons=("unrecognized capability override names: " + ", ".join(unknown),),
        evidence={
            "requested": tuple(sorted(overrides)),
            "recognized": tuple(sorted(overrides & _SUPPORTED_OVERRIDES)),
            "unrecognized": unknown,
        },
    )


def inspect_run_capabilities(spec: RunSpec) -> PreflightCapabilityResult:
    """Inspect a run without launching a runtime and return fail-closed blockers."""

    geometry = inspect_model_geometry(spec.model)
    capabilities: list[Capability] = [geometry.capability]
    blocking: list[Capability] = []

    overrides = _override_names(spec)
    unknown_override = _unknown_override_capability(overrides)
    if unknown_override is not None:
        capabilities.append(unknown_override)
        blocking.append(unknown_override)

    treatment = _kv_treatment(spec)
    if treatment is None:
        return PreflightCapabilityResult(
            capabilities=CapabilitySet(tuple(capabilities)),
            blocking=tuple(blocking),
        )

    override = _TURBO_CAPABILITY in overrides
    turbo = evaluate_turboquant_kv_capability(
        spec.model,
        treatment.config,
        experimental_override=override,
    )
    capabilities.append(turbo)

    active = bool(turbo.evidence.get("active"))
    if active:
        if turbo.status in {SupportLevel.UNSUPPORTED, SupportLevel.UNKNOWN}:
            blocking.append(turbo)
        elif turbo.status is SupportLevel.EXPERIMENTAL and not override:
            blocking.append(turbo)

    return PreflightCapabilityResult(
        capabilities=CapabilitySet(tuple(capabilities)),
        blocking=tuple(blocking),
    )


def capability_inspection_to_mapping(
    result: PreflightCapabilityResult,
) -> Mapping[str, Any]:
    """Return a JSON-friendly capability inspection representation."""

    return {
        "capabilities": result.capabilities.to_mapping(),
        "blocking": tuple(capability.name for capability in result.blocking),
        "allowed": not result.blocking,
    }
