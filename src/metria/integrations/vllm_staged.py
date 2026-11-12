"""Built-in gate for STAGED vLLM treatments pending native qualification."""

from __future__ import annotations

from ..capabilities import GeometryInspection
from ..capability_checks import CapabilityCheck, CapabilityCheckResult
from ..identity import Capability, SupportLevel
from ..models import RunSpec
from ..runtimes.vllm_treatments import (
    CAPABILITY_CHECK,
    QUALIFICATION,
    STAGED_TREATMENTS,
    is_staged_treatment,
)


def _check_run(
    spec: RunSpec, geometry: GeometryInspection, experimental_override: bool
) -> CapabilityCheckResult | None:
    del geometry
    names = sorted(
        treatment.name
        for treatment in spec.treatments
        if is_staged_treatment(treatment)
    )
    if not names:
        return None
    evidence = {
        "qualification": QUALIFICATION,
        "treatments": tuple(names),
        "runtime": spec.runtime.get("name"),
        "experimental_override": experimental_override,
    }
    if spec.runtime.get("name") != "vllm":
        return CapabilityCheckResult(
            Capability(
                name=CAPABILITY_CHECK,
                status=SupportLevel.UNSUPPORTED,
                reasons=("staged vLLM treatments require runtime.name='vllm'",),
                evidence=evidence,
            )
        )
    return CapabilityCheckResult(
        Capability(
            name=CAPABILITY_CHECK,
            status=SupportLevel.EXPERIMENTAL,
            reasons=(
                "staged vLLM treatments ("
                + ", ".join(names)
                + ") have no retained native qualification; list "
                f"{CAPABILITY_CHECK!r} in trial_policy.capability_overrides to run them",
            ),
            evidence={
                **evidence,
                "supported_treatments": tuple(sorted(STAGED_TREATMENTS)),
            },
        )
    )


VLLM_STAGED_TREATMENT_CHECK = CapabilityCheck(
    name=CAPABILITY_CHECK,
    evaluate=_check_run,
    supports_experimental_override=True,
)
