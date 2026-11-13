"""Built-in gate for STAGED runtime captures pending native qualification."""

from __future__ import annotations

from ..capabilities import GeometryInspection
from ..capability_checks import CapabilityCheck, CapabilityCheckResult
from ..identity import Capability, SupportLevel
from ..models import RunSpec

STAGED_CAPTURE_CHECK = "metria.staged_captures"
# Measurements whose required captures are reported experimental by runtimes.
_STAGED_MEASUREMENTS = frozenset({"metria.decode_time_top_logprobs"})


def _check_run(
    spec: RunSpec, geometry: GeometryInspection, experimental_override: bool
) -> CapabilityCheckResult | None:
    del geometry
    staged = sorted(set(spec.measurements) & _STAGED_MEASUREMENTS)
    if not staged:
        return None
    return CapabilityCheckResult(
        Capability(
            name=STAGED_CAPTURE_CHECK,
            status=SupportLevel.EXPERIMENTAL,
            reasons=(
                "staged measurements ("
                + ", ".join(staged)
                + ") require experimental runtime captures; list "
                f"{STAGED_CAPTURE_CHECK!r} in trial_policy.capability_overrides to run them",
            ),
            evidence={
                "qualification": "staged",
                "measurements": tuple(staged),
                "experimental_override": experimental_override,
            },
        )
    )


STAGED_CAPTURE_CHECK_RULE = CapabilityCheck(
    name=STAGED_CAPTURE_CHECK,
    evaluate=_check_run,
    supports_experimental_override=True,
)
