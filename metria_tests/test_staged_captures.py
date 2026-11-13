"""Experimental (staged) captures run only behind an explicit override."""

from __future__ import annotations

from dataclasses import replace

from metria.execution import _capture_allowed
from metria.identity import SupportLevel
from metria.inspection import inspect_run_capabilities
from metria.integrations.staged_captures import STAGED_CAPTURE_CHECK
from metria.measurements import TokenDistributionProtocol
from metria.models import RunSpec

SPEC = RunSpec(
    model={},
    runtime={"name": "llamacpp"},
    scenario={},
    measurements=(TokenDistributionProtocol.name,),
)
OVERRIDE = replace(SPEC, trial_policy={"capability_overrides": [STAGED_CAPTURE_CHECK]})


def test_experimental_captures_need_the_staged_capture_override() -> None:
    assert _capture_allowed(SupportLevel.SUPPORTED, SPEC)
    assert not _capture_allowed(SupportLevel.EXPERIMENTAL, SPEC)
    assert _capture_allowed(SupportLevel.EXPERIMENTAL, OVERRIDE)
    for status in (SupportLevel.UNKNOWN, SupportLevel.UNSUPPORTED):
        assert not _capture_allowed(status, OVERRIDE)


def test_staged_measurements_are_blocked_in_preflight_without_override() -> None:
    blocked = inspect_run_capabilities(SPEC)
    assert STAGED_CAPTURE_CHECK in {item.name for item in blocked.blocking}
    allowed = inspect_run_capabilities(OVERRIDE)
    assert not allowed.blocking
    assert allowed.checks[STAGED_CAPTURE_CHECK]["override"]["applied"] is True


def test_qualified_measurements_are_unaffected() -> None:
    spec = replace(SPEC, measurements=("kv_fidelity.decode_time_trajectory",))
    assert (
        inspect_run_capabilities(spec).checks[STAGED_CAPTURE_CHECK]["applicable"]
        is False
    )
