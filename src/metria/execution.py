"""Failure-aware execution of one Metria run into a durable RunRecord."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

from .models import RunRecord, RunSpec, RunStatus
from .protocols import MeasurementProtocol, RuntimeAdapter, RuntimeSession, SupportReport

_EXECUTOR_NAME = "metria.execute_run"
_EXECUTOR_VERSION = "1"


def _message_fingerprint(exc: Exception) -> str:
    """Fingerprint an exception message without retaining potentially sensitive text."""

    return hashlib.sha256(str(exc).encode("utf-8")).hexdigest()


def _error_event(stage: str, exc: Exception) -> dict[str, Any]:
    """Return a privacy-conscious failure event for a lifecycle stage."""

    return {
        "stage": stage,
        "kind": "error",
        "error_type": type(exc).__name__,
        "message_sha256": _message_fingerprint(exc),
    }


def _record(
    *,
    study_name: str,
    run_id: str,
    spec: RunSpec,
    status: RunStatus,
    resolved: Mapping[str, Any],
    observed: Mapping[str, Any],
    metrics: Mapping[str, Any],
    evidence: Mapping[str, Any],
    events: list[Mapping[str, Any]],
    artifacts: tuple[Mapping[str, Any], ...],
    provenance: Mapping[str, Any],
) -> RunRecord:
    """Construct the immutable record at the end of an execution path."""

    return RunRecord(
        study_name=study_name,
        run_id=run_id,
        requested=spec,
        resolved=resolved,
        observed=observed,
        status=status,
        metrics=metrics,
        evidence=evidence,
        events=tuple(events),
        artifacts=artifacts,
        provenance=provenance,
    )


def execute_run(
    *,
    study_name: str,
    run_id: str,
    spec: RunSpec,
    adapter: RuntimeAdapter,
    measurement: MeasurementProtocol,
    measurement_config: Mapping[str, Any],
    environment: Mapping[str, Any],
) -> RunRecord:
    """Execute one requested run and return evidence even when execution fails.

    The lifecycle is deliberately narrow: probe support, resolve exact runtime
    state, launch a session, execute one named measurement protocol, observe
    applied runtime state, and close the session. Failures are represented in
    the returned ``RunRecord`` rather than silently discarded.

    Exception messages are not retained verbatim in lifecycle events because a
    third-party runtime or measurement may include prompt text or other
    sensitive values in an exception. Metria stores the exception type and a
    SHA-256 message fingerprint instead. ``SupportReport.reasons`` are retained
    because adapters explicitly designate them as preflight evidence.
    """

    if not study_name.strip():
        raise ValueError("study_name must not be empty")
    if not run_id.strip():
        raise ValueError("run_id must not be empty")

    measurement_name = measurement.name
    measurement_version = measurement.version
    if not measurement_name or not measurement_version:
        raise ValueError("measurement protocols must define non-empty name and version")

    events: list[Mapping[str, Any]] = []
    resolved: Mapping[str, Any] = {}
    observed: Mapping[str, Any] = {}
    metrics: Mapping[str, Any] = {}
    evidence: Mapping[str, Any] = {}
    artifacts: tuple[Mapping[str, Any], ...] = ()
    session: RuntimeSession | None = None
    measurement_completed = False
    observation_completed = False

    provenance: dict[str, Any] = {
        "executor": {"name": _EXECUTOR_NAME, "version": _EXECUTOR_VERSION},
        "measurement": {"name": measurement_name, "version": measurement_version},
        "environment": environment,
    }

    if measurement_name not in spec.measurements:
        events.append(
            {
                "stage": "preflight",
                "kind": "measurement_not_requested",
                "measurement": measurement_name,
            }
        )
        return _record(
            study_name=study_name,
            run_id=run_id,
            spec=spec,
            status=RunStatus.PREFLIGHT_FAILED,
            resolved=resolved,
            observed=observed,
            metrics=metrics,
            evidence=evidence,
            events=events,
            artifacts=artifacts,
            provenance=provenance,
        )

    try:
        support = adapter.probe(spec, environment)
    except Exception as exc:
        events.append(_error_event("probe", exc))
        return _record(
            study_name=study_name,
            run_id=run_id,
            spec=spec,
            status=RunStatus.PREFLIGHT_FAILED,
            resolved=resolved,
            observed=observed,
            metrics=metrics,
            evidence=evidence,
            events=events,
            artifacts=artifacts,
            provenance=provenance,
        )

    provenance["preflight"] = {
        "status": support.status,
        "reasons": support.reasons,
        "evidence": support.evidence,
    }
    if support.status != "supported":
        events.append(
            {
                "stage": "preflight",
                "kind": "unsupported",
                "reasons": support.reasons,
            }
        )
        return _record(
            study_name=study_name,
            run_id=run_id,
            spec=spec,
            status=RunStatus.PREFLIGHT_FAILED,
            resolved=resolved,
            observed=observed,
            metrics=metrics,
            evidence=evidence,
            events=events,
            artifacts=artifacts,
            provenance=provenance,
        )

    events.append({"stage": "preflight", "kind": "supported"})

    try:
        resolved = adapter.resolve(spec, environment)
    except Exception as exc:
        events.append(_error_event("resolve", exc))
        return _record(
            study_name=study_name,
            run_id=run_id,
            spec=spec,
            status=RunStatus.PREFLIGHT_FAILED,
            resolved=resolved,
            observed=observed,
            metrics=metrics,
            evidence=evidence,
            events=events,
            artifacts=artifacts,
            provenance=provenance,
        )

    try:
        session = adapter.launch(resolved, environment)
        events.append({"stage": "launch", "kind": "completed"})
    except Exception as exc:
        events.append(_error_event("launch", exc))
        return _record(
            study_name=study_name,
            run_id=run_id,
            spec=spec,
            status=RunStatus.FAILED,
            resolved=resolved,
            observed=observed,
            metrics=metrics,
            evidence=evidence,
            events=events,
            artifacts=artifacts,
            provenance=provenance,
        )

    failure_status: RunStatus | None = None
    try:
        try:
            result = measurement.execute(session, spec.scenario, measurement_config)
            measurement_completed = True
            metrics = result.metrics
            evidence = {"measurements": {measurement_name: result.evidence}}
            artifacts = result.artifacts
            events.append({"stage": "measurement", "kind": "completed"})
        except TimeoutError as exc:
            failure_status = RunStatus.TIMED_OUT
            events.append(_error_event("measurement", exc))
        except Exception as exc:
            failure_status = RunStatus.FAILED
            events.append(_error_event("measurement", exc))

        try:
            observed = adapter.observe(session)
            observation_completed = True
            events.append({"stage": "observe", "kind": "completed"})
        except Exception as exc:
            events.append(_error_event("observe", exc))
            if failure_status is None:
                failure_status = RunStatus.PARTIAL
    finally:
        try:
            session.close()
            events.append({"stage": "close", "kind": "completed"})
        except Exception as exc:
            events.append(_error_event("close", exc))
            if failure_status is None:
                failure_status = RunStatus.PARTIAL

    if failure_status is not None:
        status = failure_status
    elif measurement_completed and observation_completed:
        status = RunStatus.COMPLETED
    else:
        status = RunStatus.PARTIAL

    return _record(
        study_name=study_name,
        run_id=run_id,
        spec=spec,
        status=status,
        resolved=resolved,
        observed=observed,
        metrics=metrics,
        evidence=evidence,
        events=events,
        artifacts=artifacts,
        provenance=provenance,
    )
