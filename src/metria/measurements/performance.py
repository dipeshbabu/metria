"""Method-aware impact from retained, authoritative invocation wall clocks."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from ..models import (
    MetricDefinition,
    MetricDirection,
    MetricSample,
    MetricSummary,
    RunRecord,
    RunStatus,
)
from ..protocols import MeasurementResult
from ..recipes import _json_value
from .trajectory import TokenTrajectoryProtocol

_NAME = "request_latency_seconds"
_DEFINITION = MetricDefinition(
    _NAME,
    "seconds",
    MetricDirection.LOWER_IS_BETTER,
    "metria.llamacpp_process_request_latency",
    "1",
)
_UNAVAILABLE = (
    "ttft",
    "inter_token_latency",
    "decode_throughput",
    "peak_device_memory",
    "kv_memory",
)


def measure_invocation_performance(record: RunRecord) -> MeasurementResult:
    """Measure the completed workload's cold-process request latency.

    Raw clocks remain in the run record. Each prompt is one trial with zero
    warmups and a fresh process, including loading and prompt evaluation. No
    decode-only, streaming, or memory measurements are inferred from these clocks.
    """
    unavailable: dict[str, Any] = {
        "schema": "metria.invocation_performance.v1",
        "available": False,
        "unsupported_metrics": list(_UNAVAILABLE),
    }
    if record.status is not RunStatus.COMPLETED:
        return MeasurementResult(
            evidence={**unavailable, "reason": "run did not complete"}
        )
    if record.requested.runtime.get("name") != "llamacpp":
        return MeasurementResult(
            evidence={
                **unavailable,
                "reason": "runtime has no supported authoritative timing source",
            }
        )
    invocations = record.observed.get("invocations")
    measurements = record.evidence.get("measurements", {})
    captures = (
        measurements.get(TokenTrajectoryProtocol.name, {})
        if isinstance(measurements, Mapping)
        else {}
    )
    if not isinstance(captures, Mapping):
        captures = {}
    prompts = captures.get("prompts")
    if (
        not isinstance(invocations, Sequence)
        or not isinstance(prompts, Sequence)
        or not prompts
        or len(invocations) != len(prompts)
    ):
        return MeasurementResult(
            evidence={
                **unavailable,
                "reason": "timing/workload evidence is missing or incomplete",
            }
        )
    samples = []
    workload = []
    for invocation, prompt in zip(invocations, prompts, strict=True):
        if not isinstance(invocation, Mapping) or not isinstance(prompt, Mapping):
            return MeasurementResult(
                evidence={**unavailable, "reason": "invalid timing/workload evidence"}
            )
        value = invocation.get("duration_seconds")
        fingerprint = prompt.get("prompt_sha256")
        token_count = prompt.get("token_count")
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            or invocation.get("returncode") != 0
        ):
            return MeasurementResult(
                evidence={
                    **unavailable,
                    "reason": "authoritative duration is missing or invalid",
                }
            )
        if (
            not isinstance(fingerprint, str)
            or len(fingerprint) != 64
            or any(char not in "0123456789abcdef" for char in fingerprint)
            or isinstance(token_count, bool)
            or not isinstance(token_count, int)
            or token_count <= 0
            or invocation.get("prompt_sha256") != fingerprint
        ):
            return MeasurementResult(
                evidence={
                    **unavailable,
                    "reason": "authoritative token/workload identity is incomplete",
                }
            )
        identity = {
            "prompt_id": prompt.get("id"),
            "prompt_sha256": fingerprint,
            "generation": invocation.get("generation", {}),
        }
        workload.append(identity)
        samples.append(
            MetricSample(
                float(value), metadata={"prompt_id": prompt.get("id"), "trial": 0}
            )
        )
    workload_digest = hashlib.sha256(
        json.dumps(
            _json_value(
                {"prompts": workload, "scenario": record.requested.scenario},
                path="performance.workload",
            ),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    methodology = {
        "source": "llamacpp.observed.invocations.duration_seconds",
        "clock": "monotonic",
        "scope": "process startup, model loading, prompt evaluation and generation",
        "warmup_trials": 0,
        "trials_per_prompt": 1,
        "isolation": "fresh process per prompt",
        "aggregation": "arithmetic mean across prompts",
        "workload_sha256": workload_digest,
        "sample_count": len(samples),
    }
    return MeasurementResult(
        metrics={
            _NAME: MetricSummary(
                definition=_DEFINITION,
                value=sum(sample.value / len(samples) for sample in samples),
                samples=tuple(samples),
                aggregation="mean",
                coverage=1.0,
            )
        },
        evidence={
            "schema": "metria.invocation_performance.v1",
            "available": True,
            "methodology": methodology,
            "unsupported_metrics": list(_UNAVAILABLE),
        },
    )


def compare_performance(
    reference: MeasurementResult, candidate: MeasurementResult, *, comparable: bool
) -> dict[str, Any]:
    """Calculate deltas only for complete, method-compatible verified evidence."""
    return compare_latency_results(
        reference,
        candidate,
        comparable=comparable,
        definition=_DEFINITION,
        limitations="One cold invocation per prompt; no statistical speedup claim.",
    )


def compare_latency_results(
    reference: MeasurementResult,
    candidate: MeasurementResult,
    *,
    comparable: bool,
    definition: MetricDefinition,
    limitations: str,
) -> dict[str, Any]:
    """Compare an explicitly selected latency method with matching workload evidence."""
    result: dict[str, Any] = {
        "schema": "metria.performance_impact.v1",
        "available": False,
        "unsupported_metrics": list(_UNAVAILABLE),
    }
    if not comparable:
        return {**result, "reason": "reference/candidate comparison gate did not pass"}
    left, right = reference.metrics.get(_NAME), candidate.metrics.get(_NAME)
    if left is None or right is None:
        return {**result, "reason": "authoritative request latency is unavailable"}
    methodology = reference.evidence.get("methodology")
    if (
        reference.evidence.get("available") is not True
        or candidate.evidence.get("available") is not True
        or not isinstance(methodology, Mapping)
        or not methodology
        or methodology != candidate.evidence.get("methodology")
        or reference.evidence.get("schema") != "metria.invocation_performance.v1"
        or candidate.evidence.get("schema") != "metria.invocation_performance.v1"
        or left.definition != definition
        or right.definition != definition
        or left.aggregation != "mean"
        or right.aggregation != "mean"
        or left.coverage != 1.0
        or right.coverage != 1.0
    ):
        return {
            **result,
            "reason": "performance units, method, trials, or workload differ",
        }
    values = (left.value, right.value)
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        for value in values
    ):
        return {**result, "reason": "performance values are invalid"}
    delta = right.value - left.value
    baseline_floor = 1e-9
    relative = delta / left.value if left.value > baseline_floor else None
    relative_reason = (
        "reference is zero or below timing resolution floor"
        if relative is None
        else None
    )
    if relative is not None and not math.isfinite(relative):
        relative = None
        relative_reason = "relative change exceeds finite numeric range"
    return {
        **result,
        "available": True,
        "metric": {
            "name": _NAME,
            "unit": definition.unit,
            "direction": definition.direction.value,
            "method": definition.method,
            "version": definition.version,
        },
        "methodology": _json_value(methodology, path="performance.methodology"),
        "reference": left.value,
        "candidate": right.value,
        "absolute_delta": delta,
        "relative_delta": relative,
        "relative_baseline_floor_seconds": baseline_floor,
        "relative_unavailable_reason": relative_reason,
        "direction": "improved"
        if delta < 0
        else "regressed"
        if delta > 0
        else "unchanged",
        "limitations": limitations,
    }
