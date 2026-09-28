"""Loaded-engine call latency from the qualified sequential prefix workload."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from ..models import MetricDefinition, MetricDirection, RunRecord, RunStatus
from ..protocols import MeasurementResult
from ..recipes import _json_value
from .performance import compare_latency_results
from .prefix_workload import LATENCY_METHOD, WORKLOAD_METHOD
from .trajectory import TokenTrajectoryProtocol

DEFINITION = MetricDefinition(
    "request_latency_seconds",
    "seconds",
    MetricDirection.LOWER_IS_BETTER,
    LATENCY_METHOD,
    "1",
)


def measure_prefix_performance(record: RunRecord) -> MeasurementResult:
    unavailable = MeasurementResult(
        evidence={"schema": "metria.invocation_performance.v1", "available": False}
    )
    if record.status is not RunStatus.COMPLETED:
        return unavailable
    measurements = record.evidence.get("measurements", {})
    capture = (
        measurements.get(TokenTrajectoryProtocol.name, {})
        if isinstance(measurements, Mapping)
        else {}
    )
    workload = capture.get("workload", {}) if isinstance(capture, Mapping) else {}
    if not isinstance(workload, Mapping) or workload.get("method") != WORKLOAD_METHOD:
        return unavailable
    prompts = capture.get("prompts")
    metric = record.metrics.get("request_latency_seconds")
    if (
        metric is None
        or not isinstance(prompts, Sequence)
        or not prompts
        or len(metric.samples) != len(prompts)
    ):
        return unavailable
    if any(
        not isinstance(row, Mapping) or not isinstance(row.get("prompt_sha256"), str)
        for row in prompts
    ):
        return unavailable
    identity = {
        "prompts": [
            {"id": row["id"], "prompt_sha256": row["prompt_sha256"]} for row in prompts
        ],
        "scenario": record.requested.scenario,
    }
    workload_sha256 = hashlib.sha256(
        json.dumps(
            _json_value(identity, path="prefix.performance.workload"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    methodology = {
        "source": "sequential session.infer calls",
        "runtime_hardware": record.observed.get("hardware"),
        "clock": workload.get("clock"),
        "scope": workload.get("timing_boundary"),
        "excluded": workload.get("timing_excludes"),
        "warmup_trials": workload.get("warmup_trials"),
        "trials_per_prompt": workload.get("measured_trials"),
        "isolation": workload.get("isolation"),
        "schedule": workload.get("schedule"),
        "aggregation": "arithmetic mean across complete measured request calls",
        "workload_sha256": workload_sha256,
        "sample_count": len(metric.samples),
    }
    return MeasurementResult(
        metrics={"request_latency_seconds": metric},
        evidence={
            "schema": "metria.invocation_performance.v1",
            "available": True,
            "methodology": methodology,
        },
    )


def compare_prefix_performance(
    reference: RunRecord, candidate: RunRecord, comparable: bool
) -> dict[str, Any]:
    result = compare_latency_results(
        measure_prefix_performance(reference),
        measure_prefix_performance(candidate),
        comparable=comparable,
        definition=DEFINITION,
        limitations="Sequential synchronous request calls after engine warmup; includes prompt evaluation and generation, excludes engine startup and cache reset. No streaming TTFT, serving-throughput, or statistical speedup claim.",
    )
    return {**result, "label": "Loaded-engine request latency"}
