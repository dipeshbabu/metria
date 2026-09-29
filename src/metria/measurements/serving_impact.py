"""Compare complete streaming measurements without mixing clocks or boundaries."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from ..models import (
    MetricSummary,
    RunRecord,
    RunStatus,
)
from ..protocols import MeasurementResult
from ..recipes import _json_value
from .performance import compare_latency_results
from .serving_metrics import DEFINITIONS as SOURCES
from .serving_metrics import measure_serving
from .serving_schema import DEFINITIONS, NAME, SCHEMA, VERSION
from .serving_workload import WORKLOAD_METHOD
from .trajectory import TokenTrajectoryProtocol, _capture_rows


def serving_result(record: RunRecord) -> MeasurementResult:
    unavailable = MeasurementResult(evidence={"available": False})
    if record.status is not RunStatus.COMPLETED:
        return unavailable
    capture = record.evidence.get("measurements", {}).get(
        TokenTrajectoryProtocol.name, {}
    )
    workload = capture.get("workload", {})
    if workload.get("method") != WORKLOAD_METHOD:
        return unavailable
    try:
        prompts = _capture_rows(MeasurementResult(evidence=capture))
        trials = workload["measured_trials"]
        if type(trials) is not int or trials < 1 or len(prompts) % trials:
            return unavailable
        measured = measure_serving(
            workload["batches"],
            workload["memory"],
            prompts_per_trial=len(prompts) // trials,
        )
        if len(workload["batches"]) != trials:
            return unavailable
    except (KeyError, TypeError, ValueError):
        return unavailable
    metrics = {
        key: metric
        for key, metric in measured.metrics.items()
        if record.metrics.get(key) == metric
    }
    identity = {
        "prompts": [
            {key: row[key] for key in ("id", "prompt_sha256")} for row in prompts
        ],
        "scenario": record.requested.scenario,
    }
    digest = hashlib.sha256(
        json.dumps(
            _json_value(identity, path="serving.workload"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    methodology = {
        "workload_sha256": digest,
        "runtime_hardware": record.observed.get("hardware"),
        "clock": "perf_counter_ns",
        "boundaries": measured.evidence["boundaries"],
        **{
            key: workload.get(key)
            for key in (
                "method",
                "warmup_trials",
                "measured_trials",
                "isolation",
                "schedule",
            )
        },
    }
    return MeasurementResult(
        metrics=metrics,
        evidence={
            "schema": "metria.invocation_performance.v1",
            "available": True,
            "methodology": methodology,
        },
    )


def compare_serving(
    reference: RunRecord, candidate: RunRecord, comparable: bool
) -> dict[str, Any]:
    left, right = serving_result(reference), serving_result(candidate)
    result = compare_latency_results(
        left,
        right,
        comparable=comparable,
        definition=SOURCES["request_latency_seconds"],
        limitations="Local AsyncLLM serving API after warmup; excludes engine startup, resets and HTTP/network overhead. Single-device observations; no statistical speedup claim.",
    )
    compatible = (
        comparable
        and left.evidence.get("available") is True
        and right.evidence.get("available") is True
        and left.evidence.get("methodology") == right.evidence.get("methodology")
    )
    rows = {}
    for name, definition in SOURCES.items():
        a, b = left.metrics.get(name), right.metrics.get(name)
        if (
            not compatible
            or a is None
            or b is None
            or a.definition != definition
            or b.definition != definition
            or a.coverage != 1.0
            or b.coverage != 1.0
            or a.aggregation != b.aggregation
        ):
            rows[name] = {
                "available": False,
                "reason": "complete compatible native measurements are unavailable",
            }
            continue
        delta = b.value - a.value
        ratio = b.value / a.value if a.value > 1e-9 else None
        if not math.isfinite(delta):
            rows[name] = {
                "available": False,
                "reason": "difference exceeds finite numeric range",
            }
            continue
        rows[name] = {
            "available": True,
            "reference": a.value,
            "candidate": b.value,
            "absolute_delta": delta,
            "ratio": ratio if ratio is None or math.isfinite(ratio) else None,
            "unit": definition.unit,
            "method": definition.method,
            "version": definition.version,
            "direction": definition.direction.value,
            "aggregation": a.aggregation,
            "coverage": 1.0,
        }
    return {
        **result,
        "label": "Local serving request latency",
        "metrics": rows,
        "unsupported_metrics": [
            name for name, row in rows.items() if not row["available"]
        ],
        "conditions": {
            "reference_concurrency": reference.requested.trial_policy.get(
                "concurrency"
            ),
            "candidate_concurrency": candidate.requested.trial_policy.get(
                "concurrency"
            ),
        },
    }


class ServingImpactAnalysis:
    name = NAME
    version = VERSION

    def analyze(self, left: RunRecord, right: RunRecord) -> MeasurementResult:
        result = compare_serving(left, right, True)
        metrics = {}
        for name, row in result["metrics"].items():
            if not row["available"]:
                continue
            for suffix in ("candidate", "ratio"):
                value = row[suffix]
                if value is not None:
                    key = f"{name}_{suffix}"
                    metrics[key] = MetricSummary(
                        DEFINITIONS[key], value, aggregation="derived", coverage=1.0
                    )
        return MeasurementResult(
            metrics=metrics,
            evidence={
                "schema": SCHEMA,
                "method": NAME,
                "method_version": VERSION,
                "performance": result,
            },
        )
