"""Separate task checks, reference variability and compatible systems impact."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping
from itertools import combinations
from typing import Any

from ..models import MetricSummary, RunRecord
from ..protocols import MeasurementResult
from .impact_schema import DEFINITIONS
from .impact_schema import NAME as NAME
from .impact_schema import SCHEMA as SCHEMA
from .impact_schema import VERSION as VERSION
from .performance import compare_performance, measure_invocation_performance
from .prefix_performance import compare_prefix_performance
from .prefix_workload import WORKLOAD_METHOD
from .task_checks import METHOD as CHECK_METHOD
from .task_checks import PASS_DEFINITION
from .task_checks import SCHEMA as CHECK_SCHEMA
from .task_checks import VERSION as CHECK_VERSION
from .trajectory import TokenTrajectoryProtocol, _capture_rows


def _capture(record: RunRecord) -> Mapping[str, Any]:
    measurements = record.evidence.get("measurements", {})
    capture = (
        measurements.get(TokenTrajectoryProtocol.name, {})
        if isinstance(measurements, Mapping)
        else {}
    )
    return capture if isinstance(capture, Mapping) else {}


def _quality(record: RunRecord) -> Mapping[str, Any] | None:
    quality = _capture(record).get("task_checks")
    if not isinstance(quality, Mapping) or (
        quality.get("schema"),
        quality.get("method"),
        quality.get("method_version"),
        quality.get("status"),
    ) != (CHECK_SCHEMA, CHECK_METHOD, CHECK_VERSION, "available"):
        return None
    metric = record.metrics.get("task_check_pass_rate")
    digest = quality.get("workload_sha256")
    if not _is_digest(digest):
        return None
    count, passed = quality.get("check_count"), quality.get("passed")
    checks = quality.get("checks", ())
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or count <= 0
        or isinstance(passed, bool)
        or not isinstance(passed, int)
        or not 0 <= passed <= count
    ):
        return None
    if (
        not isinstance(checks, (list, tuple))
        or len(checks) != count
        or any(
            not isinstance(row, Mapping) or not isinstance(row.get("passed"), bool)
            for row in checks
        )
    ):
        return None
    if sum(row["passed"] for row in checks) != passed:
        return None
    if not all(_valid_check_row(row) for row in checks):
        return None
    if (
        metric is None
        or metric.definition != PASS_DEFINITION
        or metric.aggregation != "mean"
        or metric.coverage != 1.0
        or isinstance(metric.coverage, bool)
        or len(metric.samples) != count
    ):
        return None
    if (
        isinstance(metric.value, bool)
        or not isinstance(metric.value, (int, float))
        or not math.isfinite(metric.value)
        or metric.value != passed / count
    ):
        return None
    return quality


def _is_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _valid_check_row(row: Mapping[str, Any]) -> bool:
    return (
        row.get("kind") in {"exact_text", "nonempty", "json_object"}
        and all(
            isinstance(row.get(key), str) and row[key]
            for key in ("prompt_id", "check_id")
        )
        and all(
            _is_digest(row.get(key))
            for key in ("definition_sha256", "output_sha256", "prompt_sha256")
        )
    )


def compare_task_checks(
    reference: RunRecord, candidate: RunRecord
) -> tuple[dict[str, Any], dict[str, float]]:
    left, right = _quality(reference), _quality(candidate)
    if left is None or right is None:
        return {
            "status": "unavailable",
            "reason": "complete declared task checks were not retained for both runs",
        }, {}
    if (
        not left.get("workload_sha256")
        or left["workload_sha256"] != right.get("workload_sha256")
        or left["check_count"] != right["check_count"]
    ):
        return {
            "status": "unavailable",
            "reason": "task-check definitions or workload identities differ",
        }, {}
    left_value, right_value = (
        left["passed"] / left["check_count"],
        right["passed"] / right["check_count"],
    )
    values = {
        "reference_task_pass_rate": left_value,
        "candidate_task_pass_rate": right_value,
        "task_pass_rate_delta": right_value - left_value,
    }
    evidence = {
        "status": "available",
        "workload_sha256": left["workload_sha256"],
        "configured_prompts": left["configured_prompts"],
        "total_prompts": left["total_prompts"],
        "check_count_per_role": left["check_count"],
        "reference_passed": left["passed"],
        "candidate_passed": right["passed"],
        "failed_candidate_checks": [
            {
                key: row[key]
                for key in (
                    "prompt_id",
                    "check_id",
                    "kind",
                    "definition_sha256",
                    "output_sha256",
                )
            }
            for row in right["checks"]
            if not row["passed"]
        ],
    }
    return evidence, values


def reference_variability(record: RunRecord) -> tuple[dict[str, Any], dict[str, float]]:
    capture = _capture(record)
    workload = capture.get("workload", {})
    if (
        not isinstance(workload, Mapping)
        or workload.get("method") != WORKLOAD_METHOD
        or workload.get("measured_trials", 0) < 2
    ):
        return {
            "status": "unavailable",
            "reason": "at least two retained reference trials per prompt are required",
        }, {}
    rows = _capture_rows(MeasurementResult(evidence=capture))
    groups: dict[str, dict[int, tuple[int, ...]]] = defaultdict(dict)
    trials = workload["measured_trials"]
    for row in rows:
        prefix, separator, prompt_id = row["id"].partition(":")
        if (
            not separator
            or not prefix.isdecimal()
            or not prompt_id
            or not 0 <= int(prefix) < trials
            or int(prefix) in groups[prompt_id]
        ):
            raise ValueError("reference trial identities are malformed or duplicated")
        groups[prompt_id][int(prefix)] = tuple(row["token_ids"])
    if any(set(group) != set(range(trials)) for group in groups.values()):
        raise ValueError("reference repeats are incomplete")
    comparisons = [
        (left == right)
        for group in groups.values()
        for left, right in combinations(group.values(), 2)
    ]
    matches = sum(comparisons)
    evidence = {
        "status": "available",
        "reference_trials_per_prompt": trials,
        "unique_prompts": len(groups),
        "compared_pairs": len(comparisons),
        "different_pairs": len(comparisons) - matches,
        "limitation": "Observed sample repeatability; not a statistical guarantee or a quality score.",
    }
    return evidence, {"reference_repeatability": matches / len(comparisons)}


class VerificationImpactAnalysis:
    name = NAME
    version = VERSION

    def analyze(self, left: RunRecord, right: RunRecord) -> MeasurementResult:
        quality, quality_metrics = compare_task_checks(left, right)
        variability, variability_metrics = reference_variability(left)
        if left.requested.runtime.get("name") == "vllm":
            performance = compare_prefix_performance(left, right, True)
        else:
            performance = compare_performance(
                measure_invocation_performance(left),
                measure_invocation_performance(right),
                comparable=True,
            )
        values = {**quality_metrics, **variability_metrics}
        if performance["available"]:
            values["candidate_latency_seconds"] = performance["candidate"]
            if performance["relative_delta"] is not None:
                values["latency_ratio"] = (
                    performance["candidate"] / performance["reference"]
                )
        metrics = {
            name: MetricSummary(
                definition=DEFINITIONS[name],
                value=value,
                aggregation="mean" if name.endswith("pass_rate") else "derived",
                coverage=1.0,
            )
            for name, value in values.items()
        }
        return MeasurementResult(
            metrics=metrics,
            evidence={
                "schema": SCHEMA,
                "method": NAME,
                "method_version": VERSION,
                "quality": quality,
                "reference_variability": variability,
                "performance": performance,
            },
        )
