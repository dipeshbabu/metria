"""Batch invariance: does each prompt's output survive concurrent load?

Production servers batch concurrent requests, and batching can change the
numerics of a forward pass. After the sequential trials, an endpoint workload
with a declared ``concurrency`` block sends every prompt ``rounds`` times with
up to ``max_in_flight`` requests outstanding and compares each greedy output
with the same prompt's first sequential output.

``batch_invariance`` is the fraction of concurrent requests whose token IDs
equal the sequential output. Client-side throughput and latency are measured on
the client clock and include the network; they are labeled as such.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from ..models import MetricDefinition, MetricDirection, MetricSummary, RunRecord
from ..protocols import InferenceRequest, MeasurementResult
from .trajectory import _first_divergence, _prompt_rows

CONCURRENCY_KEY = "concurrency"
NAME = "metria.batch_invariance"
VERSION = "1"
SCHEMA = "metria.batch_invariance.v1"
COMPARISON_SCHEMA = "metria.batch_invariance_comparison.v1"
EVIDENCE_KEY = "batch_invariance"
BOUNDARY = "client clock around each HTTP request; includes the network"

DEFINITIONS = MappingProxyType(
    {
        name: MetricDefinition(name, unit, direction, NAME, VERSION)
        for name, unit, direction in (
            (
                "reference_batch_invariance",
                "fraction",
                MetricDirection.HIGHER_IS_BETTER,
            ),
            (
                "candidate_batch_invariance",
                "fraction",
                MetricDirection.HIGHER_IS_BETTER,
            ),
            (
                "client_requests_per_second_ratio",
                "ratio",
                MetricDirection.HIGHER_IS_BETTER,
            ),
            (
                "client_request_latency_ratio",
                "ratio",
                MetricDirection.LOWER_IS_BETTER,
            ),
        )
    }
)


def concurrency_settings(config: Mapping[str, Any]) -> dict[str, int] | None:
    """Validate the optional ``concurrency`` block of an endpoint workload."""

    value = config.get(CONCURRENCY_KEY)
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) - {"max_in_flight", "rounds"}:
        raise ValueError("concurrency accepts only max_in_flight and rounds")
    result = {}
    for key, default, low, high in (
        ("max_in_flight", None, 2, 64),
        ("rounds", 2, 1, 10),
    ):
        item = value.get(key, default)
        if (
            isinstance(item, bool)
            or not isinstance(item, int)
            or not low <= item <= high
        ):
            raise ValueError(
                f"concurrency.{key} must be an integer from {low} to {high}"
            )
        result[key] = item
    return result


def without_concurrency(config: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if key != CONCURRENCY_KEY}


def _sequential_tokens(evidence: Mapping[str, Any]) -> dict[str, tuple[int, ...]]:
    """First measured sequential trial's tokens, keyed by workload prompt id."""

    result = {}
    for row in evidence["prompts"]:
        trial, _, prompt_id = str(row["id"]).partition(":")
        if trial == "0":
            result[prompt_id] = tuple(row["token_ids"])
    return result


def measure(
    session: Any,
    config: Mapping[str, Any],
    settings: Mapping[str, int],
    sequential: Mapping[str, Any],
) -> dict[str, Any]:
    """Send the declared concurrent schedule and compare with sequential output."""

    infer = getattr(session, "infer_concurrent", None)
    if infer is None:
        raise RuntimeError("this runtime session cannot send concurrent requests")
    from .prefix_workload import trajectory_config

    rows = _prompt_rows(trajectory_config(without_concurrency(config)))
    reference = _sequential_tokens(sequential)
    schedule = [row for _ in range(settings["rounds"]) for row in rows]
    tokens, timings = infer(
        [
            InferenceRequest(prompt=row["prompt"], generation=row["generation"])
            for row in schedule
        ],
        max_in_flight=settings["max_in_flight"],
    )
    per_prompt: dict[str, dict[str, Any]] = {}
    matched = 0
    for row, output in zip(schedule, tokens, strict=True):
        divergence, _ = _first_divergence(reference[row["id"]], output)
        entry = per_prompt.setdefault(
            row["id"],
            {"id": row["id"], "requests": 0, "matched": 0, "first_divergence": []},
        )
        entry["requests"] += 1
        if divergence is None:
            entry["matched"] += 1
            matched += 1
        else:
            entry["first_divergence"].append(divergence)
    wall = max((row["end_offset_seconds"] for row in timings), default=0.0)
    return {
        "schema": SCHEMA,
        "method": NAME,
        "method_version": VERSION,
        "max_in_flight": settings["max_in_flight"],
        "rounds": settings["rounds"],
        "schedule": "every prompt per round, submitted in workload order",
        "comparison": "token IDs equal the first measured sequential trial",
        "requests": len(schedule),
        "matched": matched,
        "batch_invariance": matched / len(schedule),
        "per_prompt": list(per_prompt.values()),
        "client": {
            "boundary": BOUNDARY,
            "wall_seconds": wall,
            "requests_per_second": len(schedule) / wall if wall > 0 else None,
            "mean_request_seconds": math.fsum(row["seconds"] for row in timings)
            / len(timings),
        },
    }


def _evidence(record: RunRecord) -> Mapping[str, Any]:
    measurements = record.evidence.get("measurements")
    capture = (
        measurements.get("kv_fidelity.decode_time_trajectory")
        if isinstance(measurements, Mapping)
        else None
    )
    value = capture.get(EVIDENCE_KEY) if isinstance(capture, Mapping) else None
    if not isinstance(value, Mapping) or value.get("schema") != SCHEMA:
        raise ValueError(f"run {record.run_id!r} has no batch-invariance evidence")
    return value


def _ratio(numerator: Any, denominator: Any) -> float | None:
    if not isinstance(numerator, (int, float)) or not isinstance(
        denominator, (int, float)
    ):
        return None
    if (
        denominator <= 0
        or not math.isfinite(numerator)
        or not math.isfinite(denominator)
    ):
        return None
    return float(numerator) / float(denominator)


class BatchInvarianceAnalysis:
    """Pair each run's batch invariance and client-side load behavior."""

    name = NAME
    version = VERSION

    def analyze(self, left: RunRecord, right: RunRecord) -> MeasurementResult:
        reference, candidate = _evidence(left), _evidence(right)
        if (reference["max_in_flight"], reference["rounds"]) != (
            candidate["max_in_flight"],
            candidate["rounds"],
        ):
            raise ValueError("reference and candidate used different schedules")
        values: dict[str, float | None] = {
            "reference_batch_invariance": reference["batch_invariance"],
            "candidate_batch_invariance": candidate["batch_invariance"],
            "client_requests_per_second_ratio": _ratio(
                candidate["client"]["requests_per_second"],
                reference["client"]["requests_per_second"],
            ),
            "client_request_latency_ratio": _ratio(
                candidate["client"]["mean_request_seconds"],
                reference["client"]["mean_request_seconds"],
            ),
        }
        metrics = {
            name: MetricSummary(
                definition=DEFINITIONS[name],
                value=value,
                aggregation="derived",
                coverage=1.0,
            )
            for name, value in values.items()
            if value is not None
        }
        return MeasurementResult(
            metrics=metrics,
            evidence={
                "schema": COMPARISON_SCHEMA,
                "method": NAME,
                "method_version": VERSION,
                "max_in_flight": reference["max_in_flight"],
                "rounds": reference["rounds"],
                "client_boundary": BOUNDARY,
                "reference": {k: reference[k] for k in ("requests", "matched")},
                "candidate": {k: candidate[k] for k in ("requests", "matched")},
            },
        )
