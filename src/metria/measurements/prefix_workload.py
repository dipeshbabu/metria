"""Sequential cache-isolated trials using the shared trajectory capture method."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..models import MetricDefinition, MetricDirection, MetricSample, MetricSummary
from ..protocols import (
    CaptureRequest,
    InferenceBatch,
    InferenceRequest,
    MeasurementResult,
    RuntimeSession,
)
from .task_checks import evaluate_checks, prompt_checks, summarize_checks
from .trajectory import TokenTrajectoryProtocol, _prompt_rows

WORKLOAD_METHOD = "metria.prefix_cache_trials.v1"
LATENCY_METHOD = "metria.runtime_call_latency"


def trial_settings(config: Mapping[str, Any]) -> tuple[int, int]:
    """Validate bounded trials before an engine is launched."""
    values = []
    for name, default, minimum, maximum in (
        ("warmup_trials", 1, 0, 3),
        ("measured_trials", 3, 1, 10),
    ):
        value = config.get(name, default)
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not minimum <= value <= maximum
        ):
            raise ValueError(f"{name} must be an integer from {minimum} to {maximum}")
        values.append(value)
    return values[0], values[1]


def trajectory_config(config: Mapping[str, Any]) -> dict[str, Any]:
    unknown = set(config) - {
        "prompts",
        "generation",
        "warmup_trials",
        "measured_trials",
    }
    if unknown:
        raise ValueError("unsupported prefix-workload configuration fields")
    result = {
        name: value
        for name, value in config.items()
        if name in {"prompts", "generation"}
    }
    prompt_checks(result.get("prompts"))
    result["prompts"] = [
        {key: value for key, value in row.items() if key != "checks"}
        for row in result["prompts"]
    ]
    if len(_prompt_rows(result)) > 100:
        raise ValueError("prefix verification accepts at most 100 prompts")
    trial_settings(config)
    return result


@dataclass
class _SequentialSession:
    """Apply a declared sequential schedule without changing runtime mechanics."""

    session: RuntimeSession
    trial: int
    measured: bool
    timings: list[dict[str, Any]] = field(default_factory=list)
    checks: Sequence[Sequence[Mapping[str, Any]]] = ()
    prompt_ids: Sequence[str] = ()
    check_results: list[dict[str, Any]] = field(default_factory=list)

    def infer(
        self,
        requests: Sequence[InferenceRequest],
        capture: Sequence[CaptureRequest] = (),
    ) -> InferenceBatch:
        outputs: list[str] = []
        captures: dict[str, list[Any]] = {item.kind: [] for item in capture}
        for index, request in enumerate(requests):
            started = time.perf_counter_ns()
            batch = self.session.infer((request,), capture=capture)
            elapsed = (time.perf_counter_ns() - started) / 1_000_000_000
            if len(batch.outputs) != 1 or not math.isfinite(elapsed) or elapsed < 0:
                raise RuntimeError(
                    "invalid sequential inference result or clock sample"
                )
            outputs.extend(batch.outputs)
            if self.measured and self.checks:
                import hashlib

                self.check_results.extend(
                    evaluate_checks(
                        batch.outputs[0],
                        self.checks[index],
                        prompt_id=self.prompt_ids[index],
                        prompt_sha256=hashlib.sha256(
                            request.prompt.encode()
                        ).hexdigest(),
                    )
                )
            for item in capture:
                value = batch.captures.get(item.kind)
                if (
                    not isinstance(value, Sequence)
                    or isinstance(value, (str, bytes))
                    or len(value) != 1
                ):
                    raise RuntimeError(
                        "required sequential capture is missing or malformed"
                    )
                captures[item.kind].extend(value)
            invocations = batch.metadata.get("invocations", ())
            native = (
                invocations[0]
                if isinstance(invocations, Sequence)
                and len(invocations) == 1
                and isinstance(invocations[0], Mapping)
                else {}
            )
            self.timings.append(
                {
                    "trial": self.trial,
                    "prompt_index": index,
                    "measured": self.measured,
                    "seconds": elapsed,
                    "prompt_tokens": native.get("prompt_tokens"),
                    "cached_tokens": native.get("cached_tokens"),
                }
            )
        return InferenceBatch(outputs=tuple(outputs), captures=captures)

    def reset(self, scope: str = "measurement") -> None:
        self.session.reset(scope)

    def close(self) -> None:
        # The shared executor owns the underlying session's lifecycle.
        pass


def _summary(
    definition: MetricDefinition, samples: tuple[MetricSample, ...]
) -> MetricSummary:
    return MetricSummary(
        definition=definition,
        value=math.fsum(sample.value / len(samples) for sample in samples),
        samples=samples,
        aggregation="mean",
        coverage=1.0,
    )


def _merge_results(
    results: Sequence[MeasurementResult],
    timings: list[dict[str, Any]],
    warmups: int,
    trials: int,
) -> MeasurementResult:
    prompts = tuple(row for result in results for row in result.evidence["prompts"])
    metrics = {}
    for name, original in results[0].metrics.items():
        samples = tuple(
            sample for result in results for sample in result.metrics[name].samples
        )
        metrics[name] = _summary(original.definition, samples)
    latency_samples = tuple(
        MetricSample(
            row["seconds"],
            metadata={"trial": row["trial"], "prompt_index": row["prompt_index"]},
        )
        for row in timings
        if row["measured"]
    )
    metrics["request_latency_seconds"] = _summary(
        MetricDefinition(
            name="request_latency_seconds",
            unit="seconds",
            direction=MetricDirection.LOWER_IS_BETTER,
            method=LATENCY_METHOD,
            version="1",
        ),
        latency_samples,
    )
    return MeasurementResult(
        metrics=metrics,
        evidence={
            **results[0].evidence,
            "prompts": prompts,
            "n_prompts": len(prompts),
            "workload": {
                "method": WORKLOAD_METHOD,
                "warmup_trials": warmups,
                "measured_trials": trials,
                "schedule": "sequential prompts in declared order",
                "isolation": "public prefix-cache reset before every warmup/measured trial",
                "clock": "perf_counter_ns",
                "timing_boundary": "session.infer call after engine launch",
                "timing_excludes": ["engine startup", "warmup", "cache reset"],
                "samples": timings,
            },
        },
    )


class PrefixWorkloadProtocol(TokenTrajectoryProtocol):
    """Retain trajectory evidence across bounded warmup and measured trials."""

    def requirements(self, config: Mapping[str, Any]) -> tuple[CaptureRequest, ...]:
        return super().requirements(trajectory_config(config))

    def execute(
        self,
        session: RuntimeSession,
        scenario: Mapping[str, Any],
        config: Mapping[str, Any],
    ) -> MeasurementResult:
        base = trajectory_config(config)
        warmups, trials = trial_settings(config)
        results = []
        timings = []
        check_specs = prompt_checks(config["prompts"])
        check_rows: list[dict[str, Any]] = []
        protocol = TokenTrajectoryProtocol()
        for index in range(warmups + trials):
            measured = index >= warmups
            trial = index - warmups
            session.reset("prefix-cache")
            rows = [{**row, "id": f"{trial}:{row['id']}"} for row in base["prompts"]]
            sequential = _SequentialSession(
                session,
                trial,
                measured,
                checks=check_specs,
                prompt_ids=tuple(row["id"] for row in rows),
            )
            result = protocol.execute(sequential, scenario, {**base, "prompts": rows})
            check_rows.extend(sequential.check_results)
            if measured:
                results.append(result)
            timings.extend(sequential.timings)
        captured = _merge_results(results, timings, warmups, trials)
        quality, metric = summarize_checks(
            check_rows, check_specs, config["prompts"], trials
        )
        metrics = dict(captured.metrics)
        if metric is not None:
            metrics[metric.definition.name] = metric
        return MeasurementResult(
            metrics=metrics, evidence={**captured.evidence, "task_checks": quality}
        )
