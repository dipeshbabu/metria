"""Repeated concurrent serving workloads with checked native timing evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..protocols import (
    CaptureRequest,
    InferenceBatch,
    InferenceRequest,
    MeasurementResult,
    RuntimeSession,
)
from .prefix_workload import _summary, trajectory_config, trial_settings
from .serving_metrics import measure_serving
from .task_checks import evaluate_checks, prompt_checks, summarize_checks
from .trajectory import TokenTrajectoryProtocol, _sha256_text

WORKLOAD_METHOD = "metria.concurrent_serving_trials.v1"


@dataclass
class _CheckedBatch:
    session: RuntimeSession
    prompts: Sequence[Mapping[str, Any]]
    checks: Sequence[Sequence[Mapping[str, Any]]]
    rows: list[dict[str, Any]] = field(default_factory=list)
    batch: Mapping[str, Any] = field(default_factory=dict)

    def infer(
        self,
        requests: Sequence[InferenceRequest],
        capture: Sequence[CaptureRequest] = (),
    ) -> InferenceBatch:
        result = self.session.infer(requests, capture)
        if len(result.outputs) != len(requests):
            raise RuntimeError("serving output count differs from the workload")
        for request, output, prompt, checks in zip(
            requests, result.outputs, self.prompts, self.checks, strict=True
        ):
            self.rows.extend(
                evaluate_checks(
                    output,
                    checks,
                    prompt_id=prompt["id"],
                    prompt_sha256=_sha256_text(request.prompt),
                )
            )
        self.batch = result.metadata.get("serving", {})
        return result

    def reset(self, scope: str = "measurement") -> None:
        self.session.reset(scope)

    def close(self) -> None:
        pass


class ServingWorkloadProtocol(TokenTrajectoryProtocol):
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
        checks = prompt_checks(config["prompts"])
        results = []
        batches = []
        memories = []
        check_rows = []
        snapshot = getattr(session, "memory_snapshot", None)
        if not callable(snapshot):
            raise ValueError("serving workload requires its native measurement session")
        for index in range(warmups + trials):
            measured = index >= warmups
            rows = [
                {**row, "id": f"{index - warmups}:{row['id']}"}
                for row in base["prompts"]
            ]
            session.reset("prefix-cache")
            if measured:
                session.reset("serving-memory")
            checked = _CheckedBatch(session, rows, checks)
            result = TokenTrajectoryProtocol().execute(
                checked, scenario, {**base, "prompts": rows}
            )
            if measured:
                results.append(result)
                batches.append(checked.batch)
                memories.append(snapshot())
                check_rows.extend(checked.rows)
        metrics = {
            name: _summary(
                original.definition,
                tuple(
                    sample
                    for result in results
                    for sample in result.metrics[name].samples
                ),
            )
            for name, original in results[0].metrics.items()
        }
        performance = measure_serving(
            batches, memories, prompts_per_trial=len(base["prompts"])
        )
        metrics.update(performance.metrics)
        quality, metric = summarize_checks(
            check_rows, checks, config["prompts"], trials
        )
        if metric is not None:
            metrics[metric.definition.name] = metric
        prompts = tuple(row for result in results for row in result.evidence["prompts"])
        return MeasurementResult(
            metrics=metrics,
            evidence={
                **results[0].evidence,
                "prompts": prompts,
                "n_prompts": len(prompts),
                "task_checks": quality,
                "workload": {
                    "method": WORKLOAD_METHOD,
                    "warmup_trials": warmups,
                    "measured_trials": trials,
                    "schedule": "bounded concurrent requests in input order",
                    "isolation": "public prefix-cache reset before every trial; allocator peaks reset after warmup",
                    "batches": batches,
                    "memory": memories,
                },
                "serving_measurements": performance.evidence,
            },
        )
