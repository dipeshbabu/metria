"""Task checks alongside native trajectories, without changing latency methods."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..protocols import CaptureRequest, MeasurementResult, RuntimeSession
from .prefix_workload import _SequentialSession
from .task_checks import prompt_checks, summarize_checks
from .trajectory import TokenTrajectoryProtocol


def _trajectory_config(config: Mapping[str, Any]) -> dict[str, Any]:
    prompt_checks(config.get("prompts"))
    return {
        **config,
        "prompts": [
            {key: value for key, value in row.items() if key != "checks"}
            for row in config["prompts"]
        ],
    }


class CheckedTrajectoryProtocol(TokenTrajectoryProtocol):
    """Use the shared task-check boundary for one sequential native workload."""

    def requirements(self, config: Mapping[str, Any]) -> tuple[CaptureRequest, ...]:
        return super().requirements(_trajectory_config(config))

    def execute(
        self,
        session: RuntimeSession,
        scenario: Mapping[str, Any],
        config: Mapping[str, Any],
    ) -> MeasurementResult:
        base = _trajectory_config(config)
        specs = prompt_checks(config["prompts"])
        checking = _SequentialSession(
            session,
            0,
            True,
            checks=specs,
            prompt_ids=tuple(row["id"] for row in base["prompts"]),
        )
        captured = super().execute(checking, scenario, base)
        quality, metric = summarize_checks(
            checking.check_results, specs, config["prompts"], 1
        )
        metrics = dict(captured.metrics)
        if metric is not None:
            metrics[metric.definition.name] = metric
        # The native llama.cpp invocation clock owns cold-process latency. The
        # sequential wrapper's incidental call timings are deliberately omitted.
        return MeasurementResult(
            metrics=metrics, evidence={**captured.evidence, "task_checks": quality}
        )
