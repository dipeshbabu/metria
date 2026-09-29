"""Shared binding between declared task checks and retained verification evidence."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .measurements.task_checks import prompt_checks, task_workload_identity
from .measurements.verification_impact import _quality
from .models import RunRecord


def task_check_identity(
    config: Mapping[str, Any], measured_trials: int = 1
) -> dict[str, Any] | None:
    checks = prompt_checks(config["prompts"])
    count = sum(len(row) for row in checks) * measured_trials
    if not count:
        return None
    return {
        "sha256": task_workload_identity(checks, config["prompts"], measured_trials),
        "check_count": count,
    }


def task_check_gaps(
    record: RunRecord, expected: Mapping[str, Any] | None
) -> tuple[str, ...]:
    if expected is None:
        return ()
    quality = _quality(record)
    if (
        quality is None
        or quality.get("workload_sha256") != expected["sha256"]
        or quality.get("check_count") != expected["check_count"]
    ):
        return ("declared task-check evidence is missing or inconsistent",)
    return ()
