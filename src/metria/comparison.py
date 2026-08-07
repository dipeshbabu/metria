"""Comparison semantics for Metria studies."""

from __future__ import annotations

from typing import Any

from .models import (
    ComparisonPlan,
    CompatibilityIssue,
    CompatibilityReport,
    RunRecord,
    RunStatus,
)

_MISSING = object()
_MISSING_LABEL = "<missing>"


def _requested_dimension(record: RunRecord, dimension: str) -> Any:
    """Resolve a named comparison dimension without conflating missing and None."""

    spec = record.requested
    values: dict[str, Any] = {
        "model": spec.model,
        "runtime": spec.runtime,
        "scenario": spec.scenario,
        "measurements": spec.measurements,
        "treatments": spec.treatments,
        "trial_policy": spec.trial_policy,
        "environment_selector": spec.environment_selector,
    }
    if dimension in values:
        return values[dimension]
    if dimension.startswith("observed."):
        key = dimension.removeprefix("observed.")
        return record.observed[key] if key in record.observed else _MISSING
    if dimension.startswith("resolved."):
        key = dimension.removeprefix("resolved.")
        return record.resolved[key] if key in record.resolved else _MISSING
    raise KeyError(dimension)


def _display_value(value: Any) -> Any:
    """Convert the private missing sentinel into a readable issue value."""

    return _MISSING_LABEL if value is _MISSING else value


def _check_dimension(
    left: RunRecord,
    right: RunRecord,
    dimension: str,
    role: str,
) -> CompatibilityIssue | None:
    """Check one controlled or blocking dimension for pairwise comparability."""

    try:
        left_value = _requested_dimension(left, dimension)
        right_value = _requested_dimension(right, dimension)
    except KeyError:
        return CompatibilityIssue(
            dimension=dimension,
            left=None,
            right=None,
            reason=f"unknown {role} dimension",
        )

    if left_value is _MISSING or right_value is _MISSING:
        return CompatibilityIssue(
            dimension=dimension,
            left=_display_value(left_value),
            right=_display_value(right_value),
            reason=f"required {role} dimension is missing",
        )
    if left_value == right_value:
        return None
    return CompatibilityIssue(
        dimension=dimension,
        left=left_value,
        right=right_value,
        reason=(
            "controlled dimension differs"
            if role == "control"
            else "runs belong to different comparison blocks"
        ),
    )


def compare_runs(
    left: RunRecord,
    right: RunRecord,
    plan: ComparisonPlan,
) -> CompatibilityReport:
    """Determine whether two completed runs support the comparison in ``plan``.

    Dimensions listed in ``vary`` are intentionally allowed to differ.
    Controlled dimensions must match. Blocking dimensions must also match for
    a direct pairwise comparison; callers can group records into blocks before
    invoking this function.

    Direct comparison is deliberately strict about lifecycle state. A failed,
    timed-out, or partial record remains useful experiment evidence, but is not
    analysis-ready by default. Both records must therefore be ``COMPLETED``.

    Metric compatibility is stricter than study compatibility: two metrics are
    directly comparable only when their complete metric identities match.
    Methodologically different measurements remain visible but are reported as
    incompatible rather than being silently combined.
    """

    issues: list[CompatibilityIssue] = []
    if (
        left.status is not RunStatus.COMPLETED
        or right.status is not RunStatus.COMPLETED
    ):
        issues.append(
            CompatibilityIssue(
                dimension="status",
                left=left.status.value,
                right=right.status.value,
                reason="both runs must be completed for direct comparison",
            )
        )

    for dimension in sorted(plan.control):
        issue = _check_dimension(left, right, dimension, "control")
        if issue is not None:
            issues.append(issue)
    for dimension in sorted(plan.block_by):
        issue = _check_dimension(left, right, dimension, "block")
        if issue is not None:
            issues.append(issue)

    comparable_metrics: list[str] = []
    incompatible_metrics: dict[str, str] = {}
    shared_metrics = sorted(set(left.metrics) & set(right.metrics))
    for name in shared_metrics:
        left_metric = left.metrics[name]
        right_metric = right.metrics[name]
        if left_metric.definition.identity == right_metric.definition.identity:
            comparable_metrics.append(name)
        else:
            incompatible_metrics[name] = (
                "metric identities differ; compare effect sizes against a "
                "method-matched baseline instead of combining raw values"
            )

    return CompatibilityReport(
        compatible=not issues and not incompatible_metrics,
        issues=tuple(issues),
        comparable_metrics=tuple(comparable_metrics),
        incompatible_metrics=incompatible_metrics,
    )
