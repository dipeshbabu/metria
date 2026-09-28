"""Stable identity of verification decision measurements."""

from ..models import MetricDefinition, MetricDirection

NAME = "metria.verification_impact"
VERSION = "1"
SCHEMA = "metria.verification_impact.v1"
DEFINITIONS = {
    "candidate_task_pass_rate": MetricDefinition(
        "candidate_task_pass_rate",
        "fraction",
        MetricDirection.HIGHER_IS_BETTER,
        "metria.task_check_comparison",
        "1",
    ),
    "reference_task_pass_rate": MetricDefinition(
        "reference_task_pass_rate",
        "fraction",
        MetricDirection.HIGHER_IS_BETTER,
        "metria.task_check_comparison",
        "1",
    ),
    "task_pass_rate_delta": MetricDefinition(
        "task_pass_rate_delta",
        "fraction_delta",
        MetricDirection.HIGHER_IS_BETTER,
        "metria.task_check_comparison",
        "1",
    ),
    "reference_repeatability": MetricDefinition(
        "reference_repeatability",
        "fraction",
        MetricDirection.HIGHER_IS_BETTER,
        "metria.observed_reference_repeatability",
        "1",
    ),
    "candidate_latency_seconds": MetricDefinition(
        "candidate_latency_seconds",
        "seconds",
        MetricDirection.LOWER_IS_BETTER,
        "metria.compatible_latency_impact",
        "1",
    ),
    "latency_ratio": MetricDefinition(
        "latency_ratio",
        "ratio",
        MetricDirection.LOWER_IS_BETTER,
        "metria.compatible_latency_impact",
        "1",
    ),
}
