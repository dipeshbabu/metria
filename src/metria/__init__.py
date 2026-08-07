"""Metria core models for reproducible inference-systems studies."""

from .comparison import compare_runs
from .models import (
    ComparisonPlan,
    CompatibilityIssue,
    CompatibilityReport,
    MetricDefinition,
    MetricDirection,
    MetricSample,
    MetricSummary,
    RunRecord,
    RunSpec,
    RunStatus,
    StudySpec,
    TreatmentSpec,
    TreatmentType,
)
from .protocols import MeasurementResult

__version__ = "0.1.0.dev0"

__all__ = [
    "ComparisonPlan",
    "CompatibilityIssue",
    "CompatibilityReport",
    "MeasurementResult",
    "MetricDefinition",
    "MetricDirection",
    "MetricSample",
    "MetricSummary",
    "RunRecord",
    "RunSpec",
    "RunStatus",
    "StudySpec",
    "TreatmentSpec",
    "TreatmentType",
    "compare_runs",
]
