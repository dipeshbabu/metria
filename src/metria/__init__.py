"""Metria core models for reproducible inference-systems studies."""

from .comparison import compare_runs
from .execution import execute_run
from .identity import (
    ArtifactManifest,
    Capability,
    CapabilitySet,
    HardwareFingerprint,
    ModelRef,
    RuntimeConfig,
    SupportLevel,
    WorkloadSpec,
)
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
from .recipes import (
    StudyRecipe,
    dump_study_recipe,
    load_study_recipe,
    study_recipe_digest,
    study_recipe_from_data,
    study_recipe_to_data,
    study_recipe_to_json,
)
from .study_execution import (
    PairwiseAnalysisStatus,
    StudyExecutionResult,
    StudyPairAnalysis,
    StudyPairComparison,
    execute_study,
)

__version__ = "0.1.0.dev0"

__all__ = [
    "ArtifactManifest",
    "Capability",
    "CapabilitySet",
    "ComparisonPlan",
    "CompatibilityIssue",
    "CompatibilityReport",
    "HardwareFingerprint",
    "MeasurementResult",
    "MetricDefinition",
    "MetricDirection",
    "MetricSample",
    "MetricSummary",
    "ModelRef",
    "PairwiseAnalysisStatus",
    "RunRecord",
    "RunSpec",
    "RunStatus",
    "RuntimeConfig",
    "StudyExecutionResult",
    "StudyPairAnalysis",
    "StudyPairComparison",
    "StudyRecipe",
    "StudySpec",
    "SupportLevel",
    "TreatmentSpec",
    "TreatmentType",
    "WorkloadSpec",
    "compare_runs",
    "dump_study_recipe",
    "execute_run",
    "execute_study",
    "load_study_recipe",
    "study_recipe_digest",
    "study_recipe_from_data",
    "study_recipe_to_data",
    "study_recipe_to_json",
]
