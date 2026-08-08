"""Metria core models for reproducible inference-systems studies."""

from .capabilities import (
    GeometryInspection,
    ModelGeometry,
    evaluate_turboquant_kv_capability,
    inspect_model_geometry,
)
from .comparison import compare_runs
from .execution import execute_run
from .hardware import capture_hardware_fingerprint
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
from .inspection import (
    PreflightCapabilityResult,
    capability_inspection_to_mapping,
    inspect_run_capabilities,
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
    "GeometryInspection",
    "HardwareFingerprint",
    "MeasurementResult",
    "MetricDefinition",
    "MetricDirection",
    "MetricSample",
    "MetricSummary",
    "ModelGeometry",
    "ModelRef",
    "PairwiseAnalysisStatus",
    "PreflightCapabilityResult",
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
    "capability_inspection_to_mapping",
    "capture_hardware_fingerprint",
    "compare_runs",
    "dump_study_recipe",
    "evaluate_turboquant_kv_capability",
    "execute_run",
    "execute_study",
    "inspect_model_geometry",
    "inspect_run_capabilities",
    "load_study_recipe",
    "study_recipe_digest",
    "study_recipe_from_data",
    "study_recipe_to_data",
    "study_recipe_to_json",
]
