"""Durable recipe execution for the first Metria run CLI."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .hardware import capture_hardware_fingerprint
from .identity import HardwareFingerprint
from .models import CompatibilityReport, RunStatus
from .protocols import MeasurementResult
from .recipes import STUDY_RECIPE_SCHEMA, StudyRecipe, _json_value, study_recipe_digest
from .records import (
    _metric_summary_to_data,
    dump_run_record,
    run_evidence_digest,
    run_record_digest,
)
from .registry import RegistryBundle, validate_study_availability
from .study_execution import (
    PairwiseAnalysisStatus,
    StudyExecutionResult,
    StudyPairAnalysis,
    execute_study,
)

STUDY_RESULT_SCHEMA = "metria.study_result.v1"


@dataclass(frozen=True)
class PersistedStudyResult:
    """One executed study together with its durable output paths."""

    result: StudyExecutionResult
    output_dir: Path
    manifest_path: Path
    run_paths: tuple[Path, ...]
    successful: bool


def _prepare_output_dir(path: Path) -> None:
    """Require an empty/nonexistent output directory before experiment work."""

    if path.exists():
        if not path.is_dir():
            raise ValueError(f"output path exists and is not a directory: {path}")
        if any(path.iterdir()):
            raise ValueError(f"output directory must be empty: {path}")
    else:
        path.mkdir(parents=True)


def _compatibility_to_data(report: CompatibilityReport) -> dict[str, Any]:
    return {
        "compatible": report.compatible,
        "issues": [
            {
                "dimension": issue.dimension,
                "left": _json_value(issue.left, path="comparison.issue.left"),
                "right": _json_value(issue.right, path="comparison.issue.right"),
                "reason": issue.reason,
            }
            for issue in report.issues
        ],
        "comparable_metrics": list(report.comparable_metrics),
        "incompatible_metrics": dict(report.incompatible_metrics),
    }


def _measurement_result_to_data(result: MeasurementResult) -> dict[str, Any]:
    return {
        "metrics": {
            key: _metric_summary_to_data(summary, key=key)
            for key, summary in result.metrics.items()
        },
        "evidence": _json_value(result.evidence, path="analysis.result.evidence"),
        "artifacts": _json_value(result.artifacts, path="analysis.result.artifacts"),
    }


def _analysis_to_data(analysis: StudyPairAnalysis) -> dict[str, Any]:
    return {
        "name": analysis.name,
        "version": analysis.version,
        "status": analysis.status.value,
        "reason": analysis.reason,
        "error_type": analysis.error_type,
        "message_sha256": analysis.message_sha256,
        "result": (
            _measurement_result_to_data(analysis.result)
            if analysis.result is not None
            else None
        ),
    }


def _study_success(result: StudyExecutionResult) -> bool:
    records_ok = all(record.status is RunStatus.COMPLETED for record in result.records)
    comparisons_ok = all(
        comparison.report.compatible for comparison in result.comparisons
    )
    analyses_ok = all(
        analysis.status is PairwiseAnalysisStatus.COMPLETED
        for comparison in result.comparisons
        for analysis in comparison.analyses
    )
    return records_ok and comparisons_ok and analyses_ok


def _manifest_data(
    recipe: StudyRecipe,
    result: StudyExecutionResult,
    hardware: HardwareFingerprint,
    run_paths: tuple[Path, ...],
) -> dict[str, Any]:
    recipe_digest = study_recipe_digest(recipe)
    records = [
        {
            "run_id": record.run_id,
            "path": path.name,
            "status": record.status.value,
            "record_digest": run_record_digest(record),
            "evidence_digest": run_evidence_digest(record),
        }
        for record, path in zip(result.records, run_paths, strict=True)
    ]
    comparisons = [
        {
            "left_run_id": comparison.left_run_id,
            "right_run_id": comparison.right_run_id,
            "report": _compatibility_to_data(comparison.report),
            "analyses": [
                _analysis_to_data(analysis) for analysis in comparison.analyses
            ],
        }
        for comparison in result.comparisons
    ]
    return {
        "schema": STUDY_RESULT_SCHEMA,
        "study": recipe.study.name,
        "recipe": {
            "schema": STUDY_RECIPE_SCHEMA,
            "digest": recipe_digest,
        },
        "hardware": _json_value(hardware.to_mapping(), path="manifest.hardware"),
        "successful": _study_success(result),
        "records": records,
        "comparisons": comparisons,
    }


def execute_recipe_to_directory(
    recipe: StudyRecipe,
    *,
    output_dir: str | Path,
    registry: RegistryBundle,
    hardware: HardwareFingerprint | None = None,
) -> PersistedStudyResult:
    """Execute one recipe through explicit registries and persist all run evidence."""

    validate_study_availability(recipe.study, registry)
    destination = Path(output_dir)
    _prepare_output_dir(destination)
    observed_hardware = hardware or capture_hardware_fingerprint()
    recipe_digest = study_recipe_digest(recipe)
    invocation_provenance: Mapping[str, Any] = {
        "recipe": {
            "schema": STUDY_RECIPE_SCHEMA,
            "digest": recipe_digest,
        },
        "hardware": observed_hardware.to_mapping(),
        "orchestrator": {
            "name": "metria.execute_recipe_to_directory",
            "version": "1",
        },
    }
    result = execute_study(
        recipe.study,
        adapters=registry.runtimes,
        measurements=registry.measurements,
        measurement_configs=recipe.measurement_configs,
        environment=recipe.environment,
        analyses=registry.analyses,
        invocation_provenance=invocation_provenance,
    )

    run_paths: list[Path] = []
    for record in result.records:
        path = destination / f"{record.run_id}.json"
        dump_run_record(path, record)
        run_paths.append(path)

    paths = tuple(run_paths)
    manifest = _manifest_data(recipe, result, observed_hardware, paths)
    manifest_path = destination / "study-result.json"
    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return PersistedStudyResult(
        result=result,
        output_dir=destination,
        manifest_path=manifest_path,
        run_paths=paths,
        successful=bool(manifest["successful"]),
    )
