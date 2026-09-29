"""One local reference/candidate verification with durable evidence."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any

from ._freeze import freeze_mapping
from .capability_checks import CapabilityCheckRegistry
from .comparison import _declared_dimension_issue, compare_runs
from .hardware import capture_hardware_fingerprint
from .inspection import resolve_capability_checks
from .measurements import TokenTrajectoryProtocol, TrajectoryAgreementAnalysis
from .measurements.performance import (
    compare_performance,
    measure_invocation_performance,
)
from .measurements.verification_impact import VerificationImpactAnalysis
from .models import CompatibilityReport, RunRecord, RunStatus
from .policies import PolicyDecision, evaluate_policy
from .protocols import (
    MeasurementProtocol,
    MeasurementResult,
    PairwiseAnalysis,
    RuntimeAdapter,
)
from .recipes import StudyRecipe, _json_value, study_recipe_digest
from .records import (
    _metric_summary_to_data,
    run_evidence_digest,
    run_record_digest,
    run_record_to_json,
)
from .reporting import render_verification as render_verification
from .runtimes.llamacpp import (
    LlamaCppAdapter,
)
from .study_execution import PairwiseAnalysisStatus, StudyPairAnalysis, execute_study
from .verification_cpu import validate_cpu_run
from .verification_route import VerificationRoute
from .verification_schema import VERIFICATION_ROLES as _ROLES
from .verification_schema import VERIFICATION_SCHEMA, VERIFICATION_SCOPE

LOCAL_CPU_THREAD_VARIATIONS = frozenset(
    {
        "runtime.threads",
        "resolved.runtime.threads",
        "observed.runtime.threads",
        "observed.identity.applied.fields.threads",
    }
)

_CAPTURE_KEY = "llama_cpp_token_ids_capture_sha256"
VERIFICATION_EXIT_CODES = {
    "VERIFIED": 0,
    "PASS": 0,
    "FAIL": 1,
    "INVALID_CONFIGURATION": 2,
    "NOT_COMPARABLE": 3,
    "INSUFFICIENT_EVIDENCE": 4,
    "EXECUTION_FAILED": 5,
}


class VerificationVerdict(str, Enum):
    """Distinct outcomes; VERIFIED denotes completed comparison, not task quality."""

    VERIFIED = "VERIFIED"
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_COMPARABLE = "NOT_COMPARABLE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    EXECUTION_FAILED = "EXECUTION_FAILED"


@dataclass(frozen=True)
class VerificationResult:
    """The saved verification result and its process exit status."""

    output_dir: Path
    manifest: Mapping[str, Any]
    exit_code: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "manifest", freeze_mapping(self.manifest))

    def to_data(self) -> dict[str, Any]:
        return dict(_json_value(self.manifest, path="verification"))


@dataclass(frozen=True)
class _Registries:
    adapters: Mapping[str, RuntimeAdapter]
    measurements: Mapping[str, MeasurementProtocol]
    analyses: Mapping[str, PairwiseAnalysis]


def _builtin_registries() -> _Registries:
    measurement = TokenTrajectoryProtocol()
    analysis = TrajectoryAgreementAnalysis()
    return _Registries(
        {"llamacpp": LlamaCppAdapter()},
        {measurement.name: measurement},
        {
            analysis.name: analysis,
            VerificationImpactAnalysis.name: VerificationImpactAnalysis(),
        },
    )


def _validate_comparison(recipe: StudyRecipe) -> None:
    if len(recipe.study.runs) != 2:
        raise ValueError("verify requires exactly two runs: reference, then candidate")
    if not recipe.study.comparison.vary:
        raise ValueError(
            "verify requires an explicit intended change in comparison.vary"
        )
    if recipe.study.comparison.vary != LOCAL_CPU_THREAD_VARIATIONS:
        raise ValueError(
            "local CPU verification requires exactly the four qualified thread-change paths; use the preparation tool"
        )
    if recipe.study.comparison.waivers:
        raise ValueError(
            "local CPU verification does not permit comparison waivers outside its qualified thread-change scope"
        )


def _validate(recipe: StudyRecipe, registries: _Registries) -> None:
    _validate_comparison(recipe)
    if recipe.study.comparison.analyses not in {
        (TrajectoryAgreementAnalysis.name,),
        (TrajectoryAgreementAnalysis.name, VerificationImpactAnalysis.name),
    }:
        raise ValueError(
            "local verify requires the kv_fidelity.trajectory_match analysis"
        )
    if any(
        name not in registries.analyses for name in recipe.study.comparison.analyses
    ):
        raise ValueError("verification analysis is not registered")
    if set(recipe.environment) - {_CAPTURE_KEY, "llama_cpp_bin_dir"}:
        raise ValueError(
            "local verify environment accepts only the binary directory and qualified capture digest"
        )
    digest = recipe.environment.get(_CAPTURE_KEY)
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(c not in "0123456789abcdefABCDEF" for c in digest)
    ):
        raise ValueError(f"local verify requires environment.{_CAPTURE_KEY}")
    if set(recipe.measurement_configs) != {TokenTrajectoryProtocol.name}:
        raise ValueError(
            "local verify requires exactly one trajectory measurement configuration"
        )
    config = recipe.measurement_configs[TokenTrajectoryProtocol.name]
    pins = []
    for index, run in enumerate(recipe.study.runs):
        pins.append(validate_cpu_run(run, index, config, registries))
    if pins[0] != pins[1]:
        raise ValueError(
            "the first local verifier requires the same pinned model for both runs"
        )
    left, right = recipe.study.runs
    if left.model != right.model or left.scenario != right.scenario:
        raise ValueError(
            "local CPU thread verification requires identical model and scenario inputs"
        )
    left_runtime = {
        key: value for key, value in left.runtime.items() if key != "threads"
    }
    right_runtime = {
        key: value for key, value in right.runtime.items() if key != "threads"
    }
    if left_runtime != right_runtime:
        raise ValueError(
            "the first local verifier permits only runtime.threads to change"
        )


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _invocations(record: RunRecord) -> Sequence[Any]:
    value = record.observed.get("invocations", ())
    return (
        value
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes))
        else ()
    )


def _observed_facts(record: RunRecord) -> dict[str, Any]:
    identity = _mapping(record.observed.get("identity"))
    fields = _mapping(_mapping(identity.get("applied")).get("fields"))
    return {
        "model_sha256": _mapping(identity.get("model")).get("sha256"),
        "provider_sha256": _mapping(identity.get("runtime")).get("completion_sha256"),
        **{
            name: fields.get(name)
            for name in (
                "threads",
                "threads_batch",
                "context",
                "vocab_size",
                "chat_template_applied",
            )
        },
    }


def _evidence_gaps(record: RunRecord, expected_provider: str) -> tuple[str, ...]:
    gaps: list[str] = []
    identity = _mapping(record.observed.get("identity"))
    model = _mapping(identity.get("model"))
    if identity.get("status") == "mismatch":
        gaps.append("runtime identity is inconsistent or mismatched")
    expected_model = record.requested.model.get("sha256", "")
    if (
        model.get("status") != "verified"
        or model.get("sha256") != str(expected_model).lower()
    ):
        gaps.append("pinned model content was not verified")
    runtime = _mapping(identity.get("runtime"))
    if runtime.get("completion_sha256") != expected_provider.lower():
        gaps.append("runtime does not match the qualified capture provider")
    invocations = _invocations(record)
    if not invocations:
        gaps.append("no runtime invocation evidence was retained")
    for invocation in invocations:
        capture = (
            invocation.get("runtime_capture")
            if isinstance(invocation, Mapping)
            else None
        )
        if not isinstance(capture, Mapping):
            gaps.append("structured runtime readback is missing")
            continue
        for name in ("threads", "threads_batch"):
            if capture.get(name) != record.requested.runtime.get(name):
                gaps.append(f"runtime-applied {name} does not match the request")
        generation = _mapping(invocation.get("generation"))
        if capture.get("context") != generation.get("context"):
            gaps.append("runtime-applied context does not match the request")
        if capture.get("chat_template_applied") is not False:
            gaps.append("plain completion was not confirmed by the runtime")
    coverage = record.metrics.get("trajectory_nonempty_capture_rate")
    if coverage is None or isinstance(coverage.value, bool) or coverage.value != 1.0:
        gaps.append("every prompt must retain non-empty sampled token IDs")
    return tuple(dict.fromkeys(gaps))


class _CheckedAnalysis:
    def __init__(
        self,
        analysis: PairwiseAnalysis,
        evidence_gaps: Callable[[RunRecord], tuple[str, ...]],
    ) -> None:
        self.name, self.version = analysis.name, analysis.version
        self._analysis, self._gaps = analysis, evidence_gaps

    def analyze(self, left: RunRecord, right: RunRecord) -> MeasurementResult:
        if self._gaps(left) or self._gaps(right):
            raise ValueError(
                "verification evidence is insufficient for behavioral analysis"
            )
        return self._analysis.analyze(left, right)


def _write_atomic(path: Path, text: str) -> None:
    temporary = path.with_name("." + path.name + ".tmp")
    created = False
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            created = True
            stream.write(text)
        if path.exists():
            raise FileExistsError(f"verification artifact already exists: {path.name}")
        temporary.replace(path)
    finally:
        if created:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def _comparison_data(report: CompatibilityReport) -> dict[str, Any]:
    # Values may contain private configuration. Keep paths and reasons only.
    return {
        "compatible": report.compatible,
        "issues": [
            {"dimension": issue.dimension, "reason": issue.reason}
            for issue in report.issues
        ],
        "waived_differences": [
            {
                "dimension": issue.dimension,
                "reason_sha256": hashlib.sha256(
                    issue.reason.encode("utf-8")
                ).hexdigest(),
            }
            for issue in report.waived_differences
        ],
        "comparable_metrics": list(report.comparable_metrics),
        "incompatible_metrics": dict(report.incompatible_metrics),
    }


def _analysis_data(outcome: StudyPairAnalysis) -> dict[str, Any]:
    result = outcome.result
    metrics: dict[str, Any] = {}
    metric_errors: dict[str, Any] = {}
    if result is not None:
        for key, value in result.metrics.items():
            try:
                metrics[key] = _metric_summary_to_data(value, key=key)
            except (TypeError, ValueError, AttributeError) as exc:
                metric_errors[key] = {
                    "error_type": type(exc).__name__,
                    "message_sha256": hashlib.sha256(
                        str(exc).encode("utf-8")
                    ).hexdigest(),
                }
    return {
        "name": outcome.name,
        "version": outcome.version,
        "status": outcome.status.value,
        "reason": outcome.reason,
        "error_type": outcome.error_type,
        "message_sha256": outcome.message_sha256,
        "metrics": metrics,
        "metric_errors": metric_errors,
        "diagnostics": {}
        if result is None
        else _json_value(result.evidence, path="analysis.evidence"),
    }


def _wall_time(record: RunRecord) -> dict[str, Any]:
    values: list[float] = []
    for row in _invocations(record):
        value = _mapping(row).get("duration_seconds")
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            return {"available": False}
        values.append(float(value))
    if not values:
        return {"available": False}
    return {
        "available": True,
        "method": "llamacpp.subprocess_monotonic.v1",
        "includes": "process startup, model loading, prompt evaluation and generation",
        "samples_seconds": values,
        "mean_seconds": sum(values) / len(values),
    }


def _implementation_identity(scope: str) -> dict[str, str]:
    from . import __version__

    implementation = {"name": "metria.verify", "version": __version__}
    if scope != VERIFICATION_SCOPE:
        from .verification_trials import _implementation_digest

        implementation["source_sha256"] = _implementation_digest()
    return implementation


def _legacy_performance(
    left: RunRecord, right: RunRecord, comparable: bool
) -> dict[str, Any]:
    return compare_performance(
        measure_invocation_performance(left),
        measure_invocation_performance(right),
        comparable=comparable,
    )


def _verification_route(recipe: StudyRecipe) -> VerificationRoute:
    if recipe.study.runs and any(
        run.trial_policy.get("method") == "metria.concurrent_serving_trials.v1"
        for run in recipe.study.runs
    ):
        from .verification_serving import build_route as build_serving_route

        return build_serving_route(recipe)
    if "runtime_environments" in recipe.environment:
        from .verification_upgrade import build_route as build_upgrade_route

        return build_upgrade_route(recipe)
    if "gguf_quantization" in recipe.environment:
        from .verification_quantization import build_route as build_quantization_route

        return build_quantization_route(recipe)
    if "llama_cpp_capture_providers" in recipe.environment:
        from .verification_build import build_route as build_llamacpp_route

        return build_llamacpp_route(recipe)
    if recipe.study.runs and all(
        run.runtime.get("name") == "vllm" for run in recipe.study.runs
    ):
        from .verification_vllm import build_route

        return build_route(recipe)
    registries = _builtin_registries()
    _validate(recipe, registries)
    provider = str(recipe.environment[_CAPTURE_KEY])
    return VerificationRoute(
        scope=VERIFICATION_SCOPE,
        adapters=registries.adapters,
        measurements=registries.measurements,
        analyses=registries.analyses,
        evidence_gaps=lambda record: _evidence_gaps(record, provider),
        observed_facts=_observed_facts,
        change={
            "reference_threads": recipe.study.runs[0].runtime["threads"],
            "candidate_threads": recipe.study.runs[1].runtime["threads"],
        },
        performance=_legacy_performance,
    )


def verify_recipe(
    recipe: StudyRecipe,
    output_dir: str | Path = "verification",
    *,
    capability_checks: CapabilityCheckRegistry | None = None,
) -> VerificationResult:
    """Validate one qualified profile and retain its reference/candidate evidence."""
    resolve_capability_checks(capability_checks)
    route = _verification_route(recipe)
    if (
        route.isolated or route.run_executor is not None
    ) and capability_checks is not None:
        raise ValueError(
            "qualified vLLM verification uses the built-in capability registry"
        )
    if route.isolated:
        from .verification_worker import verify_isolated

        return verify_isolated(recipe, output_dir, route)
    return _verify_with_profile(
        recipe, output_dir, route, capability_checks=capability_checks
    )


def _verify_with_profile(
    recipe: StudyRecipe,
    output_dir: str | Path,
    route: VerificationRoute,
    *,
    reserved_output: bool = False,
    defer_result: bool = False,
    capability_checks: CapabilityCheckRegistry | None = None,
) -> VerificationResult:
    """Execute the first local CPU verifier and persist each run immediately.

    The output directory must be new. Completed files survive later execution
    interruption or write failure; the manifest is written last and is never
    presented as complete when persistence failed.
    """
    resolve_capability_checks(capability_checks)
    registries = route
    recipe_digest = study_recipe_digest(recipe)
    hardware = capture_hardware_fingerprint().to_mapping()
    implementation = _implementation_identity(route.scope)
    context = {
        "recipe_digest": recipe_digest,
        "scope": route.scope,
        "implementation": implementation,
        "hardware": hardware,
    }
    output = Path(output_dir).expanduser()
    if not reserved_output:
        output.mkdir(parents=True, exist_ok=False)
    saved: list[RunRecord] = []

    def save_record(record: RunRecord) -> None:
        decorated = replace(
            record, provenance={**record.provenance, "verification": context}
        )
        _write_atomic(
            output / f"{_ROLES[len(saved)]}.run.json",
            run_record_to_json(decorated) + "\n",
        )
        saved.append(decorated)

    analyses = {
        name: _CheckedAnalysis(analysis, route.evidence_gaps)
        for name, analysis in registries.analyses.items()
    }
    interrupted = False
    try:
        execution = execute_study(
            recipe.study,
            adapters=registries.adapters,
            measurements=registries.measurements,
            measurement_configs=recipe.measurement_configs,
            environment=recipe.environment,
            analyses=analyses,
            record_sink=save_record,
            capability_checks=capability_checks,
            _run_executor=route.run_executor,
        )
        comparison = execution.comparisons[0].report
        outcomes = execution.comparisons[0].analyses
    except KeyboardInterrupt:
        interrupted = True
        first_missing = len(saved)
        for index in range(first_missing, 2):
            save_record(
                RunRecord(
                    study_name=recipe.study.name,
                    run_id=f"run-{index:04d}",
                    requested=recipe.study.runs[index],
                    resolved={},
                    observed={},
                    status=RunStatus.INTERRUPTED
                    if index == first_missing
                    else RunStatus.CANCELLED,
                    events=({"stage": "verification", "kind": "interrupted"},),
                )
            )
        comparison = compare_runs(saved[0], saved[1], recipe.study.comparison)
        outcomes = ()

    analysis_data = [_analysis_data(outcome) for outcome in outcomes]
    gaps = [route.evidence_gaps(record) for record in saved]
    failed = {
        RunStatus.PREFLIGHT_FAILED,
        RunStatus.FAILED,
        RunStatus.TIMED_OUT,
        RunStatus.INTERRUPTED,
        RunStatus.CANCELLED,
    }
    if interrupted or any(record.status in failed for record in saved):
        verdict = VerificationVerdict.EXECUTION_FAILED
    elif any(record.status is not RunStatus.COMPLETED for record in saved) or any(gaps):
        verdict = VerificationVerdict.INSUFFICIENT_EVIDENCE
    elif not comparison.compatible:
        verdict = VerificationVerdict.NOT_COMPARABLE
    elif not outcomes or any(
        outcome.status is not PairwiseAnalysisStatus.COMPLETED for outcome in outcomes
    ):
        verdict = VerificationVerdict.EXECUTION_FAILED
    elif any(analysis["metric_errors"] for analysis in analysis_data):
        verdict = VerificationVerdict.INSUFFICIENT_EVIDENCE
    else:
        verdict = VerificationVerdict.VERIFIED
    policy_result = None
    if recipe.policy is not None:
        policy_result = evaluate_policy(
            recipe.policy, outcomes, verification_status=verdict.value
        )
        if policy_result.status is not PolicyDecision.NOT_EVALUATED:
            verdict = VerificationVerdict(policy_result.status.value)
    exit_code = 130 if interrupted else VERIFICATION_EXIT_CODES[verdict.value]
    manifest = {
        "schema": VERIFICATION_SCHEMA,
        "scope": route.scope,
        "study": recipe.study.name,
        "recipe_digest": recipe_digest,
        "implementation": context["implementation"],
        "hardware": hardware,
        "verdict": verdict.value,
        "exit_code": exit_code,
        "acceptance_policy_evaluated": policy_result is not None
        and policy_result.status is not PolicyDecision.NOT_EVALUATED,
        "lifecycle": {
            "status": (
                "failed"
                if any(
                    record.status in failed
                    or record.status is RunStatus.PREFLIGHT_FAILED
                    for record in saved
                )
                else "partial"
                if any(record.status is not RunStatus.COMPLETED for record in saved)
                else "completed"
            ),
            "records": {
                role: record.status.value
                for role, record in zip(_ROLES, saved, strict=True)
            },
        },
        "comparison_status": (
            "NOT_EVALUATED"
            if any(gaps)
            or any(record.status is not RunStatus.COMPLETED for record in saved)
            else "VALID"
            if comparison.compatible
            else "NOT_COMPARABLE"
        ),
        "policy_status": policy_result.status.value
        if policy_result is not None
        else "NOT_CONFIGURED",
        "change": route.change,
        "comparison_plan": {
            "vary": sorted(recipe.study.comparison.vary),
            "control": sorted(recipe.study.comparison.control),
            "block_by": sorted(recipe.study.comparison.block_by),
        },
        "records": {
            role: {
                "path": f"{role}.run.json",
                "run_id": record.run_id,
                "status": record.status.value,
                "record_digest": run_record_digest(record),
                "evidence_digest": run_evidence_digest(record),
                "evidence_gaps": list(gaps[index]),
                "observed": route.observed_facts(record),
            }
            for index, (role, record) in enumerate(zip(_ROLES, saved, strict=True))
        },
        "comparison": _comparison_data(comparison),
        "controls": [
            {
                "dimension": dimension,
                "role": role,
                "status": "matched"
                if issue is None
                else "missing"
                if "missing" in issue.reason or "unknown" in issue.reason
                else "different",
            }
            for role, dimensions in (
                ("control", recipe.study.comparison.control),
                ("block", recipe.study.comparison.block_by),
            )
            for dimension in sorted(dimensions)
            for issue in (
                _declared_dimension_issue(saved[0], saved[1], dimension, role),
            )
        ],
        "analyses": analysis_data,
        "systems": {
            role: _wall_time(record) for role, record in zip(_ROLES, saved, strict=True)
        },
        "performance": route.performance(
            saved[0],
            saved[1],
            comparison.compatible
            and not any(gaps)
            and all(record.status is RunStatus.COMPLETED for record in saved),
        ),
    }
    if policy_result is not None:
        manifest["policy"] = policy_result.to_data()
    if defer_result:
        _write_atomic(
            output / "worker-result.json",
            json.dumps(
                _json_value(manifest, path="verification"),
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
        )
    else:
        _persist_verification(output, manifest)
    return VerificationResult(
        output,
        manifest,
        exit_code,
    )


def _persist_verification(output: Path, manifest: Mapping[str, Any]) -> None:
    _write_atomic(output / "report.md", render_verification(manifest))
    serialized = (
        json.dumps(
            _json_value(manifest, path="verification"),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    # Keep the original name as a compatibility alias. The canonical result is
    # published last, so its presence denotes a fully persisted evidence bundle.
    _write_atomic(output / "manifest.json", serialized)
    _write_atomic(output / "verification.json", serialized)
