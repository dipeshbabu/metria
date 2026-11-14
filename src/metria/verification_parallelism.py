"""STAGED: vLLM tensor/pipeline parallelism verification on local GPUs.

``local_vllm_parallelism.v1`` compares a single-device reference with a
candidate that splits the same model across devices (tensor and/or pipeline
parallel). Every other input is identical. The engine must read back the
requested parallel sizes for both runs, and the candidate's host must expose at
least as many CUDA devices as it uses. Parallelism changes the order of
floating-point reductions, so the profile always requires task checks and an
explicit policy, and ends in PASS or FAIL.

No native multi-GPU run has qualified this profile yet.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from functools import partial
from typing import Any

from .measurements import TokenTrajectoryProtocol, TrajectoryDivergenceAnalysis
from .measurements.prefix_distributions import staged_distributions
from .measurements.prefix_workload import PrefixWorkloadProtocol
from .measurements.task_checks import prompt_checks
from .measurements.trajectory_analysis import TrajectoryAgreementAnalysis
from .measurements.verification_impact import VerificationImpactAnalysis
from .models import RunRecord, RunSpec
from .recipes import StudyRecipe
from .runtimes.vllm import VLLMAdapter
from .verification_evidence import task_check_identity
from .verification_kv_precision import _analyses
from .verification_route import VerificationRoute
from .verification_vllm import (
    VLLM_VARIATIONS,
    _mapping,
    _validate_run,
    _workload_gaps,
    identity_gaps,
    observed_facts,
    trial_identity,
)

PARALLELISM_SCOPE = "local_vllm_parallelism.v1"
PARALLEL_KEYS = ("tensor_parallel_size", "pipeline_parallel_size")
QUALIFIED_PARALLEL_VERSIONS = frozenset({"0.30.0+cu129"})
MAX_DEVICES = 8


def parallel_sizes(run: RunSpec) -> dict[str, int]:
    """Requested sizes, defaulting to 1, for the keys a run declares."""

    return {key: run.runtime.get(key, 1) for key in PARALLEL_KEYS}


def is_parallelism_recipe(recipe: StudyRecipe) -> bool:
    return any(
        size != 1 for run in recipe.study.runs for size in parallel_sizes(run).values()
    )


def parallel_vary(left: RunSpec, right: RunSpec) -> frozenset[str]:
    """Every record path a requested parallel size moves."""

    paths: set[str] = set()
    for key in PARALLEL_KEYS:
        if parallel_sizes(left)[key] != parallel_sizes(right)[key]:
            paths |= {
                f"runtime.{key}",
                f"resolved.runtime.settings.{key}",
                f"observed.configured.runtime.{key}",
                f"observed.applied.fields.parallel.{key}",
                f"observed.identity.applied.fields.parallel.{key}",
            }
    return frozenset(paths)


def _single_device(run: RunSpec) -> RunSpec:
    runtime = {k: v for k, v in run.runtime.items() if k != "pipeline_parallel_size"}
    runtime["tensor_parallel_size"] = 1
    return replace(run, runtime=runtime)


def _validate_sizes(left: RunSpec, right: RunSpec) -> None:
    declared = [
        {key for key in PARALLEL_KEYS if key in run.runtime} for run in (left, right)
    ]
    if declared[0] != declared[1]:
        raise ValueError("both runs must declare the same parallel settings")
    for run in (left, right):
        for key, size in parallel_sizes(run).items():
            if isinstance(size, bool) or not isinstance(size, int) or size < 1:
                raise ValueError(f"runtime.{key} must be a positive integer")
    reference, candidate = parallel_sizes(left), parallel_sizes(right)
    devices = candidate["tensor_parallel_size"] * candidate["pipeline_parallel_size"]
    if any(size != 1 for size in reference.values()) or not 2 <= devices <= MAX_DEVICES:
        raise ValueError(
            "parallelism verification compares a single-device reference with a "
            f"candidate spanning 2-{MAX_DEVICES} devices"
        )


def validate_parallelism_recipe(
    recipe: StudyRecipe,
) -> tuple[dict[str, int], dict[str, int]]:
    if len(recipe.study.runs) != 2:
        raise ValueError("verify requires exactly two runs: reference, then candidate")
    left, right = recipe.study.runs
    _validate_sizes(left, right)
    comparison = recipe.study.comparison
    if comparison.vary != parallel_vary(left, right) or comparison.waivers:
        raise ValueError(
            "parallelism verification varies only the requested parallel sizes"
        )
    if set(recipe.measurement_configs) != {TokenTrajectoryProtocol.name}:
        raise ValueError("parallelism verification requires one workload")
    config = recipe.measurement_configs[TokenTrajectoryProtocol.name]
    if comparison.analyses != _analyses(config):
        raise ValueError(
            "parallelism verification requires trajectory and impact analyses, "
            "plus metria.trajectory_kld exactly when staged_distributions is true"
        )
    if recipe.policy is None:
        raise ValueError("parallelism verification requires an explicit policy")
    if not any(prompt_checks(config.get("prompts", ()))):
        raise ValueError("parallelism verification requires declared task checks")
    from .verification_vllm import validate_vllm_recipe

    single = (_single_device(left), _single_device(right))
    validate_vllm_recipe(
        replace(
            recipe,
            policy=None,
            study=replace(
                recipe.study,
                runs=single,
                comparison=replace(
                    comparison, vary=VLLM_VARIATIONS, analyses=comparison.analyses[:2]
                ),
            ),
        ),
        check_prefix_change=False,
    )
    for run in single:
        _validate_run(run, config, versions=QUALIFIED_PARALLEL_VERSIONS)
        if run.runtime["enable_prefix_caching"] is not False:
            raise ValueError("parallelism verification requires prefix caching off")
    if single[0].runtime != single[1].runtime:
        raise ValueError("only the parallel sizes may differ between runs")
    return parallel_sizes(left), parallel_sizes(right)


def _parallel_gaps(record: RunRecord) -> list[str]:
    requested = parallel_sizes(record.requested)
    gaps = []
    identity = _mapping(record.observed.get("identity"))
    for applied in (record.observed.get("applied"), identity.get("applied")):
        fields = _mapping(_mapping(applied).get("fields"))
        for key, size in requested.items():
            if key == "pipeline_parallel_size" and key not in record.requested.runtime:
                continue
            if fields.get(f"parallel.{key}") != size:
                gaps.append(f"runtime-applied {key} does not match the request")
    devices = _mapping(record.observed.get("hardware")).get("devices")
    needed = requested["tensor_parallel_size"] * requested["pipeline_parallel_size"]
    if not isinstance(devices, (list, tuple)) or len(devices) < needed:
        gaps.append("the host did not expose enough CUDA devices for the request")
    return gaps


def evidence_gaps(
    record: RunRecord,
    *,
    runtime_pin: str,
    trials: Mapping[str, Any],
    quality_identity: Mapping[str, Any] | None,
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            (
                *identity_gaps(
                    record, runtime_pin=runtime_pin, quality_identity=quality_identity
                ),
                *_workload_gaps(record, trials),
                *_parallel_gaps(record),
            )
        )
    )


def parallel_observed_facts(record: RunRecord) -> dict[str, Any]:
    facts = observed_facts(record)
    fields = _mapping(_mapping(record.observed.get("applied")).get("fields"))
    devices = _mapping(record.observed.get("hardware")).get("devices")
    facts.update(
        tensor_parallel_size=fields.get("parallel.tensor_parallel_size"),
        pipeline_parallel_size=fields.get("parallel.pipeline_parallel_size"),
        device_count=len(devices) if isinstance(devices, (list, tuple)) else None,
    )
    return facts


def build_route(recipe: StudyRecipe) -> VerificationRoute:
    from .measurements.prefix_performance import compare_prefix_performance

    reference, candidate = validate_parallelism_recipe(recipe)
    measurement = PrefixWorkloadProtocol()
    config = recipe.measurement_configs[measurement.name]
    trials = trial_identity(config)
    analyses: dict[str, Any] = {
        TrajectoryAgreementAnalysis.name: TrajectoryAgreementAnalysis(),
        VerificationImpactAnalysis.name: VerificationImpactAnalysis(),
    }
    if staged_distributions(config):
        analyses[TrajectoryDivergenceAnalysis.name] = TrajectoryDivergenceAnalysis()
    return VerificationRoute(
        scope=PARALLELISM_SCOPE,
        adapters={"vllm": VLLMAdapter()},
        measurements={measurement.name: measurement},
        analyses=analyses,
        evidence_gaps=partial(
            evidence_gaps,
            runtime_pin=recipe.environment["vllm_distribution_sha256"],
            trials=trials,
            quality_identity=task_check_identity(config, trials["measured_trials"]),
        ),
        observed_facts=parallel_observed_facts,
        change={
            "parameter": "parallelism",
            "reference": reference,
            "candidate": candidate,
        },
        performance=compare_prefix_performance,
        isolated=True,
    )
