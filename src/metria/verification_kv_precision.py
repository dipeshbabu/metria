"""Qualified vLLM KV-cache precision verification (automatic cache vs FP8).

The reference requests the model's automatic KV-cache dtype and the candidate
an FP8 dtype; every other input is identical. The engine's own configuration
must report the requested cache dtype for both runs. Because FP8 is a lossy
change, this profile always requires declared task checks and an explicit
acceptance policy, so it ends in PASS or FAIL rather than VERIFIED.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from functools import partial
from typing import Any

from .measurements import (
    TokenTrajectoryProtocol,
    TrajectoryAgreementAnalysis,
    TrajectoryDivergenceAnalysis,
)
from .measurements.prefix_distributions import (
    DISTRIBUTIONS_KEY,
    staged_distributions,
)
from .measurements.prefix_workload import PrefixWorkloadProtocol
from .measurements.task_checks import prompt_checks, task_workload_identity
from .measurements.verification_impact import VerificationImpactAnalysis
from .models import RunRecord, TreatmentType
from .recipes import StudyRecipe
from .runtimes.vllm import VLLMAdapter
from .verification_route import VerificationRoute
from .verification_schema import VLLM_KV_PRECISION_SCOPE
from .verification_vllm import (
    _mapping,
    _validate_run,
    _workload_gaps,
    identity_gaps,
    observed_facts,
    trial_identity,
)

KV_PRECISION_VARIATIONS = frozenset(
    {
        "treatments",
        "resolved.kv_cache.dtype",
        "observed.configured.kv_cache.dtype",
        "observed.applied.fields.cache.cache_dtype",
        "observed.identity.applied.fields.cache.cache_dtype",
    }
)
FP8_KV_DTYPES = frozenset({"fp8", "fp8_e4m3", "fp8_e5m2"})
QUALIFIED_KV_VERSIONS = frozenset({"0.30.0+cu129"})
_TREATMENT = "vllm.kv_cache"


def is_kv_precision_recipe(recipe: StudyRecipe) -> bool:
    """Return whether a vLLM recipe declares any KV-cache treatment."""

    return any(
        treatment.name == _TREATMENT
        for run in recipe.study.runs
        for treatment in run.treatments
    )


def _kv_dtype(run: Any) -> str:
    treatments = run.treatments
    if (
        len(treatments) != 1
        or treatments[0].name != _TREATMENT
        or treatments[0].kind is not TreatmentType.RUNTIME_FEATURE
        or set(treatments[0].config) != {"dtype"}
    ):
        raise ValueError(
            "KV-precision verification requires exactly one vllm.kv_cache "
            "runtime_feature treatment with only a dtype"
        )
    dtype = treatments[0].config["dtype"]
    if not isinstance(dtype, str):
        raise TypeError("vllm.kv_cache dtype must be a string")
    return dtype


def _analyses(config: Any) -> tuple[str, ...]:
    """Qualified analyses, plus STAGED trajectory KLD when distributions are on."""

    names: tuple[str, ...] = (
        TrajectoryAgreementAnalysis.name,
        VerificationImpactAnalysis.name,
    )
    if isinstance(config, Mapping) and staged_distributions(config):
        names += (TrajectoryDivergenceAnalysis.name,)
    return names


def validate_kv_precision_recipe(recipe: StudyRecipe) -> tuple[str, str]:
    """Validate the profile and return the reference and candidate dtypes."""

    if len(recipe.study.runs) != 2:
        raise ValueError("verify requires exactly two runs: reference, then candidate")
    comparison = recipe.study.comparison
    if comparison.vary != KV_PRECISION_VARIATIONS or comparison.waivers:
        raise ValueError(
            "KV-precision verification permits only its five KV-cache "
            "variation paths and no waivers"
        )
    if set(recipe.measurement_configs) != {TokenTrajectoryProtocol.name}:
        raise ValueError(
            "KV-precision verification requires exactly one workload configuration"
        )
    if comparison.analyses != _analyses(
        recipe.measurement_configs[TokenTrajectoryProtocol.name]
    ):
        raise ValueError(
            "KV-precision verification requires trajectory and impact analyses, "
            "plus metria.trajectory_kld exactly when staged_distributions is true"
        )
    if recipe.policy is None:
        raise ValueError(
            "KV-precision verification requires an explicit acceptance policy"
        )
    config = recipe.measurement_configs[TokenTrajectoryProtocol.name]
    if not any(prompt_checks(config.get("prompts", ()))):
        raise ValueError(
            "KV-precision verification requires declared task checks on the workload"
        )
    left, right = recipe.study.runs
    reference, candidate = _kv_dtype(left), _kv_dtype(right)
    if reference != "auto" or candidate not in FP8_KV_DTYPES:
        raise ValueError(
            "KV-precision verification compares an automatic reference cache "
            "with an FP8 candidate cache"
        )
    from .verification_vllm import validate_vllm_recipe

    # Reuse every prefix-profile control after removing the declared treatment.
    validate_vllm_recipe(
        replace(
            recipe,
            policy=None,
            study=replace(
                recipe.study,
                runs=(replace(left, treatments=()), replace(right, treatments=())),
                comparison=replace(
                    comparison,
                    analyses=comparison.analyses[:2],
                    vary=frozenset(
                        {
                            "runtime.enable_prefix_caching",
                            "resolved.runtime.settings.enable_prefix_caching",
                            "observed.configured.runtime.enable_prefix_caching",
                            "observed.applied.fields.cache.enable_prefix_caching",
                            "observed.identity.applied.fields.cache.enable_prefix_caching",
                        }
                    ),
                ),
            ),
        ),
        check_prefix_change=False,
    )
    for run in (left, right):
        _validate_run(
            replace(run, treatments=()), config, versions=QUALIFIED_KV_VERSIONS
        )
        if run.runtime["enable_prefix_caching"] is not False:
            raise ValueError("KV-precision verification requires prefix caching off")
    if left.runtime != right.runtime:
        raise ValueError(
            "KV-precision verification requires identical runtime settings"
        )
    return reference, candidate


def _kv_gaps(record: RunRecord) -> list[str]:
    requested = _kv_dtype(record.requested)
    gaps = []
    if _mapping(record.resolved.get("kv_cache")).get("dtype") != requested:
        gaps.append("resolved KV-cache dtype differs from the request")
    identity = _mapping(record.observed.get("identity"))
    for applied in (record.observed.get("applied"), identity.get("applied")):
        fields = _mapping(_mapping(applied).get("fields"))
        if fields.get("cache.cache_dtype") != requested:
            gaps.append("runtime-applied KV-cache dtype does not match the request")
    return gaps


def _distribution_gaps(record: RunRecord) -> list[str]:
    """STAGED: require raw top-k evidence for every native call when opted in."""

    capture = _mapping(
        _mapping(record.evidence.get("measurements")).get(TokenTrajectoryProtocol.name)
    )
    retained = _mapping(capture.get(DISTRIBUTIONS_KEY))
    if retained.get("qualification") != "staged" or retained.get(
        "n_prompts"
    ) != capture.get("n_prompts"):
        return ["staged distribution evidence is missing or incomplete"]
    invocations = record.observed.get("invocations", ())
    if not invocations or any(
        _mapping(_mapping(row).get("top_logprobs")).get("mode") != "raw_logprobs"
        for row in invocations
    ):
        return ["native calls did not retain raw top-k logprob provenance"]
    return []


def evidence_gaps(
    record: RunRecord,
    *,
    runtime_pin: str,
    trials: Mapping[str, Any],
    quality_identity: Mapping[str, Any],
    distributions: bool = False,
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            (
                *identity_gaps(
                    record, runtime_pin=runtime_pin, quality_identity=quality_identity
                ),
                *_workload_gaps(record, trials),
                *_kv_gaps(record),
                *(_distribution_gaps(record) if distributions else ()),
            )
        )
    )


def kv_observed_facts(record: RunRecord) -> dict[str, Any]:
    facts = observed_facts(record)
    identity = _mapping(record.observed.get("identity"))
    fields = _mapping(_mapping(identity.get("applied")).get("fields"))
    facts["kv_cache_dtype"] = fields.get("cache.cache_dtype")
    return facts


def build_route(recipe: StudyRecipe) -> VerificationRoute:
    from .measurements.prefix_performance import compare_prefix_performance

    reference, candidate = validate_kv_precision_recipe(recipe)
    measurement = PrefixWorkloadProtocol()
    config = recipe.measurement_configs[measurement.name]
    trials = trial_identity(config)
    checks = prompt_checks(config["prompts"])
    quality_identity = {
        "sha256": task_workload_identity(
            checks, config["prompts"], trials["measured_trials"]
        ),
        "check_count": sum(len(row) for row in checks) * trials["measured_trials"],
    }
    analysis = TrajectoryAgreementAnalysis()
    distributions = staged_distributions(config)
    analyses: dict[str, Any] = {
        analysis.name: analysis,
        VerificationImpactAnalysis.name: VerificationImpactAnalysis(),
    }
    if distributions:
        analyses[TrajectoryDivergenceAnalysis.name] = TrajectoryDivergenceAnalysis()
    return VerificationRoute(
        scope=VLLM_KV_PRECISION_SCOPE,
        adapters={"vllm": VLLMAdapter()},
        measurements={measurement.name: measurement},
        analyses=analyses,
        evidence_gaps=partial(
            evidence_gaps,
            runtime_pin=recipe.environment["vllm_distribution_sha256"],
            trials=trials,
            quality_identity=quality_identity,
            distributions=distributions,
        ),
        observed_facts=kv_observed_facts,
        change={
            "parameter": "kv_cache_dtype",
            "reference": reference,
            "candidate": candidate,
        },
        performance=compare_prefix_performance,
        isolated=True,
    )
