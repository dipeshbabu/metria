"""Qualified local vLLM CPU runtime-stack upgrade with separate interpreters."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from .measurements import TokenTrajectoryProtocol, TrajectoryAgreementAnalysis
from .measurements.prefix_performance import compare_prefix_performance
from .measurements.prefix_workload import PrefixWorkloadProtocol, trajectory_config
from .measurements.verification_impact import VerificationImpactAnalysis
from .models import RunRecord, RunSpec
from .recipes import StudyRecipe
from .runtime_environment import source_digest, validate_environment
from .runtime_executor import RuntimeRunExecutor
from .runtimes.vllm import VLLMAdapter
from .verification_evidence import task_check_identity
from .verification_route import VerificationRoute
from .verification_schema import VLLM_UPGRADE_SCOPE
from .verification_vllm import (
    _mapping,
    _validate_run,
    evidence_gaps,
    observed_facts,
    trial_identity,
)

ENVIRONMENTS_KEY = "runtime_environments"
UPGRADE_VERSIONS = ("0.29.0+cpu", "0.30.0+cpu")
UPGRADE_VARIATIONS = frozenset(
    {
        "runtime.version",
        "environment_selector",
        "resolved.runtime.version",
        "resolved.runtime.requested_version",
        "resolved.runtime.artifact",
        "resolved.runtime_environment",
        "observed.runtime.version",
        "observed.identity.runtime.version",
        "observed.identity.runtime.distribution_version",
        "observed.identity.runtime.module_version",
        "observed.artifacts.runtime",
        "observed.runtime_environment",
        "observed.environment.vllm_distribution_sha256",
    }
)


def validate_upgrade_run(run: RunSpec, config: Mapping[str, Any]) -> None:
    if set(run.environment_selector) != {
        "runtime_environment"
    } or run.environment_selector["runtime_environment"] not in {
        "reference",
        "candidate",
    }:
        raise ValueError(
            "upgrade runs require an explicit reference/candidate environment selector"
        )
    # Placement has its own strict descriptor contract; reuse all unchanged
    # native runtime, model, generation and workload checks.
    _validate_run(
        replace(run, environment_selector={}),
        config,
        versions=frozenset(UPGRADE_VERSIONS),
    )
    if (
        run.runtime["enable_prefix_caching"] is not False
        or run.runtime["dtype"] != "float32"
    ):
        raise ValueError(
            "qualified CPU upgrades require float32 with prefix caching disabled in both runs"
        )


def validate_upgrade_recipe(
    recipe: StudyRecipe,
) -> tuple[Mapping[str, Any], tuple[int, int]]:
    if len(recipe.study.runs) != 2:
        raise ValueError("runtime upgrades require reference and candidate runs")
    if (
        recipe.study.comparison.vary != UPGRADE_VARIATIONS
        or recipe.study.comparison.waivers
    ):
        raise ValueError(
            "runtime upgrades permit only their declared environment/version differences and no waivers"
        )
    if recipe.study.comparison.analyses not in {
        (TrajectoryAgreementAnalysis.name,),
        (TrajectoryAgreementAnalysis.name, VerificationImpactAnalysis.name),
    }:
        raise ValueError("runtime upgrades require the registered trajectory analyses")
    if set(recipe.environment) != {
        ENVIRONMENTS_KEY,
        "verification_timeout_s",
        "cpu_binding",
    }:
        raise ValueError(
            "runtime upgrades require two pinned environments, CPU binding and a shared deadline"
        )
    targets = recipe.environment[ENVIRONMENTS_KEY]
    if not isinstance(targets, Mapping) or set(targets) != {"reference", "candidate"}:
        raise ValueError("exactly two pinned runtime environments are required")
    code = source_digest()
    for role, version in zip(("reference", "candidate"), UPGRADE_VERSIONS, strict=True):
        target = validate_environment(targets[role], metria_sha256=code)
        if target["runtime"]["version"] != version:
            raise ValueError(
                "the qualified CPU upgrade is vLLM 0.29.0+cpu to 0.30.0+cpu"
            )
    if (
        targets["reference"]["prefix"] == targets["candidate"]["prefix"]
        or targets["reference"]["python_sha256"]
        != targets["candidate"]["python_sha256"]
    ):
        raise ValueError(
            "use separate runtime prefixes with the same Python interpreter content"
        )
    binding = _cpu_binding(recipe.environment["cpu_binding"], targets)
    timeout = recipe.environment["verification_timeout_s"]
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or not 1 <= timeout <= 3600
    ):
        raise ValueError("verification_timeout_s must be between 1 and 3600 seconds")
    _validate_pair(recipe, targets)
    return targets, binding


def _cpu_binding(value: Any, targets: Mapping[str, Any]) -> tuple[int, int]:
    if (
        not isinstance(value, (tuple, list))
        or len(value) != 2
        or any(type(core) is not int or core < 0 for core in value)
        or value[0] >= value[1]
    ):
        raise ValueError("CPU upgrade binding requires two distinct ordered CPU IDs")
    if any(
        not set(value) <= set(target["cpu_affinity"]) for target in targets.values()
    ):
        raise ValueError("CPU binding is outside the observed allowed affinity")
    return value[0], value[1]


def _validate_pair(recipe: StudyRecipe, targets: Mapping[str, Any]) -> None:
    if set(recipe.measurement_configs) != {TokenTrajectoryProtocol.name}:
        raise ValueError("runtime upgrades require one trajectory workload")
    config = recipe.measurement_configs[TokenTrajectoryProtocol.name]
    trajectory_config(config)
    for role, run in zip(("reference", "candidate"), recipe.study.runs, strict=True):
        validate_upgrade_run(run, config)
        if (
            run.environment_selector["runtime_environment"] != role
            or run.runtime["version"] != targets[role]["runtime"]["version"]
        ):
            raise ValueError(
                "run version and environment do not match their declared role"
            )
    left, right = recipe.study.runs
    if (
        left.model != right.model
        or left.scenario != right.scenario
        or left.trial_policy != right.trial_policy
    ):
        raise ValueError(
            "runtime upgrades require identical model/tokenizer, workload and trial controls"
        )
    if {k: v for k, v in left.runtime.items() if k != "version"} != {
        k: v for k, v in right.runtime.items() if k != "version"
    }:
        raise ValueError(
            "only the requested runtime version and pinned environment may differ"
        )


def build_route(recipe: StudyRecipe) -> VerificationRoute:
    targets, cores = validate_upgrade_recipe(recipe)
    config = recipe.measurement_configs[TokenTrajectoryProtocol.name]
    trials = trial_identity(config)
    quality = task_check_identity(config, trials["measured_trials"])
    binding = (
        f"{cores[0]}-{cores[1]}"
        if cores[1] == cores[0] + 1
        else f"{cores[0]},{cores[1]}"
    )

    def gaps(record: RunRecord) -> tuple[str, ...]:
        target = targets[record.requested.environment_selector["runtime_environment"]]
        missing = list(
            evidence_gaps(
                record,
                runtime_pin=target["runtime"]["sha256"],
                trials=trials,
                quality_identity=quality,
            )
        )
        if _mapping(record.observed.get("hardware")).get("device_type") != "cpu":
            missing.append("qualified runtime upgrade requires observed CPU execution")
        placement = _mapping(record.observed.get("worker_placement"))
        affinity = placement.get("cpu_affinity")
        if (
            placement.get("device_type") != "cpu"
            or placement.get("affinity_source") != "linux_worker_thread_union"
            or not isinstance(affinity, (list, tuple))
            or tuple(affinity) != cores
        ):
            missing.append("native worker did not confirm the controlled CPU placement")
        if record.observed.get("runtime_environment") != target:
            missing.append("runtime environment was not independently confirmed")
        return tuple(missing)

    return VerificationRoute(
        scope=VLLM_UPGRADE_SCOPE,
        adapters={"vllm": VLLMAdapter()},
        measurements={TokenTrajectoryProtocol.name: PrefixWorkloadProtocol()},
        analyses={
            TrajectoryAgreementAnalysis.name: TrajectoryAgreementAnalysis(),
            VerificationImpactAnalysis.name: VerificationImpactAnalysis(),
        },
        evidence_gaps=gaps,
        observed_facts=observed_facts,
        change={
            "parameter": "runtime_environment",
            "reference": UPGRADE_VERSIONS[0],
            "candidate": UPGRADE_VERSIONS[1],
            "cpu_binding": list(cores),
        },
        performance=compare_prefix_performance,
        run_executor=RuntimeRunExecutor(
            targets,
            timeout_s=recipe.environment["verification_timeout_s"],
            binding=binding,
            cpu_ids=cores,
        ),
    )
