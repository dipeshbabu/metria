"""A strict CPU comparison between independently qualified llama.cpp builds."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .measurements import TokenTrajectoryProtocol, TrajectoryAgreementAnalysis
from .measurements.verification_impact import VerificationImpactAnalysis
from .models import RunRecord
from .recipes import StudyRecipe
from .runtimes.llamacpp_qualified import PROVIDERS_KEY, QualifiedLlamaCppAdapter
from .verification_cpu import validate_cpu_run
from .verification_route import VerificationRoute
from .verification_schema import LLAMACPP_BUILD_SCOPE

BUILD_VARIATIONS = frozenset(
    {
        "runtime.bin_dir",
        "resolved.runtime.bin_dir",
        "resolved.runtime.cli",
        "resolved.runtime.completion",
        "observed.runtime.bin_dir",
        "observed.runtime.cli",
        "observed.runtime.completion",
        "observed.identity.runtime.cli_sha256",
        "observed.identity.runtime.completion_sha256",
    }
)


def _providers(recipe: StudyRecipe) -> dict[str, str]:
    if set(recipe.environment) != {PROVIDERS_KEY}:
        raise ValueError(
            "build verification requires only its two qualified provider pins"
        )
    providers = recipe.environment[PROVIDERS_KEY]
    if not isinstance(providers, Mapping) or len(providers) != 2:
        raise ValueError("build verification requires exactly two provider pins")
    result = {}
    for directory, digest in providers.items():
        if (
            not isinstance(directory, str)
            or not directory
            or str(Path(directory).expanduser().resolve()) != directory
        ):
            raise ValueError("provider directories must be canonical absolute paths")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError("each capture provider needs a lowercase SHA256 pin")
        result[directory] = digest
    if len(set(result.values())) != 2:
        raise ValueError("a build comparison requires two different provider contents")
    return result


def validate_build_recipe(recipe: StudyRecipe, registries: Any) -> dict[str, str]:
    if len(recipe.study.runs) != 2:
        raise ValueError("build verification requires reference and candidate runs")
    comparison = recipe.study.comparison
    if comparison.vary != BUILD_VARIATIONS or comparison.waivers:
        raise ValueError(
            "build verification permits only its binary paths and no waivers"
        )
    if comparison.analyses not in {
        (TrajectoryAgreementAnalysis.name,),
        (TrajectoryAgreementAnalysis.name, VerificationImpactAnalysis.name),
    }:
        raise ValueError("build verification requires registered trajectory analysis")
    if set(recipe.measurement_configs) != {TokenTrajectoryProtocol.name}:
        raise ValueError("build verification requires one trajectory workload")
    providers = _providers(recipe)
    config = recipe.measurement_configs[TokenTrajectoryProtocol.name]
    for index, run in enumerate(recipe.study.runs):
        validate_cpu_run(run, index, config, registries)
        if run.runtime.get("bin_dir") not in providers:
            raise ValueError("each run must name its pinned provider directory")
    left, right = recipe.study.runs
    if left.model != right.model or left.scenario != right.scenario:
        raise ValueError(
            "build verification requires identical model and scenario inputs"
        )
    left_runtime = {
        key: value for key, value in left.runtime.items() if key != "bin_dir"
    }
    right_runtime = {
        key: value for key, value in right.runtime.items() if key != "bin_dir"
    }
    if left_runtime != right_runtime:
        raise ValueError("only the provider binary directory may change")
    if left.runtime["bin_dir"] == right.runtime["bin_dir"]:
        raise ValueError("reference and candidate must use different qualified builds")
    return providers


def build_route(recipe: StudyRecipe) -> VerificationRoute:
    from .verification import (
        _builtin_registries,
        _evidence_gaps,
        _legacy_performance,
        _observed_facts,
    )

    registries = _builtin_registries()
    providers = validate_build_recipe(recipe, registries)

    def gaps(record: RunRecord) -> tuple[str, ...]:
        return _evidence_gaps(record, providers[record.requested.runtime["bin_dir"]])

    left, right = recipe.study.runs
    return VerificationRoute(
        scope=LLAMACPP_BUILD_SCOPE,
        adapters={"llamacpp": QualifiedLlamaCppAdapter(providers)},
        measurements=registries.measurements,
        analyses=registries.analyses,
        evidence_gaps=gaps,
        observed_facts=_observed_facts,
        change={
            "parameter": "capture_provider_sha256",
            "reference": providers[left.runtime["bin_dir"]],
            "candidate": providers[right.runtime["bin_dir"]],
        },
        performance=_legacy_performance,
    )
