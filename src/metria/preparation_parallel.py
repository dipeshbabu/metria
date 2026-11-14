"""Build a STAGED single-device vs multi-device vLLM parallelism recipe."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .measurements import TokenTrajectoryProtocol
from .models import ComparisonPlan, RunSpec, StudySpec
from .recipes import StudyRecipe


def candidate_parallel(
    tensor: int | None, pipeline: int | None
) -> dict[str, int] | None:
    """Collect the candidate's requested sizes from command-line options."""

    if tensor is None and pipeline is None:
        return None
    sizes = {"tensor_parallel_size": tensor or 1}
    if pipeline is not None:
        sizes["pipeline_parallel_size"] = pipeline
    return sizes


def parallel_recipe(
    model: Mapping[str, Any],
    runtime: Mapping[str, Any],
    scenario: Mapping[str, Any],
    config: Mapping[str, Any],
    runtime_identity: Mapping[str, Any],
    timeout_s: float,
    candidate: Mapping[str, int],
) -> StudyRecipe:
    """Reference on one device; candidate with the requested parallel sizes."""

    from .verification_kv_precision import _analyses
    from .verification_parallelism import parallel_vary
    from .verification_vllm import trial_identity

    reference = {key: 1 for key in candidate}
    runs = tuple(
        RunSpec(
            model=model,
            runtime={**runtime, "enable_prefix_caching": False, **sizes},
            scenario=scenario,
            measurements=(TokenTrajectoryProtocol.name,),
            trial_policy=trial_identity(config),
        )
        for sizes in (reference, candidate)
    )
    return StudyRecipe(
        study=StudySpec(
            name="vllm-parallelism-change",
            runs=runs,
            comparison=ComparisonPlan(
                vary=parallel_vary(runs[0], runs[1]),
                control=frozenset(
                    {"model", "scenario", "measurements", "trial_policy"}
                ),
                analyses=_analyses(config),
            ),
        ),
        measurement_configs={TokenTrajectoryProtocol.name: config},
        environment={
            "vllm_distribution_sha256": runtime_identity["sha256"],
            "verification_timeout_s": timeout_s,
        },
    )
