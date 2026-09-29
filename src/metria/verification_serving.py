"""Qualified local streaming serving API with one explicit concurrency change."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from functools import partial
from typing import Any

from .measurements import TokenTrajectoryProtocol, TrajectoryAgreementAnalysis
from .measurements.prefix_workload import trial_settings
from .measurements.serving_impact import ServingImpactAnalysis, compare_serving
from .measurements.serving_metrics import measure_serving
from .measurements.serving_workload import WORKLOAD_METHOD, ServingWorkloadProtocol
from .measurements.trajectory import _sha256_text
from .measurements.verification_impact import VerificationImpactAnalysis
from .models import RunRecord
from .recipes import StudyRecipe
from .runtimes.vllm_serving import ServingAdapter
from .verification_evidence import task_check_identity
from .verification_route import VerificationRoute
from .verification_schema import VLLM_SERVING_SCOPE
from .verification_vllm import (
    VLLM_VARIATIONS,
    identity_gaps,
    observed_facts,
    validate_vllm_recipe,
)
from .verification_vllm import (
    trial_identity as prefix_trials,
)

SERVING_CONTROLS = frozenset(
    {
        "model",
        "runtime",
        "scenario",
        "measurements",
        "trial_policy.method",
        "trial_policy.warmup_trials",
        "trial_policy.measured_trials",
    }
)
SERVING_VARIATIONS = frozenset(
    {"trial_policy.concurrency", "resolved.serving.concurrency"}
)


def trial_identity(config: Mapping[str, Any], concurrency: int) -> dict[str, Any]:
    warmup, measured = trial_settings(config)
    return {
        "method": WORKLOAD_METHOD,
        "warmup_trials": warmup,
        "measured_trials": measured,
        "concurrency": concurrency,
    }


def validate_serving_recipe(recipe: StudyRecipe) -> None:
    if recipe.study.comparison.control != SERVING_CONTROLS:
        raise ValueError("serving verification requires its complete fixed control set")
    if (
        len(recipe.study.runs) != 2
        or recipe.study.comparison.vary != SERVING_VARIATIONS
    ):
        raise ValueError(
            "serving verification requires its exact reference 1 to candidate 2 concurrency change"
        )
    config = recipe.measurement_configs.get(TokenTrajectoryProtocol.name, {})
    if len(config.get("prompts", ())) < 2:
        raise ValueError("serving verification requires at least two prompts")
    left, right = recipe.study.runs
    if (
        left.runtime != right.runtime
        or left.model != right.model
        or left.scenario != right.scenario
    ):
        raise ValueError(
            "serving verification holds model, engine capacity and generation controls fixed"
        )
    for run, concurrency in zip((left, right), (1, 2), strict=True):
        if (
            run.runtime.get("enable_prefix_caching") is not False
            or run.runtime.get("max_num_seqs") != 2
            or run.trial_policy != trial_identity(config, concurrency)
        ):
            raise ValueError(
                "serving verification requires capacity 2, caching off and exact trial controls"
            )
    analyses = recipe.study.comparison.analyses
    if analyses != (
        TrajectoryAgreementAnalysis.name,
        VerificationImpactAnalysis.name,
        ServingImpactAnalysis.name,
    ):
        raise ValueError(
            "serving verification requires trajectory, task and serving analyses"
        )
    # Validate the shared pinned vLLM/model/generation/environment contract using
    # its existing validator, after checking every serving-specific field above.
    canonical = tuple(
        replace(
            run,
            runtime={
                **run.runtime,
                "max_num_seqs": 1,
                "enable_prefix_caching": enabled,
            },
            trial_policy=prefix_trials(config),
        )
        for run, enabled in zip((left, right), (False, True), strict=True)
    )
    validate_vllm_recipe(
        replace(
            recipe,
            policy=None,
            study=replace(
                recipe.study,
                runs=canonical,
                comparison=replace(
                    recipe.study.comparison,
                    vary=VLLM_VARIATIONS,
                    analyses=analyses[:2],
                    control=frozenset(
                        {"model", "scenario", "measurements", "trial_policy"}
                    ),
                ),
            ),
        )
    )


def _workload_gaps(record: RunRecord, config: Mapping[str, Any]) -> tuple[str, ...]:
    if record.observed.get("serving") != {
        "engine_capacity": 2,
        "transport": "local AsyncLLM.generate",
    }:
        return ("native serving engine capacity or transport evidence is missing",)
    capture = record.evidence.get("measurements", {}).get(
        TokenTrajectoryProtocol.name, {}
    )
    workload = capture.get("workload", {})
    expected = record.requested.trial_policy
    if any(
        workload.get(key) != expected[key]
        for key in ("method", "warmup_trials", "measured_trials")
    ):
        return ("serving trial evidence is missing or inconsistent",)
    batches, memories = workload.get("batches", ()), workload.get("memory", ())
    count, warmups, trials = (
        len(config["prompts"]),
        expected["warmup_trials"],
        expected["measured_trials"],
    )
    if len(batches) != trials or len(memories) != trials:
        return ("serving trials are incomplete",)
    prompts = capture.get("prompts", ())
    invocations = record.observed.get("invocations", ())
    if len(prompts) != count * trials or len(invocations) != count * (warmups + trials):
        return ("serving native capture counts differ from the workload",)
    if any(row.get("finished") is not True for row in invocations):
        return ("native serving invocation did not complete",)
    for index, batch in enumerate(batches):
        if (
            batch.get("concurrency_limit") != expected["concurrency"]
            or batch.get("peak_inflight") != expected["concurrency"]
        ):
            return (
                "measured workload did not demonstrate the declared serving concurrency",
            )
        if len(batch.get("requests", ())) != count:
            return ("serving request traces are incomplete",)
        for offset, trace in enumerate(batch["requests"]):
            row, prompt = prompts[index * count + offset], config["prompts"][offset]
            native = invocations[(warmups + index) * count + offset]
            if (
                row.get("id") != f"{index}:{prompt['id']}"
                or row.get("prompt_sha256") != _sha256_text(prompt["prompt"])
                or native.get("prompt_sha256") != row.get("prompt_sha256")
                or trace.get("output_tokens") != row.get("token_count")
                or native.get("output_tokens") != row.get("token_count")
            ):
                return (
                    "streaming timing does not match retained native token and prompt evidence",
                )
    resets = record.observed.get("reset_events", ())
    if len(resets) != warmups + trials or any(
        row != {"scope": "prefix-cache", "mode": "public_prefix_cache_reset"}
        for row in resets
    ):
        return ("public reset confirmation is incomplete",)
    try:
        measured = measure_serving(batches, memories, prompts_per_trial=count)
    except (TypeError, ValueError, KeyError):
        return ("native serving measurement evidence is invalid",)
    if any(
        record.metrics.get(name) != value for name, value in measured.metrics.items()
    ):
        return ("serving metrics do not match native measurement evidence",)
    return ()


def evidence_gaps(
    record: RunRecord, *, runtime_pin: str, config: Mapping[str, Any]
) -> tuple[str, ...]:
    return (
        *identity_gaps(
            record,
            runtime_pin=runtime_pin,
            quality_identity=task_check_identity(config, trial_settings(config)[1]),
        ),
        *_workload_gaps(record, config),
    )


def build_route(recipe: StudyRecipe) -> VerificationRoute:
    validate_serving_recipe(recipe)
    measurement, trajectory, quality, serving = (
        ServingWorkloadProtocol(),
        TrajectoryAgreementAnalysis(),
        VerificationImpactAnalysis(),
        ServingImpactAnalysis(),
    )
    return VerificationRoute(
        scope=VLLM_SERVING_SCOPE,
        adapters={"vllm": ServingAdapter()},
        measurements={measurement.name: measurement},
        analyses={item.name: item for item in (trajectory, quality, serving)},
        evidence_gaps=partial(
            evidence_gaps,
            runtime_pin=recipe.environment["vllm_distribution_sha256"],
            config=recipe.measurement_configs[measurement.name],
        ),
        observed_facts=observed_facts,
        change={
            "parameter": "client_concurrency",
            "reference": 1,
            "candidate": 2,
            "engine_capacity": 2,
        },
        performance=compare_serving,
        isolated=True,
    )
