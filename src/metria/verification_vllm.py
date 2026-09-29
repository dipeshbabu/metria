"""Qualified local vLLM prefix-cache verification, using the shared lifecycle."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any

from .measurements import TokenTrajectoryProtocol, TrajectoryAgreementAnalysis
from .measurements.prefix_workload import (
    WORKLOAD_METHOD,
    PrefixWorkloadProtocol,
    trajectory_config,
    trial_settings,
)
from .measurements.task_checks import prompt_checks, task_workload_identity
from .measurements.verification_impact import VerificationImpactAnalysis, _quality
from .models import RunRecord
from .protocols import InferenceRequest
from .recipes import StudyRecipe
from .runtimes.vllm import (
    VLLMAdapter,
    _generation_options,
    _model_source,
    _runtime_config,
)
from .runtimes.vllm_artifacts import validate_file_pins
from .verification_route import VerificationRoute
from .verification_schema import VLLM_VERIFICATION_SCOPE as VLLM_SCOPE

VLLM_VARIATIONS = frozenset(
    {
        "runtime.enable_prefix_caching",
        "resolved.runtime.settings.enable_prefix_caching",
        "observed.configured.runtime.enable_prefix_caching",
        "observed.applied.fields.cache.enable_prefix_caching",
        "observed.identity.applied.fields.cache.enable_prefix_caching",
    }
)
QUALIFIED_VLLM_VERSIONS = frozenset({"0.30.0+cpu", "0.30.0+cu129"})


def trial_identity(config: Mapping[str, Any]) -> dict[str, Any]:
    warmup, measured = trial_settings(config)
    return {
        "method": WORKLOAD_METHOD,
        "warmup_trials": warmup,
        "measured_trials": measured,
        "schedule": "sequential",
        "isolation": "reset_prefix_cache_before_each_trial",
    }


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _validate_run(
    run: Any,
    config: Mapping[str, Any],
    *,
    versions: frozenset[str] = QUALIFIED_VLLM_VERSIONS,
) -> None:
    _model_source(run)
    if run.runtime.get("name") != "vllm" or run.runtime.get("version") not in versions:
        raise ValueError(
            "prefix verification requires a qualified vLLM 0.30.0 CPU/CUDA wheel version"
        )
    runtime = _runtime_config(run)
    required = {
        "name",
        "version",
        "dtype",
        "gpu_memory_utilization",
        "max_num_seqs",
        "max_model_len",
        "tensor_parallel_size",
        "enforce_eager",
        "enable_prefix_caching",
        "trust_remote_code",
    }
    if set(run.runtime) != required:
        raise ValueError(
            "prefix verification requires explicit qualified runtime settings; use the preparation flow"
        )
    if (
        runtime["trust_remote_code"]
        or not runtime["enforce_eager"]
        or runtime["tensor_parallel_size"] != 1
        or runtime["max_num_seqs"] != 1
    ):
        raise ValueError(
            "prefix verification requires eager, single-device, sequential local inference without remote code"
        )
    if runtime["dtype"] not in {"float16", "float32", "bfloat16", "auto"}:
        raise ValueError("unsupported verification model dtype")
    if set(run.model) - {"path", "id", "revision", "files"} or not isinstance(
        run.model.get("path"), str
    ):
        raise ValueError("prefix verification requires local model.path and file pins")
    validate_file_pins(run.model.get("files"))
    if not Path(run.model["path"]).expanduser().is_dir():
        raise ValueError("the pinned local model directory is unavailable")
    if (
        run.treatments
        or run.environment_selector
        or run.trial_policy != trial_identity(config)
    ):
        raise ValueError(
            "unsupported treatment/placement or mismatched prefix trial policy"
        )
    if set(run.scenario) - {
        "name",
        "context",
        "max_tokens",
        "seed",
        "temperature",
        "chat_template",
    }:
        raise ValueError("unsupported prefix-verification scenario fields")
    if run.measurements != (TokenTrajectoryProtocol.name,):
        raise ValueError("prefix verification requires the trajectory workload method")
    for row in trajectory_config(config)["prompts"]:
        options = _generation_options(
            InferenceRequest(
                row["prompt"],
                {**config.get("generation", {}), **row.get("generation", {})},
            ),
            run.scenario,
        )
        if (
            options["temperature"] != 0
            or options["chat_template"]
            or options["system"]
            or options["max_tokens"] <= 0
        ):
            raise ValueError(
                "prefix verification requires greedy, nonempty plain completion"
            )


def validate_vllm_recipe(recipe: StudyRecipe) -> None:
    if len(recipe.study.runs) != 2:
        raise ValueError("verify requires exactly two runs: reference, then candidate")
    comparison = recipe.study.comparison
    if comparison.vary != VLLM_VARIATIONS or comparison.waivers:
        raise ValueError(
            "prefix verification permits only its five prefix-cache variation paths and no waivers"
        )
    if comparison.analyses not in {
        (TrajectoryAgreementAnalysis.name,),
        (TrajectoryAgreementAnalysis.name, VerificationImpactAnalysis.name),
    }:
        raise ValueError(
            "prefix verification requires the registered trajectory analysis"
        )
    if set(recipe.measurement_configs) != {TokenTrajectoryProtocol.name}:
        raise ValueError(
            "prefix verification requires exactly one workload configuration"
        )
    if set(recipe.environment) != {
        "vllm_distribution_sha256",
        "verification_timeout_s",
    }:
        raise ValueError(
            "prefix verification requires the pinned runtime digest and a whole-verification timeout"
        )
    digest = recipe.environment["vllm_distribution_sha256"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(char not in "0123456789abcdef" for char in digest)
    ):
        raise ValueError("vllm_distribution_sha256 must be a lowercase SHA256 digest")
    timeout = recipe.environment["verification_timeout_s"]
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or not 1 <= timeout <= 3600
    ):
        raise ValueError(
            "verification_timeout_s must be finite and between 1 and 3600 seconds"
        )
    config = recipe.measurement_configs[TokenTrajectoryProtocol.name]
    trajectory_config(config)
    left, right = recipe.study.runs
    for run in (left, right):
        _validate_run(run, config)
    if left.model != right.model or left.scenario != right.scenario:
        raise ValueError(
            "prefix verification requires identical model and scenario inputs"
        )
    if {k: v for k, v in left.runtime.items() if k != "enable_prefix_caching"} != {
        k: v for k, v in right.runtime.items() if k != "enable_prefix_caching"
    }:
        raise ValueError("only the prefix-cache runtime setting may differ")
    if (
        left.runtime["enable_prefix_caching"] is not False
        or right.runtime["enable_prefix_caching"] is not True
    ):
        raise ValueError(
            "prefix verification requires caching off in the reference and on in the candidate"
        )


def observed_facts(record: RunRecord) -> dict[str, Any]:
    artifacts = _mapping(record.observed.get("artifacts"))
    identity = _mapping(record.observed.get("identity"))
    fields = _mapping(_mapping(identity.get("applied")).get("fields"))
    return {
        "runtime_hardware": record.observed.get("hardware"),
        "model_sha256": _mapping(artifacts.get("model")).get("sha256"),
        "provider_sha256": _mapping(artifacts.get("runtime")).get("sha256"),
        "prefix_caching": fields.get("cache.enable_prefix_caching"),
        "context": fields.get("model.max_model_len"),
        "runtime_version": _mapping(identity.get("runtime")).get("version"),
    }


def _workload_gaps(record: RunRecord, expected_trials: Mapping[str, Any]) -> list[str]:
    capture = _mapping(
        _mapping(record.evidence.get("measurements")).get(TokenTrajectoryProtocol.name)
    )
    workload = _mapping(capture.get("workload"))
    if workload.get("method") != WORKLOAD_METHOD or any(
        workload.get(key) != expected_trials[key]
        for key in ("warmup_trials", "measured_trials")
    ):
        return ["declared prefix trial evidence is missing or inconsistent"]
    samples = workload.get("samples", ())
    invocations = record.observed.get("invocations", ())
    if (
        not isinstance(samples, Sequence)
        or isinstance(samples, (str, bytes))
        or not samples
        or not isinstance(invocations, Sequence)
        or len(samples) != len(invocations)
    ):
        return ["native invocation and workload sample counts differ or are missing"]
    measured = [
        row
        for row in samples
        if isinstance(row, Mapping) and row.get("measured") is True
    ]
    if not measured or len(measured) != capture.get("n_prompts"):
        return ["measured trial samples are incomplete"]
    for invocation, sample in zip(invocations, samples, strict=True):
        if (
            not isinstance(invocation, Mapping)
            or not isinstance(sample, Mapping)
            or invocation.get("finished") is not True
        ):
            return ["completed native request evidence is missing"]
        prompt, cached = sample.get("prompt_tokens"), sample.get("cached_tokens")
        if (
            isinstance(prompt, bool)
            or not isinstance(prompt, int)
            or isinstance(cached, bool)
            or not isinstance(cached, int)
            or prompt <= 0
            or not 0 <= cached <= prompt
        ):
            return ["native prompt/cache token counts are missing or invalid"]
        if prompt != invocation.get("prompt_tokens") or cached != invocation.get(
            "cached_tokens"
        ):
            return ["workload cache counts do not match native runtime evidence"]
    cached_values = [row["cached_tokens"] for row in measured]
    enabled = record.requested.runtime["enable_prefix_caching"]
    if enabled and not any(value > 0 for value in cached_values):
        return [
            "the measured workload did not demonstrate a candidate prefix-cache hit"
        ]
    if not enabled and any(value != 0 for value in cached_values):
        return ["the reference unexpectedly reused cached prefix tokens"]
    resets = record.observed.get("reset_events", ())
    expected_resets = (
        expected_trials["warmup_trials"] + expected_trials["measured_trials"]
    )
    if (
        not isinstance(resets, Sequence)
        or isinstance(resets, (str, bytes))
        or len(resets) != expected_resets
        or any(
            row != {"scope": "prefix-cache", "mode": "public_prefix_cache_reset"}
            for row in resets
        )
    ):
        return ["public cache reset confirmation is incomplete"]
    return []


def identity_gaps(
    record: RunRecord,
    *,
    runtime_pin: str,
    quality_identity: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    gaps = []
    hardware = _mapping(record.observed.get("hardware"))
    if hardware.get("status") != "observed" or hardware.get("device_type") not in {
        "cpu",
        "cuda",
    }:
        gaps.append("native runtime device identity is unavailable")
    identity = _mapping(record.observed.get("identity"))
    artifacts = _mapping(record.observed.get("artifacts"))
    model = _mapping(artifacts.get("model"))
    runtime = _mapping(artifacts.get("runtime"))
    resolved_model = _mapping(record.resolved.get("model"))
    if identity.get("status") == "mismatch":
        gaps.append("runtime identity is mismatched")
    if _mapping(record.observed.get("applied")).get("status") != "introspected":
        gaps.append("runtime-applied state was not independently observed")
    if (
        model.get("status") != "verified"
        or not model.get("sha256")
        or model.get("sha256")
        != _mapping(resolved_model.get("artifacts")).get("sha256")
    ):
        gaps.append("pinned model and tokenizer content were not verified after launch")
    if runtime.get("status") != "verified" or runtime.get("sha256") != runtime_pin:
        gaps.append(
            "loaded runtime content does not match the qualified installation pin"
        )
    if _mapping(identity.get("runtime")).get("version") != record.requested.runtime.get(
        "version"
    ):
        gaps.append("loaded runtime version differs from the requested wheel")
    enabled = record.requested.runtime.get("enable_prefix_caching")
    for applied in (record.observed.get("applied"), identity.get("applied")):
        fields = _mapping(_mapping(applied).get("fields"))
        if fields.get("cache.enable_prefix_caching") is not enabled:
            gaps.append("runtime-applied prefix-cache state does not match the request")
        if fields.get("model.max_model_len") != record.requested.runtime.get(
            "max_model_len"
        ):
            gaps.append("runtime-applied context differs from the request")
    coverage = record.metrics.get("trajectory_nonempty_capture_rate")
    if coverage is None or isinstance(coverage.value, bool) or coverage.value != 1.0:
        gaps.append("every measured prompt must retain nonempty sampled token IDs")
    if quality_identity is not None:
        quality = _quality(record)
        if (
            quality is None
            or quality.get("workload_sha256") != quality_identity["sha256"]
            or quality.get("check_count") != quality_identity["check_count"]
        ):
            gaps.append("declared task-check evidence is missing or inconsistent")
    return tuple(dict.fromkeys(gaps))


def evidence_gaps(
    record: RunRecord,
    *,
    runtime_pin: str,
    trials: Mapping[str, Any],
    quality_identity: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            (
                *identity_gaps(
                    record, runtime_pin=runtime_pin, quality_identity=quality_identity
                ),
                *_workload_gaps(record, trials),
            )
        )
    )


def build_route(recipe: StudyRecipe) -> VerificationRoute:
    from .measurements.prefix_performance import compare_prefix_performance

    validate_vllm_recipe(recipe)
    measurement = PrefixWorkloadProtocol()
    analysis = TrajectoryAgreementAnalysis()
    trials = trial_identity(recipe.measurement_configs[measurement.name])
    config = recipe.measurement_configs[measurement.name]
    checks = prompt_checks(config["prompts"])
    count = sum(len(row) for row in checks) * trials["measured_trials"]
    quality_identity = (
        {
            "sha256": task_workload_identity(
                checks, config["prompts"], trials["measured_trials"]
            ),
            "check_count": count,
        }
        if count
        else None
    )
    return VerificationRoute(
        scope=VLLM_SCOPE,
        adapters={"vllm": VLLMAdapter()},
        measurements={measurement.name: measurement},
        analyses={
            analysis.name: analysis,
            VerificationImpactAnalysis.name: VerificationImpactAnalysis(),
        },
        evidence_gaps=partial(
            evidence_gaps,
            runtime_pin=recipe.environment["vllm_distribution_sha256"],
            trials=trials,
            quality_identity=quality_identity,
        ),
        observed_facts=observed_facts,
        change={
            "parameter": "enable_prefix_caching",
            "reference": False,
            "candidate": True,
        },
        performance=compare_prefix_performance,
        isolated=True,
    )
