"""Build explicit qualified recipes from local immutable artifact inputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .measurements import TokenTrajectoryProtocol, TrajectoryAgreementAnalysis
from .measurements.verification_impact import VerificationImpactAnalysis
from .models import ComparisonPlan, RunSpec, StudySpec
from .recipes import StudyRecipe
from .runtimes.vllm_artifacts import installed_runtime_identity, verify_model_files
from .verification_vllm import VLLM_VARIATIONS, trial_identity, validate_vllm_recipe


def prepare_vllm_prefix_recipe(
    model_dir: str | Path,
    descriptor: Mapping[str, Any],
    prompts: Sequence[Mapping[str, Any]],
    *,
    context: int = 512,
    max_tokens: int = 16,
    warmup_trials: int = 1,
    measured_trials: int = 3,
    timeout_s: float = 900,
    gpu_memory_utilization: float = 0.35,
    kv_cache_dtype: str | None = None,
) -> StudyRecipe:
    """Pin current runtime content and verify a separately trusted model manifest."""
    directory = Path(model_dir).expanduser().resolve()
    files = descriptor.get("files")
    verify_model_files(directory, files)
    runtime_identity = installed_runtime_identity()
    version = runtime_identity["version"]
    config = {
        "prompts": tuple(prompts),
        "warmup_trials": warmup_trials,
        "measured_trials": measured_trials,
    }
    model: dict[str, Any] = {"path": str(directory), "files": files}
    for source, target in (("model_id", "id"), ("revision", "revision")):
        if descriptor.get(source) is not None:
            model[target] = descriptor[source]
    runtime = {
        "name": "vllm",
        "version": version,
        "dtype": "float32" if version.endswith("+cpu") else "float16",
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_num_seqs": 1,
        "max_model_len": context,
        "tensor_parallel_size": 1,
        "enforce_eager": True,
        "trust_remote_code": False,
    }
    scenario = {
        "context": context,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": 42,
        "chat_template": False,
    }
    if kv_cache_dtype is not None:
        return _kv_precision_recipe(
            model,
            runtime,
            scenario,
            config,
            runtime_identity,
            timeout_s,
            kv_cache_dtype,
        )
    runs = tuple(
        RunSpec(
            model=model,
            runtime={**runtime, "enable_prefix_caching": enabled},
            scenario=scenario,
            measurements=(TokenTrajectoryProtocol.name,),
            trial_policy=trial_identity(config),
        )
        for enabled in (False, True)
    )
    recipe = StudyRecipe(
        study=StudySpec(
            name="vllm-prefix-cache-change",
            runs=runs,
            comparison=ComparisonPlan(
                vary=VLLM_VARIATIONS,
                control=frozenset(
                    {"model", "scenario", "measurements", "trial_policy"}
                ),
                analyses=(
                    TrajectoryAgreementAnalysis.name,
                    VerificationImpactAnalysis.name,
                ),
            ),
        ),
        measurement_configs={TokenTrajectoryProtocol.name: config},
        environment={
            "vllm_distribution_sha256": runtime_identity["sha256"],
            "verification_timeout_s": timeout_s,
        },
    )
    validate_vllm_recipe(recipe)
    return recipe


def _kv_precision_recipe(
    model: Mapping[str, Any],
    runtime: Mapping[str, Any],
    scenario: Mapping[str, Any],
    config: Mapping[str, Any],
    runtime_identity: Mapping[str, Any],
    timeout_s: float,
    kv_cache_dtype: str,
) -> StudyRecipe:
    """Build an automatic-vs-FP8 KV-cache recipe; validated once a policy is set."""

    from .models import TreatmentSpec, TreatmentType
    from .verification_kv_precision import FP8_KV_DTYPES, KV_PRECISION_VARIATIONS

    if kv_cache_dtype not in FP8_KV_DTYPES:
        raise ValueError(
            "kv_cache_dtype must be one of: " + ", ".join(sorted(FP8_KV_DTYPES))
        )
    runs = tuple(
        RunSpec(
            model=model,
            runtime={**runtime, "enable_prefix_caching": False},
            scenario=scenario,
            measurements=(TokenTrajectoryProtocol.name,),
            trial_policy=trial_identity(config),
            treatments=(
                TreatmentSpec(
                    name="vllm.kv_cache",
                    kind=TreatmentType.RUNTIME_FEATURE,
                    config={"dtype": dtype},
                ),
            ),
        )
        for dtype in ("auto", kv_cache_dtype)
    )
    return StudyRecipe(
        study=StudySpec(
            name="vllm-kv-cache-precision-change",
            runs=runs,
            comparison=ComparisonPlan(
                vary=KV_PRECISION_VARIATIONS,
                control=frozenset(
                    {"model", "scenario", "measurements", "trial_policy"}
                ),
                analyses=(
                    TrajectoryAgreementAnalysis.name,
                    VerificationImpactAnalysis.name,
                ),
            ),
        ),
        measurement_configs={TokenTrajectoryProtocol.name: config},
        environment={
            "vllm_distribution_sha256": runtime_identity["sha256"],
            "verification_timeout_s": timeout_s,
        },
    )
