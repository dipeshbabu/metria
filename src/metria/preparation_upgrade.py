"""Installed preparation for a pinned local CPU runtime-stack upgrade."""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, TextIO

from .measurements import TokenTrajectoryProtocol, TrajectoryAgreementAnalysis
from .measurements.verification_impact import VerificationImpactAnalysis
from .models import ComparisonPlan, RunSpec, StudySpec
from .policies import policy_from_data
from .recipes import StudyRecipe, study_recipe_digest, study_recipe_to_json
from .runtime_environment import (
    inspect_environment,
    source_digest,
    validate_environment,
)
from .runtimes.vllm_artifacts import verify_model_files
from .verification_upgrade import (
    ENVIRONMENTS_KEY,
    UPGRADE_VARIATIONS,
    validate_upgrade_recipe,
)
from .verification_vllm import trial_identity


def prepare_vllm_upgrade_recipe(
    model_dir: Path,
    descriptor: Mapping[str, Any],
    prompts: Sequence[Mapping[str, Any]],
    targets: Mapping[str, Any],
    *,
    context: int = 512,
    max_tokens: int = 16,
    warmup_trials: int = 1,
    measured_trials: int = 3,
    timeout_s: float = 900,
) -> StudyRecipe:
    directory = model_dir.expanduser().resolve()
    files = descriptor.get("files")
    verify_model_files(directory, files)
    if set(targets) != {"reference", "candidate"}:
        raise ValueError("reference and candidate runtime descriptors are required")
    code = source_digest()
    for target in targets.values():
        validate_environment(target, metria_sha256=code)
    cores = sorted(
        set(targets["reference"]["cpu_affinity"])
        & set(targets["candidate"]["cpu_affinity"])
    )[:2]
    if len(cores) != 2:
        raise ValueError("both CPU environments must allow at least two shared CPU IDs")
    config = {
        "prompts": list(prompts),
        "warmup_trials": warmup_trials,
        "measured_trials": measured_trials,
    }
    model: dict[str, Any] = {"path": str(directory), "files": files}
    for key, target in (("model_id", "id"), ("revision", "revision")):
        if descriptor.get(key) is not None:
            model[target] = descriptor[key]
    runtime = {
        "name": "vllm",
        "dtype": "float32",
        "gpu_memory_utilization": 0.35,
        "max_num_seqs": 1,
        "max_model_len": context,
        "tensor_parallel_size": 1,
        "enforce_eager": True,
        "enable_prefix_caching": False,
        "trust_remote_code": False,
    }
    scenario = {
        "context": context,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": 42,
        "chat_template": False,
    }
    runs = tuple(
        RunSpec(
            model=model,
            runtime={**runtime, "version": targets[role]["runtime"]["version"]},
            scenario=scenario,
            measurements=(TokenTrajectoryProtocol.name,),
            trial_policy=trial_identity(config),
            environment_selector={"runtime_environment": role},
        )
        for role in ("reference", "candidate")
    )
    recipe = StudyRecipe(
        study=StudySpec(
            name="vllm-cpu-runtime-upgrade",
            runs=runs,
            comparison=ComparisonPlan(
                vary=UPGRADE_VARIATIONS,
                control=frozenset(
                    {
                        "model",
                        "scenario",
                        "measurements",
                        "trial_policy",
                        "observed.hardware",
                        "observed.worker_placement",
                    }
                ),
                analyses=(
                    TrajectoryAgreementAnalysis.name,
                    VerificationImpactAnalysis.name,
                ),
            ),
        ),
        measurement_configs={TokenTrajectoryProtocol.name: config},
        environment={
            ENVIRONMENTS_KEY: targets,
            "verification_timeout_s": timeout_s,
            "cpu_binding": cores,
        },
    )
    validate_upgrade_recipe(recipe)
    return recipe


def prepare_upgrade_command(args: Any, stdout: TextIO) -> int:
    from .onboarding import _json, _read
    from .verification import _write_atomic

    if sys.platform != "linux":
        raise ValueError(
            "prepare runtime upgrades on Linux/WSL with both pinned CPU environments installed"
        )
    if args.output.exists():
        raise FileExistsError("choose a new recipe output path")
    descriptor = _json(_read(args.descriptor))
    if not isinstance(descriptor, dict):
        raise ValueError("model descriptor must be a trusted file manifest")
    prompts = [
        _json(line) for line in _read(args.workload).splitlines() if line.strip()
    ]
    from .measurements.prefix_workload import trajectory_config

    trajectory_config(
        {
            "prompts": prompts,
            "warmup_trials": args.warmup_trials,
            "measured_trials": args.measured_trials,
        }
    )
    policy = policy_from_data(_json(_read(args.policy))) if args.policy else None
    targets = {
        role: inspect_environment(python)
        for role, python in (
            ("reference", args.reference_python),
            ("candidate", args.candidate_python),
        )
    }
    recipe = prepare_vllm_upgrade_recipe(
        args.model,
        descriptor,
        prompts,
        targets,
        context=args.context,
        max_tokens=args.max_tokens,
        warmup_trials=args.warmup_trials,
        measured_trials=args.measured_trials,
        timeout_s=args.timeout,
    )
    recipe = replace(recipe, policy=policy)
    _write_atomic(args.output, study_recipe_to_json(recipe) + "\n")
    stdout.write(f"Prepared {args.output}\nRecipe: {study_recipe_digest(recipe)}\n")
    stdout.write(
        "Both interpreter and runtime installations are pinned; verification retains each environment's native evidence.\n"
    )
    return 0
